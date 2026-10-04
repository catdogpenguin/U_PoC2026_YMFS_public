"""render.py — 選択した解を可視化するラッパ。

描画そのものは既存の `src/visualization.py` に任せ、
ここでは「設計ベクトル → シミュレーション → 画像ファイル」の
段取りとキャッシュ管理だけを行う。src/ には手を入れない。
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import replace
from functools import wraps
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.constraints import design_variable_count, to_integer_layout  # noqa: E402
from src.models import Config  # noqa: E402
from src.parallel import StaticContext, build_context, evaluate_layout  # noqa: E402
from src.venue import apply_layout  # noqa: E402
from src.visualization import (  # noqa: E402
    draw_venue,
    plot_heatmaps,
    render_animation,
)

#: 可視化用シミュレーションのタイムアウト(秒)。
#: 最適化中は §17.6 の打ち切りが効くが、可視化では途中で切れると
#: 図が意味をなさないため十分に長くとる。
VIZ_TIMEOUT_SECONDS = 3600.0

#: 描画の版。`ui_cache/` のファイル名に混ぜてキャッシュを世代管理する。
#: src/visualization.py の見た目を変えたら +1 する(古い画像の再利用を防ぐ)。
#: 2: テーブル・椅子を灰色の矩形から fig/table.png のアイコンに変更。
#: 3: 選択解の実際の歩行可能セルを重ねた通路図を追加。
RENDER_VERSION = 11

VENUE_FIGURE_WIDTH = 9.0
VENUE_AXES_LEFT = 0.02
VENUE_AXES_RIGHT = 0.70
VENUE_AXES_BOTTOM = 0.02
VENUE_AXES_TOP = 0.96
VENUE_LEGEND_FONTSIZE = 16

# Matplotlib はスレッドセーフではない。Streamlit の複数セッションや
# キャッシュ計算が重なっても描画状態が混線しないよう、画像生成を直列化する。
_RENDER_LOCK = threading.RLock()


def _serialized_render(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        with _RENDER_LOCK:
            return func(*args, **kwargs)

    return wrapped


def viz_context(cfg: Config) -> StaticContext:
    """可視化専用の静的コンテキスト(タイムアウトのみ緩める)。"""
    return build_context(replace(cfg, timeout_seconds=VIZ_TIMEOUT_SECONDS))


def design_key(x) -> str:
    """設計ベクトルからキャッシュ用の短いハッシュを作る。"""
    s = ",".join(str(int(v)) for v in x)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def decode_positions(cfg: Config, x) -> tuple[tuple[int, int, int], ...]:
    """新120変数と保存済み旧3変数/配置物の両方を内部配置へ戻す。

    現行形式は通常ブース ``(x,y,theta)``、テーブル ``(x,y)`` なので、
    必ず cfg を渡して復元する。旧UIが保存した比較基準だけは全配置物を
    ``(x,y,theta)`` としたベクトルのため、履歴表示用に後方互換対応する。
    """
    values = tuple(x)
    if len(values) == design_variable_count(cfg):
        return to_integer_layout(values, cfg)
    if len(values) == len(cfg.booths) * 3:
        positions = to_integer_layout(values)
        return tuple(
            (int(px), int(py), 0 if booth.is_table else int(theta))
            for booth, (px, py, theta) in zip(cfg.booths, positions)
        )
    raise ValueError(
        "unsupported design vector length for visualization: "
        f"expected {design_variable_count(cfg)} (current) or "
        f"{len(cfg.booths) * 3} (legacy), got {len(values)}"
    )


# ---------------------------------------------------------------- レイアウト図


def _venue_figure(cfg: Config):
    """会場の縦横比と凡例領域を考慮した共通キャンバスを作る。"""
    axes_width = VENUE_FIGURE_WIDTH * (VENUE_AXES_RIGHT - VENUE_AXES_LEFT)
    axes_height_fraction = VENUE_AXES_TOP - VENUE_AXES_BOTTOM
    venue_aspect = cfg.grid_height / max(cfg.grid_width, 1)
    figure_height = axes_width * venue_aspect / axes_height_fraction
    return plt.subplots(figsize=(VENUE_FIGURE_WIDTH, figure_height))


def _replace_venue_legend(
    ax,
    extra_handles: list | None = None,
    extra_labels: list[str] | None = None,
) -> None:
    """会場図2種類の凡例スタイルを同じ設定で作り直す。"""
    legend = ax.get_legend()
    handles = []
    labels = []
    if legend is not None:
        handles = list(
            getattr(legend, "legend_handles", getattr(legend, "legendHandles", []))
        )
        labels = [text.get_text() for text in legend.get_texts()]
        legend.remove()

    handles.extend(extra_handles or [])
    labels.extend(extra_labels or [])
    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        bbox_to_anchor=(0.97, 1.0),
        fontsize=VENUE_LEGEND_FONTSIZE,
        framealpha=0.9,
        labelspacing=0.5,
        handlelength=1.6,
        handleheight=0.8,
        borderpad=0.5,
    )


def _save_venue_figure(fig, path: Path) -> Path:
    """凡例の項目数に左右されない共通サイズで会場図を保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(
        left=VENUE_AXES_LEFT,
        right=VENUE_AXES_RIGHT,
        top=VENUE_AXES_TOP,
        bottom=VENUE_AXES_BOTTOM,
    )
    # 別セッションが書き込み途中のPNGを読まないよう、完成後に原子的に置き換える。
    temporary = path.with_name(
        f".{path.stem}.{threading.get_ident()}.tmp{path.suffix}"
    )
    try:
        fig.savefig(temporary, dpi=130)
        temporary.replace(path)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()
    return path


@_serialized_render
def render_layout(cfg: Config, ctx: StaticContext, x, path: Path) -> Path:
    """会場レイアウト図(ブース・受付・退出・待機列)を PNG で保存する。"""
    positions = decode_positions(cfg, x)
    layout = apply_layout(cfg, ctx.venue, positions)

    fig, ax = _venue_figure(cfg)
    draw_venue(ax, cfg, layout, legend=True)
    _replace_venue_legend(ax)
    ax.set_anchor("N")
    ax.set_title("会場レイアウト")
    return _save_venue_figure(fig, path)


# ---------------------------------------------------------------- 歩行可能通路


@_serialized_render
def render_walkable_corridors(
    cfg: Config,
    ctx: StaticContext,
    x,
    path: Path,
) -> Path:
    """選択解のブース配置に実際の歩行可能セルを重ねて PNG 保存する。

    独自に通路を推定せず、``apply_layout`` が生成した ``layout.walkable``
    をそのまま使う。したがって、4→3→2→1 セル幅の順に各ブースへの
    到達経路を確保する現行の会場生成ロジックと表示結果が一致する。
    """
    positions = decode_positions(cfg, x)
    layout = apply_layout(cfg, ctx.venue, positions)

    fig, ax = _venue_figure(cfg)
    draw_venue(ax, cfg, layout, legend=True)

    # RGBA 配列を使い、歩行可能セルだけを半透明の青で塗る。
    # ブース番号・受付・退出記号を読める程度の薄さに固定する。
    overlay = np.zeros((*layout.walkable.shape, 4), dtype=float)
    overlay[np.asarray(layout.walkable, dtype=bool)] = (0.10, 0.55, 0.95, 0.28)
    ax.imshow(
        overlay,
        origin="upper",
        interpolation="nearest",
        extent=(-0.5, cfg.grid_width - 0.5, cfg.grid_height - 0.5, -0.5),
        zorder=3,
    )

    # draw_venue の既存凡例を保ち、歩行可能セルだけを追記する。
    _replace_venue_legend(
        ax,
        extra_handles=[
            Patch(
                facecolor=(0.10, 0.55, 0.95, 0.28),
                edgecolor="#1976d2",
                label="歩行可能セル",
            )
        ],
        extra_labels=["歩行可能セル"],
    )
    ax.set_anchor("N")

    ax.set_title("歩行可能通路")
    return _save_venue_figure(fig, path)


# ---------------------------------------------------------------- ヒートマップ


@_serialized_render
def render_heatmap(cfg: Config, ctx: StaticContext, x, path: Path) -> tuple[Path, dict]:
    """累積滞在人数のヒートマップを PNG で保存する。

    戻り値は (画像パス, KPI dict)。
    """
    positions = decode_positions(cfg, x)
    result = evaluate_layout(ctx, positions, collect_heatmap=True)
    layout = apply_layout(cfg, ctx.venue, positions)

    path.parent.mkdir(parents=True, exist_ok=True)
    plot_heatmaps(
        cfg,
        layout,
        result.heatmap,
        result.over_threshold_map,
        path,
        title="選択解",
        show_over_map=False,
    )

    kpis = dict(result.kpis)
    kpis["f_move"] = float(result.f_move)
    kpis["f_congestion"] = float(result.f_congestion)
    kpis["timed_out"] = bool(result.timed_out)
    return path, kpis


# ---------------------------------------------------------------- 時系列

def simulate_series(cfg: Config, ctx: StaticContext, x) -> dict:
    """1 解をシミュレーションし、混雑度・移動負担の時系列を返す。

    戻り値:
        {"time_seconds": [...], "congestion": [...], "movement": [...],
         "f_move": float, "f_congestion": float, "timed_out": bool}

    time_seconds はステップ番号に cfg.dt_seconds を掛けた実時間(秒)。
    """
    positions = decode_positions(cfg, x)
    result = evaluate_layout(ctx, positions)

    steps = list(result.time_steps)
    cong = list(result.congestion_history)
    move = list(result.movement_history)

    # 3 本の長さは揃っているはずだが、途中打ち切り時のずれに備えて短い方に合わせる。
    n = min(len(steps), len(cong), len(move))
    dt = float(cfg.dt_seconds)

    cong = [float(cong[i]) for i in range(n)]

    # congestion_history は「歩行可能セルあたりの重み付き平均」で、
    # f_congestion(= 重み付き総和 / n_steps)とは正規化が違う。
    # 曲線の時間平均が f_congestion に一致するようスケールを揃えておくと、
    # パレート図の縦軸と直接見比べられる。
    total = sum(cong)
    if n and total > 0.0:
        scale = float(result.f_congestion) * n / total
    else:
        scale = 1.0

    return {
        "time_seconds": [float(steps[i]) * dt for i in range(n)],
        "congestion": [c * scale for c in cong],
        "congestion_raw": cong,
        "movement": [float(move[i]) for i in range(n)],
        "f_move": float(result.f_move),
        "f_congestion": float(result.f_congestion),
        "timed_out": bool(result.timed_out),
    }


# ---------------------------------------------------------------- アニメーション


@_serialized_render
def render_agent_animation(
    cfg: Config,
    ctx: StaticContext,
    x,
    path: Path,
    frame_interval: int = 5,
    fps: int = 8,
) -> Path:
    """エージェントの時間発展アニメーションを GIF で保存する。

    frame_interval を大きくするほどフレーム数が減り、生成が速くなる。
    """
    positions = decode_positions(cfg, x)
    result = evaluate_layout(
        ctx,
        positions,
        record_frames=True,
        frame_interval=max(1, int(frame_interval)),
    )
    layout = apply_layout(cfg, ctx.venue, positions)

    path.parent.mkdir(parents=True, exist_ok=True)
    render_animation(cfg, layout, result.frames, path, fps=fps)
    return path


# ---------------------------------------------------------------- 会場プレビュー


@_serialized_render
def render_empty_venue(cfg: Config, path: Path) -> Path:
    """ブースを置く前の会場だけを描く(入力パラメータの確認用)。

    最適化を回さずに「この会場サイズで合っているか」を確かめられるようにする。
    """
    from src.venue import build_base_venue

    venue = build_base_venue(cfg)
    # ブース 0 個の Layout は作れないため、grid をそのまま描く。
    fig, ax = plt.subplots(figsize=(7, 7 * cfg.grid_height / max(cfg.grid_width, 1)))
    ax.imshow(venue.base_grid, cmap="Pastel1", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"会場 {cfg.grid_width} × {cfg.grid_height} セル")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
