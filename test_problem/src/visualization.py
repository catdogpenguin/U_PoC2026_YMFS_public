"""visualization.py — 最適化終了後の可視化(§20)。最適化中は使用しない(§17.1)。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 日本語ラベル用フォント(存在するもののみ適用)
for _f in ("Noto Sans CJK JP", "IPAexGothic", "Hiragino Sans", "Yu Gothic", "Meiryo"):
    try:
        import matplotlib.font_manager as _fm

        if any(f.name == _f for f in _fm.fontManager.ttflist):
            plt.rcParams["font.family"] = _f
            break
    except Exception:  # pragma: no cover
        break
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .models import (
    BOOTH,
    CHOOSING,
    ENTRANCE,
    EXIT,
    EXITING,
    EXPERIENCING,
    FORBIDDEN,
    QUEUE,
    QUEUING,
    WALKING,
    Config,
)
from .venue import Layout

#: §20.1 エージェント色分け(赤=移動不可待機は blocked_count で近似表示)
STATE_COLORS = {
    CHOOSING: "#1f77b4",
    WALKING: "#1f77b4",   # 青: ブースへ移動中
    QUEUING: "#ffcc00",   # 黄: 待ち行列
    EXPERIENCING: "#2ca02c",  # 緑: 体験中
    EXITING: "#9467bd",   # 紫: 出口へ移動中
}
BLOCKED_COLOR = "#d62728"  # 赤: セル容量により移動できず待機中

_CELL_FACE = {
    BOOTH: "#8c564b",
    QUEUE: "#f7e8a0",
    ENTRANCE: "#a6d8a8",
    EXIT: "#f4a9a8",
    FORBIDDEN: "#e6e6e6",
}


def draw_venue(ax, cfg: Config, layout: Layout) -> None:
    """会場・ブース・受付・行列セルの静的描画。"""
    H, W = cfg.grid_height, cfg.grid_width
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)  # y は下へ増加(§3.5)
    ax.set_aspect("equal")
    grid = layout.grid
    for ctype, color in _CELL_FACE.items():
        ys, xs = np.where(grid == ctype)
        for x, y in zip(xs, ys):
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color, edgecolor="none"))
    for b, (bx, by) in enumerate(layout.positions):
        spec = cfg.booths[b]
        ax.add_patch(
            Rectangle(
                (bx - 0.5, by - 0.5), spec.width, spec.height,
                fill=False, edgecolor="black", linewidth=1.2,
            )
        )
        ax.text(bx + spec.width / 2 - 0.5, by + spec.height / 2 - 0.5, f"B{b + 1}",
                ha="center", va="center", fontsize=9)
        rx, ry = layout.receptions[b]
        ax.plot(rx, ry, marker="v", color="black", markersize=5)
        ex, ey = layout.booth_exits[b]
        ax.plot(ex, ey, marker="^", color="black", markersize=5)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_pareto_front(
    F: np.ndarray,
    reps: dict[str, int],
    path: str | Path,
    initial_f: np.ndarray | None = None,
) -> None:
    """
    最終パレート解、代表解、および比較用の初期解を描画する。

    Parameters
    ----------
    F
        最終パレート解の目的関数値。
        各行は [f_move, f_congestion]。
    reps
        代表解名とF内のインデックスの対応。
    path
        グラフの保存先。
    initial_f
        初期解の目的関数値 [f_move, f_congestion]。
        Noneの場合は初期解を描画しない。
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # ---------------------------------------------------------
    # 最終パレート解
    # ---------------------------------------------------------
    ax.scatter(
        F[:, 0],
        F[:, 1],
        c="tab:blue",
        s=30,
        alpha=0.75,
        label="Pareto solutions",
        zorder=2,
    )

    # ---------------------------------------------------------
    # 代表解
    # ---------------------------------------------------------
    marks = {
        "best_movement": (
            "s",
            "tab:green",
            "Movement-oriented solution",
        ),
        "best_congestion": (
            "^",
            "tab:red",
            "Congestion-oriented solution",
        ),
        "knee": (
            "*",
            "tab:orange",
            "Balanced solution (knee)",
        ),
    }

    for name, idx in reps.items():
        if name not in marks:
            continue

        marker, color, label = marks[name]

        ax.scatter(
            F[idx, 0],
            F[idx, 1],
            marker=marker,
            c=color,
            s=150,
            edgecolors="black",
            linewidths=0.8,
            label=label,
            zorder=4,
        )

    # ---------------------------------------------------------
    # 比較に使用した初期解
    # ---------------------------------------------------------
    if initial_f is not None:
        initial_f = np.asarray(initial_f, dtype=float).reshape(-1)

        if initial_f.size != 2:
            raise ValueError(
                "initial_f must contain exactly two values: "
                "[f_move, f_congestion]."
            )

        if not np.all(np.isfinite(initial_f)):
            raise ValueError(
                f"initial_f contains non-finite values: {initial_f}"
            )

        ax.scatter(
            initial_f[0],
            initial_f[1],
            marker="D",
            c="black",
            s=110,
            edgecolors="white",
            linewidths=1.0,
            label="Initial solution",
            zorder=5,
        )

        ax.annotate(
            "Initial",
            xy=(initial_f[0], initial_f[1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            zorder=6,
        )

    ax.set_xlabel("f_move: mean free-walk time [s]")
    ax.set_ylabel("f_congestion")
    ax.set_title("Pareto front and initial solution")

    # どちらも最小化目的であることを明示
    ax.text(
        0.02,
        0.98,
        "Lower-left is better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.8,
        },
    )

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    
def plot_hypervolume_history(
    hv_history: pd.DataFrame,
    path: str | Path,
) -> None:
    """世代ごとのHypervolume推移を画像として保存する。"""

    required = {
        "generation",
        "hypervolume",
    }

    missing = required.difference(
        hv_history.columns
    )

    if missing:
        raise ValueError(
            "hv_history is missing columns: "
            f"{sorted(missing)}"
        )

    if hv_history.empty:
        raise ValueError(
            "hv_history must not be empty"
        )

    generations = hv_history[
        "generation"
    ].to_numpy(
        dtype=int,
    )

    hypervolume = hv_history[
        "hypervolume"
    ].to_numpy(
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        generations,
        hypervolume,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
    )

    ax.set_xlabel(
        "Generation"
    )
    ax.set_ylabel(
        "Hypervolume (HV)"
    )
    ax.set_title(
        "Hypervolume over generations"
    )
    ax.grid(
        alpha=0.3
    )

    # 値が非常に小さい、または大きい場合は科学表記を使用
    ax.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(-3, 4),
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150,
    )

    plt.close(fig)
    
def plot_time_series_comparison(
    initial_result,
    representative_results: dict,
    congestion_path: str | Path,
    movement_path: str | Path,
) -> None:
    """
    初期解と代表3解について、1パネルのグラフを2ファイル保存する。

    congestion_path:
        タイムステップ－混雑度

    movement_path:
        タイムステップ－移動負担
    """

    series = [
        ("初期解", initial_result),
        ("移動重視解", representative_results["best_movement"]),
        ("バランス解", representative_results["knee"]),
        ("混雑重視解", representative_results["best_congestion"]),
    ]

    # =========================================================
    # 混雑度
    # =========================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, result in series:
        ax.plot(
            result.time_steps,
            result.congestion_history,
            linewidth=1.8,
            label=label,
        )

    ax.set_xlabel("タイムステップ")
    ax.set_ylabel("混雑度")
    ax.set_title("タイムステップごとの混雑度")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(congestion_path, dpi=150)
    plt.close(fig)

    # =========================================================
    # 移動負担
    # =========================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, result in series:
        ax.plot(
            result.time_steps,
            result.movement_history,
            linewidth=1.8,
            label=label,
        )

    ax.set_xlabel("タイムステップ")
    ax.set_ylabel("移動負担 [s/人]")
    ax.set_title("タイムステップごとの移動負担")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(movement_path, dpi=150)
    plt.close(fig)


def plot_heatmaps(cfg: Config, layout: Layout, heatmap, over_map, path: str | Path, title="") -> None:
    """§20.2: 累積滞在人数と混雑閾値超過時間のヒートマップ。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, name in (
        (axes[0], heatmap, "cumulative presence [person-steps]"),
        (axes[1], over_map, "time over safe occupancy [s]"),
    ):
        masked = np.where(layout.walkable, data, np.nan)
        im = ax.imshow(masked, cmap="hot_r", origin="upper")
        draw_venue_outline(ax, cfg, layout)
        ax.set_title(f"{title} {name}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_venue_outline(ax, cfg: Config, layout: Layout) -> None:
    for b, (bx, by) in enumerate(layout.positions):
        spec = cfg.booths[b]
        ax.add_patch(Rectangle((bx - 0.5, by - 0.5), spec.width, spec.height,
                               fill=True, facecolor="#cccccc", edgecolor="black", linewidth=1.0))
        ax.text(bx + spec.width / 2 - 0.5, by + spec.height / 2 - 0.5, f"B{b + 1}",
                ha="center", va="center", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def comparison_table(results: dict[str, dict], path: str | Path) -> pd.DataFrame:
    """§20.3 の比較表を CSV へ保存し DataFrame を返す。

    results: {"移動重視": kpis+目的関数 dict, ...}
    """
    rows = {
        "平均自由歩行換算時間 [s]": "mean_free_walk_seconds",
        "平均待ち時間 [s]": "mean_queue_seconds",
        "平均ブロック待機時間 [s]": "mean_blocked_seconds",
        "混雑度 f_congestion": "f_congestion",
        "最大占有率": "max_cell_occupancy_ratio",
        "体験成功率": "experience_success_rate",
        "諦め回数": "give_up_total",
        "平均幸福度": "mean_happiness",
    }
    table = pd.DataFrame(
        {name: {label: vals.get(key, np.nan) for label, key in rows.items()}
         for name, vals in results.items()}
    )
    table.to_csv(path, encoding="utf-8-sig")
    return table


def render_animation(cfg: Config, layout: Layout, frames, path: str | Path, fps: int = 8) -> None:
    """§20.1: エージェントアニメーション(GIF)。frames は Simulation.record_frames の出力。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(right=0.72)
    draw_venue(ax, cfg, layout)
    scat = ax.scatter([], [], s=18)
    label = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", fontsize=9)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=STATE_COLORS[WALKING],
               label="移動中・目的地選択", markersize=6),
        Line2D([0], [0], marker="o", linestyle="none", color=STATE_COLORS[QUEUING],
               label="待機列", markersize=6),
        Line2D([0], [0], marker="o", linestyle="none", color=STATE_COLORS[EXPERIENCING],
               label="体験中（人数欄）", markersize=6),
        Line2D([0], [0], marker="o", linestyle="none", color=STATE_COLORS[EXITING],
               label="退出中", markersize=6),
        Line2D([0], [0], marker="o", linestyle="none", color=BLOCKED_COLOR,
               label="移動ブロック中", markersize=6),
    ]
    ax.legend(
        handles=legend_handles,
        title="エージェントの色",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=8,
        title_fontsize=9,
    )
    booth_count_text = ax.text(
        1.02,
        0.60,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        linespacing=1.5,
    )

    def update(i):
        t, snap, booth_stats = frames[i]
        xs = [s[1] for s in snap]
        ys = [s[2] for s in snap]
        colors = [
            BLOCKED_COLOR if s[5] > 0 else STATE_COLORS.get(s[3], "#333333")
            for s in snap
        ]
        scat.set_offsets(np.c_[xs, ys] if xs else np.empty((0, 2)))
        scat.set_color(colors)
        label.set_text(f"t = {t * cfg.dt_seconds:.0f} s  agents = {len(snap)}")
        count_lines = ["ブース別人数", "（体験中 / 待機中）"]
        count_lines.extend(
            f"B{booth_id + 1}: {active:>2} / {queued:>2}"
            for booth_id, active, queued in booth_stats
        )
        booth_count_text.set_text("\n".join(count_lines))
        return scat, label, booth_count_text

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
    ani.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
