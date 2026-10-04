"""preview_venue_shape.py — config の venue_shape から会場外形とブース固定配置を図で出力する。

会場形状・固定配置(booths.yaml の fixed_position)の微調整用。
default_config.yaml / booths.yaml を編集するたびに実行して確認する。

使用例(リポジトリルートから):
    python real_world_problem/tests/preview_venue_shape.py
    python real_world_problem/tests/preview_venue_shape.py --out real_world_problem/results/venue_shape.png
    python real_world_problem/tests/preview_venue_shape.py --no-booths
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from src.models import EMPTY, FORBIDDEN, WALL, load_config  # noqa: E402
from src.venue import apply_layout, build_base_venue, oriented_booth_size  # noqa: E402
from src.visualization import (  # noqa: E402
    _FRONT_MARKERS,
    _OPPOSITE_THETA,
    _draw_table_icon,
    booth_style_map,
)

QUEUE_FACE = "#f7e8a0"  # src/visualization.py の _CELL_FACE[QUEUE] と統一

# 日本語ラベル用フォント(存在するもののみ適用)
import matplotlib.font_manager as _fm  # noqa: E402

for _f in ("Noto Sans CJK JP", "IPAexGothic", "Hiragino Sans", "Yu Gothic", "Meiryo"):
    if any(f.name == _f for f in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break


def _draw_fixed_booths(ax, cfg) -> list[tuple[str, str]]:
    """booths.yaml の fixed_position ((x, y, theta)) をブース属性色で重ね描きする。

    戻り値は凡例用に描画した (kind, theme) キーの一覧。
    """
    missing = [s.name or s.booth_id for s in cfg.booths if s.fixed_position is None]
    if missing:
        print(f"警告: fixed_position 未指定のブースは描画をスキップします: {missing}")

    positions = tuple(
        s.fixed_position if s.fixed_position is not None else (0, 0, 0)
        for s in cfg.booths
    )
    layout = apply_layout(cfg, build_base_venue(cfg), positions)
    style_map = booth_style_map(cfg)

    used_attrs: list[tuple[str, str]] = []
    for b, (bx, by, theta) in enumerate(layout.positions):
        if cfg.booths[b].fixed_position is None:
            continue
        spec = cfg.booths[b]
        width, height = oriented_booth_size(cfg, b, theta)

        for qx, qy in layout.queue_cells[b]:
            ax.add_patch(
                Rectangle(
                    (qx - 0.5, qy - 0.5),
                    1,
                    1,
                    facecolor=QUEUE_FACE,
                    edgecolor="none",
                )
            )

        key = (spec.kind, "normal" if spec.is_table else spec.theme)
        face = style_map.get(key, style_map[("booth", "normal")])[0]
        # テーブル・椅子は fig/table.png のアイコンだけで示す(src/visualization.py と統一)。
        drawn_as_icon = spec.is_table and _draw_table_icon(
            ax, bx - 0.5, by - 0.5, width, height, zorder=2
        )
        if not drawn_as_icon:
            if key not in used_attrs:
                used_attrs.append(key)
            ax.add_patch(
                Rectangle(
                    (bx - 0.5, by - 0.5),
                    width,
                    height,
                    facecolor=face,
                    edgecolor="black",
                    linewidth=0.8,
                )
            )
        if not spec.is_table:
            ax.text(
                bx + width / 2 - 0.5,
                by + height / 2 - 0.5,
                f"B{b + 1}",
                ha="center",
                va="center",
                fontsize=5,
                color="white",
            )

        if layout.receptions[b] is not None:
            rx, ry = layout.receptions[b]
            ax.plot(
                rx,
                ry,
                marker=_FRONT_MARKERS[_OPPOSITE_THETA[int(theta)]],
                color="black",
                markersize=4,
            )

        if layout.booth_exits[b] is not None:
            ex, ey = layout.booth_exits[b]
            ax.plot(ex, ey, marker=_FRONT_MARKERS[int(theta)], color="black", markersize=4)
    return used_attrs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config"), help="config ディレクトリ")
    ap.add_argument(
        "--out",
        default=str(ROOT / "results" / "venue_shape.png"),
        help="出力 PNG パス",
    )
    ap.add_argument(
        "--no-booths",
        action="store_true",
        help="ブース配置を描画しない(会場外形のみ)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    venue = build_base_venue(cfg)
    g = venue.base_grid
    H, W = cfg.grid_height, cfg.grid_width

    # 表示区分: 0=広場(配置可) / 1=場外・障害物 / 2=道路(歩行のみ) / 3=開口部(入退場)
    disp = np.zeros((H, W), dtype=int)
    disp[g == WALL] = 1
    disp[g == FORBIDDEN] = 2
    for (x, y) in {*venue.entrance_cells, *venue.exit_cells}:
        disp[y, x] = 3
    cmap = ListedColormap(["#ffffff", "#4d4d4d", "#cfe8cf", "#f4a9a8"])

    fig, ax = plt.subplots(figsize=(max(4.0, W / 6), max(6.0, H / 6)))
    ax.imshow(disp, cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(range(0, W, 5))
    ax.set_yticks(range(0, H, 5))
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="#eeeeee", linewidth=0.2)
    ax.grid(which="major", color="#bbbbbb", linewidth=0.4)

    handles = [
        Patch(facecolor="#ffffff", edgecolor="#999999", label="広場(ブース配置可)"),
        Patch(facecolor="#cfe8cf", label="道路(歩行のみ)"),
        Patch(facecolor="#f4a9a8", label="開口部(入退場)"),
        Patch(facecolor="#4d4d4d", label="場外・障害物"),
    ]

    title = (
        f"会場外形 {W}×{H} セル(1 セル = {cfg.cell_size_meters:g} m)\n"
        f"広場 {int(np.sum(g == EMPTY))} / 道路 {int(np.sum(g == FORBIDDEN))} / "
        f"開口部 {len(venue.entrance_cells)} セル"
    )

    if not args.no_booths:
        used_attrs = _draw_fixed_booths(ax, cfg)
        style_map = booth_style_map(cfg)
        handles.extend(
            Patch(facecolor=style_map[k][0], edgecolor="black", label=style_map[k][1])
            for k in used_attrs
        )
        handles.append(Patch(facecolor=QUEUE_FACE, label="待機列"))
        handles.append(
            Line2D([0], [0], marker="^", linestyle="none", color="black",
                   label="受付", markersize=5)
        )
        handles.append(
            Line2D([0], [0], marker="v", linestyle="none", color="black",
                   label="体験後退出", markersize=5)
        )
        n_placed = sum(s.fixed_position is not None for s in cfg.booths)
        title += f" / 固定配置ブース {n_placed} 個"

    ax.set_title(title)
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        framealpha=0.9,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"会場外形図を出力しました: {out}")


if __name__ == "__main__":
    main()
