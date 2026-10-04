"""visualize_solution.py — 配置 JSON の可視化(§20)。

使用例:
    python scripts/visualize_solution.py results/knee_layout.json --animate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import load_config  # noqa: E402
from src.parallel import build_context, evaluate_layout  # noqa: E402
from src.venue import apply_layout  # noqa: E402
from src.visualization import plot_heatmaps, render_animation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("layout_json", help="best_*_layout.json / knee_layout.json のパス")
    ap.add_argument("--config", default=str(ROOT / "config"))
    ap.add_argument("--out", default=None, help="出力先(既定: JSON と同じディレクトリ)")
    ap.add_argument("--animate", action="store_true", help="エージェント GIF を生成")
    ap.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="GIF の記録間隔(ステップ、既定: 1 = 現設定では1秒)",
    )
    ap.add_argument("--fps", type=int, default=8, help="GIF の再生フレーム数/秒")
    args = ap.parse_args()

    payload = json.loads(Path(args.layout_json).read_text(encoding="utf-8"))
    positions = tuple(tuple(p) for p in payload["positions"])
    stem = Path(args.layout_json).stem
    out = Path(args.out) if args.out else Path(args.layout_json).parent

    cfg = load_config(args.config)
    ctx = build_context(cfg)
    res = evaluate_layout(
        ctx,
        positions,
        collect_heatmap=True,
        record_frames=args.animate,
        frame_interval=args.frame_interval,
    )
    layout = apply_layout(cfg, ctx.venue, positions)
    print(f"f_move={res.f_move:.1f}s f_congestion={res.f_congestion:.3e}")

    plot_heatmaps(cfg, layout, res.heatmap, res.over_threshold_map,
                  out / f"{stem}_heatmap.png", title=stem)
    print(f"保存: {out / (stem + '_heatmap.png')}")
    if args.animate:
        gif = out / f"{stem}_animation.gif"
        render_animation(cfg, layout, res.frames, gif, fps=args.fps)
        print(f"保存: {gif}")


if __name__ == "__main__":
    main()
