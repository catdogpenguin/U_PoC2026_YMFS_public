"""run_fixed_layout.py — 固定配置パターンでの人流シミュレーション(最適化なし)。

booths.yaml の各ブースに記した fixed_position を配置として採用し、
制約チェック → 1 回シミュレーション → KPI 表示と可視化を行う。
最適化(run_optimization.py)前のベースライン評価に使う。

使用例(リポジトリルートから):
    python real_world_problem/scripts/run_fixed_layout.py
    python real_world_problem/scripts/run_fixed_layout.py --animate --frame-interval 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constraints import compute_violation  # noqa: E402
from src.models import load_config  # noqa: E402
from src.parallel import build_context, evaluate_layout  # noqa: E402
from src.venue import apply_layout  # noqa: E402
from src.visualization import (  # noqa: E402
    draw_venue,
    plot_heatmaps,
    render_animation,
)

import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config"))
    ap.add_argument("--out", default=str(ROOT / "results" / "fixed"))
    ap.add_argument("--animate", action="store_true", help="エージェント GIF を生成")
    ap.add_argument("--frame-interval", type=int, default=1, help="GIF の記録間隔(ステップ)")
    ap.add_argument("--fps", type=int, default=8, help="GIF の再生フレーム数/秒")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- booths.yaml の fixed_position を配置として採用
    missing = [s.name or s.booth_id for s in cfg.booths if s.fixed_position is None]
    if missing:
        raise SystemExit(f"fixed_position 未指定のブースがあります: {missing}")
    positions = tuple(s.fixed_position for s in cfg.booths)

    ctx = build_context(cfg)
    g = compute_violation(cfg, ctx.venue, positions)
    if g > 0:
        raise SystemExit(
            f"固定配置が制約違反です(g = {g})。booths.yaml の fixed_position を見直してください"
        )
    print(f"制約チェック OK(g = 0、オブジェクト {len(positions)} 個)")

    # --- 配置図(属性の色分け付き)
    layout = apply_layout(cfg, ctx.venue, positions)
    fig, ax = plt.subplots(figsize=(9, 12))
    draw_venue(ax, cfg, layout, legend=True)
    ax.set_title("固定配置パターン")
    fig.savefig(out / "fixed_layout.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- シミュレーション
    res = evaluate_layout(
        ctx,
        positions,
        collect_heatmap=True,
        record_frames=args.animate,
        frame_interval=args.frame_interval,
    )
    if res.timed_out:
        print("警告: シミュレーションがタイムアウトしました(timeout_seconds を確認)")
    print(f"f_move       = {res.f_move:.4f}")
    print(f"f_congestion = {res.f_congestion:.6e}")
    print("--- KPI ---")
    for k, v in res.kpis.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- 成果物
    plot_heatmaps(
        cfg, layout, res.heatmap, res.over_threshold_map,
        out / "fixed_heatmap.png", title="固定配置",
    )
    payload = {
        "positions": [list(p) for p in positions],
        "f_move": res.f_move,
        "f_congestion": res.f_congestion,
        "kpis": res.kpis,
    }
    (out / "fixed_layout.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.animate:
        render_animation(cfg, layout, res.frames, out / "fixed_animation.gif", fps=args.fps)
        print(f"保存: {out / 'fixed_animation.gif'}")
    print(f"成果物: {out}/")


if __name__ == "__main__":
    main()
