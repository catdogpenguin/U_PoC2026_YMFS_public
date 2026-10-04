"""benchmark.py — 1 秒評価目標のベンチマーク(§18)。

ランダムな有効配置を 10 件生成し、直列評価時間を測定する。
合格基準: 平均 1.0 秒以下、最大 2.0 秒以下。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constraints import to_integer_layout  # noqa: E402
from src.models import load_config  # noqa: E402
from src.optimization import sample_valid_layout, variable_bounds  # noqa: E402
from src.parallel import build_context, evaluate_layout  # noqa: E402
from src.rng import sampling_rng  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config"))
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ctx = build_context(cfg)
    rng = sampling_rng(cfg.evaluation_seed)
    lower, upper = variable_bounds(cfg)

    benchmark_layouts = [
        to_integer_layout(sample_valid_layout(rng, lower, upper, cfg, ctx.venue))
        for _ in range(args.n)
    ]

    times = []
    for i, layout in enumerate(benchmark_layouts):
        start = time.perf_counter()
        res = evaluate_layout(ctx, layout)
        times.append(time.perf_counter() - start)
        print(
            f"[{i + 1}/{args.n}] {times[-1]:.3f}s  "
            f"f_move={res.f_move:.1f} f_cong={res.f_congestion:.3e}"
        )
    mean_t, max_t = float(np.mean(times)), float(np.max(times))
    print(f"\n平均 {mean_t:.3f}s / 最大 {max_t:.3f}s")
    ok = mean_t <= 1.0 and max_t <= 2.0
    print("合格基準(平均 1.0s 以下・最大 2.0s 以下):", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
