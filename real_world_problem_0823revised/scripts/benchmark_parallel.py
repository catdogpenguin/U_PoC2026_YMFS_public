"""138次元・1層型GAの並列化が実際に短縮へ効いているか測定する。

同じseed・同じ初期集団をworkers数だけ変えて1世代評価し、経過時間、
speedup、目的値の完全一致を確認する。
"""
from __future__ import annotations

import argparse
from functools import partial
from multiprocessing import Pool
import os
from pathlib import Path
import sys
import time

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from pymoo.parallelization import StarmapParallelization
except ImportError:
    from pymoo.core.problem import StarmapParallelization

from src.models import load_config
from src.parallel import init_worker
from src.routing import warmup_routing_kernels
from src.direct_optimization import (
    prepare_random_feasible_direct_population,
    run_direct_nsga2,
)


def _sorted_objectives(result) -> np.ndarray:
    F = np.atleast_2d(np.asarray(result.F, dtype=float))
    return F[np.lexsort((F[:, 1], F[:, 0]))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config"))
    parser.add_argument("--pop", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-orderliness", type=float, default=0.25)
    args = parser.parse_args()

    cfg = load_config(args.config)
    # 全workers条件で完全に同じ初期個体を使い、キャッシュ構築時間と
    # シミュレーション並列時間を混同しない。
    initial_designs = prepare_random_feasible_direct_population(
        cfg,
        pop_size=args.pop,
        seed=args.seed,
        max_orderliness=args.max_orderliness,
    )
    warmup_routing_kernels()
    baseline_F = None
    baseline_seconds = None
    all_equal = True

    print("workers  seconds  speedup  same_objectives")
    for workers in args.workers:
        started = time.perf_counter()
        if workers <= 1:
            result = run_direct_nsga2(
                cfg,
                pop_size=args.pop,
                n_gen=1,
                seed=args.seed,
                verbose=False,
                max_orderliness=args.max_orderliness,
                initial_genotypes=initial_designs,
            )
        else:
            with Pool(workers, initializer=init_worker, initargs=(cfg,)) as pool:
                result = run_direct_nsga2(
                    cfg,
                    pop_size=args.pop,
                    n_gen=1,
                    elementwise_runner=StarmapParallelization(
                        partial(pool.starmap, chunksize=1)
                    ),
                    seed=args.seed,
                    verbose=False,
                    max_orderliness=args.max_orderliness,
                    initial_genotypes=initial_designs,
                )
        elapsed = time.perf_counter() - started
        F = _sorted_objectives(result)
        if baseline_F is None:
            baseline_F = F
            baseline_seconds = elapsed
            same = True
        else:
            same = np.array_equal(F, baseline_F)
            all_equal = all_equal and same
        speedup = baseline_seconds / elapsed
        print(f"{workers:7d}  {elapsed:7.2f}  {speedup:7.2f}  {same}")

    if not all_equal:
        print("ERROR: workers数により目的値が変化しました")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
