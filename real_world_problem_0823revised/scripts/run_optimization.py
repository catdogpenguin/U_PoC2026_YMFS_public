"""通常ブース3変数・テーブル2変数の直接表現NSGA-IIを実行する。

例:
    python scripts/run_optimization.py --pop 100 --gen 100 --workers 8

初期集団は、座標・向きを完全ランダムに決めた物理制約充足個体で生成する。
列・外周モチーフや ``booths.yaml`` の固定配置は初期化に使用しない。
``booths.yaml`` の固定配置は、最適化後の比較にだけ使用する。
"""
from __future__ import annotations

import argparse
from functools import partial
import json
from multiprocessing import Pool
import os
from pathlib import Path
import sys
import time

# プロセス並列とBLAS/OpenMP内部並列の二重並列化を避ける。
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from pymoo.parallelization import StarmapParallelization
except ImportError:
    from pymoo.core.problem import StarmapParallelization

from src.constraints import (
    compute_violation,
    design_variable_count,
    positions_to_design,
    to_integer_layout,
)
from src.models import load_config
from src.optimization import calculate_hv_history, save_results, select_representatives
from src.parallel import build_context, default_workers, evaluate_layout, init_worker
from src.routing import warmup_routing_kernels
from src.direct_optimization import (
    ORDERLINESS_COMPONENT_KEYS,
    OrderlinessLimits,
    compute_direct_orderliness,
    prepare_random_feasible_direct_population,
    run_direct_nsga2,
    save_direct_metadata,
)
from src.venue import apply_layout
from src.visualization import (
    comparison_table,
    plot_heatmaps,
    plot_hypervolume_history,
    plot_pareto_front,
    plot_time_series_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config"))
    parser.add_argument("--pop", type=int, default=None)
    parser.add_argument("--gen", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max-orderliness",
        type=float,
        default=None,
        help="旧版互換: 0.25を基準に整然さ成分上限を一様拡縮する",
    )
    parser.add_argument(
        "--max-organization-nonmembership",
        "--max-lane-nonmembership",
        dest="max_organization_nonmembership",
        type=float,
        default=0.30,
        help="監査値のみ: 内部列または外周追従モチーフに属さない割合",
    )
    parser.add_argument(
        "--max-motif-complexity",
        type=float,
        default=1.20,
        help="監査値のみ: 必要標準モチーフ数に対する実モチーフ数比",
    )
    parser.add_argument(
        "--max-strong-structure-nonmembership",
        type=float,
        default=0.55,
        help="3個以上の列/外周チェーンに属さない割合の上限(既定: 0.55)",
    )
    parser.add_argument(
        "--max-alignment-error",
        "--max-lane-error",
        dest="max_alignment_error",
        type=float,
        default=0.36,
        help="内部列基準線または外周オフセットからの誤差上限(既定: 0.36)",
    )
    parser.add_argument(
        "--max-gap-variation",
        type=float,
        default=0.30,
        help="選択モチーフ内の外周間隔変動上限(既定: 0.30)",
    )
    parser.add_argument(
        "--max-orientation-error",
        type=float,
        default=0.25,
        help="各群の向き不一致率の上限(既定: 0.25)",
    )
    parser.add_argument(
        "--max-isolation",
        type=float,
        default=0.20,
        help="内部列・外周追従の近傍を持たない配置物割合の上限(既定: 0.20)",
    )
    # 旧版CLIを壊さないため受理するが、新方式では用途限定制約を廃止した。
    parser.add_argument(
        "--max-table-lane-nonmembership",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-table-lane-count-excess",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-table-gap-variation",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-front-clearance-violation",
        type=float,
        default=0.0,
        help="ブース正面2セル通路の閉塞セル割合上限(既定: 0)",
    )
    parser.add_argument(
        "--max-wide-aisle-deficit",
        type=float,
        default=0.35,
        help="3セル幅で往復到達できない主要ブース割合の上限(既定: 0.35)",
    )
    parser.add_argument(
        "--max-path-turns",
        type=float,
        default=4.5,
        help="主要ブース間の平均方向転換回数上限(既定: 4.5、違反量は3倍)",
    )
    parser.add_argument(
        "--initial-max-trials",
        type=int,
        default=5000,
        help="ランダム制約充足初期個体1件あたりの最大試行数",
    )
    parser.add_argument(
        "--mutation-individual-prob",
        type=float,
        default=0.20,
        help="子個体に三段階突然変異を適用する確率(既定: 0.20)",
    )
    parser.add_argument(
        "--mutation-local-weight",
        type=float,
        default=0.55,
        help="突然変異対象個体で局所変異を選ぶ確率(既定: 0.55)",
    )
    parser.add_argument(
        "--mutation-meso-weight",
        type=float,
        default=0.30,
        help="突然変異対象個体で中規模変異を選ぶ確率(既定: 0.30)",
    )
    parser.add_argument(
        "--mutation-macro-weight",
        type=float,
        default=0.15,
        help="突然変異対象個体で大域変異を選ぶ確率(既定: 0.15)",
    )
    parser.add_argument(
        "--exploration-generations",
        type=int,
        default=30,
        help="大域突然変異を50%%へ強化する初期世代数(既定: 30)",
    )
    parser.add_argument(
        "--donor-pool-size",
        type=int,
        default=32,
        help="大域突然変異が保持する多様な整然配置ドナー数(既定: 32)",
    )
    parser.add_argument(
        "--minimum-layout-distance",
        type=float,
        default=0.08,
        help="環境選択で見た目の近似重複を後回しにする配置距離(既定: 0.08)",
    )
    parser.add_argument(
        "--memetic-interval",
        type=int,
        default=5,
        help="非劣解近傍探索の実行間隔(既定: 5世代)",
    )
    parser.add_argument(
        "--memetic-start-generation",
        type=int,
        default=30,
        help="Memetic近傍探索を開始する世代(既定: 30)",
    )
    parser.add_argument(
        "--memetic-parents",
        type=int,
        default=10,
        help="近傍探索する非劣解数(既定: 10)",
    )
    parser.add_argument(
        "--memetic-neighbors",
        type=int,
        default=10,
        help="非劣解1個体当たりの近傍評価数(既定: 10)",
    )
    parser.add_argument(
        "--memetic-max-neighbor-trials",
        type=int,
        default=1000,
        help="1親から実行可能近傍を生成する最大試行数",
    )
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()
    mutation_weights = (
        args.mutation_local_weight
        + args.mutation_meso_weight
        + args.mutation_macro_weight
    )
    if not np.isclose(mutation_weights, 1.0):
        parser.error("三段階突然変異の3つの重みの合計は1にしてください")
    if not 0.0 <= args.mutation_individual_prob <= 1.0:
        parser.error("--mutation-individual-probは0以上1以下にしてください")
    if args.exploration_generations < 0:
        parser.error("--exploration-generationsは0以上にしてください")
    if args.donor_pool_size < 1:
        parser.error("--donor-pool-sizeは1以上にしてください")
    if not 0.0 <= args.minimum_layout_distance <= 1.0:
        parser.error("--minimum-layout-distanceは0以上1以下にしてください")
    if args.memetic_start_generation < 1:
        parser.error("--memetic-start-generationは1以上にしてください")
    if min(args.memetic_interval, args.memetic_parents, args.memetic_neighbors) < 1:
        parser.error("Memeticの間隔・親数・近傍数は1以上にしてください")
    if args.memetic_max_neighbor_trials < args.memetic_neighbors:
        parser.error("近傍生成最大試行数は近傍数以上にしてください")

    orderliness_limits = OrderlinessLimits(
        organization_nonmembership=args.max_organization_nonmembership,
        motif_complexity=args.max_motif_complexity,
        strong_structure_nonmembership=(
            args.max_strong_structure_nonmembership
        ),
        alignment_error=args.max_alignment_error,
        gap_variation=args.max_gap_variation,
        orientation=args.max_orientation_error,
        isolation=args.max_isolation,
        front_clearance=args.max_front_clearance_violation,
        wide_aisle_deficit=args.max_wide_aisle_deficit,
        path_turns=args.max_path_turns,
    )

    cfg = load_config(args.config)
    n_variables = design_variable_count(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    workers = args.workers if args.workers is not None else default_workers()
    print(
        f"{n_variables}次元・1層型GA設定: pop={args.pop or cfg.population_size}, "
        f"gen={args.gen or cfg.n_generations}, workers={workers}, "
        f"orderliness_limits={orderliness_limits.as_dict()}, "
        "mutation="
        f"{args.mutation_individual_prob:.2f}×"
        f"({args.mutation_local_weight:.2f}/"
        f"{args.mutation_meso_weight:.2f}/"
        f"{args.mutation_macro_weight:.2f}), "
        f"exploration=1-{args.exploration_generations}世代は0.40×(0.20/0.30/0.50), "
        f"donors={args.donor_pool_size}, diversity={args.minimum_layout_distance:.3f}, "
        "orderliness=fixed_from_generation_1, crossover=two_point_booth_block, "
        f"memetic={args.memetic_start_generation}世代以降"
        f"{args.memetic_interval}世代ごと×"
        f"{args.memetic_parents}非劣解×{args.memetic_neighbors}近傍"
    )

    initial_designs = None

    jit_started = time.perf_counter()
    warmup_routing_kernels()
    print(f"経路探索JIT準備: {time.perf_counter() - jit_started:.1f}s")

    def execute(runner=None):
        return run_direct_nsga2(
            cfg=cfg,
            pop_size=args.pop,
            n_gen=args.gen,
            elementwise_runner=runner,
            seed=args.seed,
            orderliness_limits=orderliness_limits,
            max_orderliness=args.max_orderliness,
            initial_max_trials=args.initial_max_trials,
            initial_genotypes=initial_designs,
            mutation_individual_prob=args.mutation_individual_prob,
            mutation_local_weight=args.mutation_local_weight,
            mutation_meso_weight=args.mutation_meso_weight,
            mutation_macro_weight=args.mutation_macro_weight,
            exploration_generations=args.exploration_generations,
            donor_pool_size=args.donor_pool_size,
            minimum_layout_distance=args.minimum_layout_distance,
            memetic_interval=args.memetic_interval,
            memetic_start_generation=args.memetic_start_generation,
            memetic_parents=args.memetic_parents,
            memetic_neighbors=args.memetic_neighbors,
            memetic_max_neighbor_trials=args.memetic_max_neighbor_trials,
        )

    started = time.perf_counter()
    if workers > 1:
        # 評価用Poolを初期集団とMemetic近傍の構成にも再利用する。
        preparation_started = time.perf_counter()
        with Pool(workers, initializer=init_worker, initargs=(cfg,)) as pool:
            pool_starmap = partial(pool.starmap, chunksize=1)
            initial_designs = prepare_random_feasible_direct_population(
                cfg,
                pop_size=int(args.pop or cfg.population_size),
                seed=args.seed,
                orderliness_limits=orderliness_limits,
                max_orderliness=args.max_orderliness,
                initial_max_trials=args.initial_max_trials,
                starmap=pool_starmap,
            )
            print(
                "完全ランダム物理制約充足初期集団の準備: "
                f"{time.perf_counter() - preparation_started:.1f}s"
            )
            result = execute(
                StarmapParallelization(pool_starmap)
            )
    else:
        preparation_started = time.perf_counter()
        initial_designs = prepare_random_feasible_direct_population(
            cfg,
            pop_size=int(args.pop or cfg.population_size),
            seed=args.seed,
            orderliness_limits=orderliness_limits,
            max_orderliness=args.max_orderliness,
            initial_max_trials=args.initial_max_trials,
        )
        print(
            "完全ランダム物理制約充足初期集団の準備: "
            f"{time.perf_counter() - preparation_started:.1f}s"
        )
        result = execute()
    elapsed = time.perf_counter() - started

    F = np.atleast_2d(np.asarray(result.F, dtype=float))
    print(
        f"\n{n_variables}次元・直接表現NSGA-II完了: {elapsed:.1f}s / "
        f"パレート解 {len(F)} 件"
    )

    representative_layouts = save_results(cfg, result, out)
    save_direct_metadata(result, out)
    representatives = select_representatives(F)

    pd.DataFrame(
        {
            "f_move": F[:, 0],
            "f_congestion": F[:, 1],
            "orderliness": np.asarray(result.orderliness_scores, dtype=float),
            **{
                key: [item[key] for item in result.orderliness_components]
                for key in ORDERLINESS_COMPONENT_KEYS
            },
        }
    ).to_csv(out / "optimization_log.csv", index_label="pareto_index")

    hv_history, hv_reference_point = calculate_hv_history(result, reference_margin=0.1)
    hv_history.to_csv(out / "hv_history.csv", index=False, encoding="utf-8-sig")
    plot_hypervolume_history(hv_history, out / "hypervolume_over_generations.png")

    # 実際に最適化へ投入した完全ランダム初期集団の第1個体。
    initial_design = np.asarray(initial_designs[0], dtype=np.int64).copy()
    initial_orderliness = compute_direct_orderliness(
        cfg, initial_design, limits=result.orderliness_limits
    )
    ctx = build_context(cfg)
    initial_positions = to_integer_layout(initial_design, cfg)
    initial_result = evaluate_layout(ctx, initial_positions)
    initial_f = np.asarray(
        [initial_result.f_move, initial_result.f_congestion], dtype=float
    )
    (out / "random_initial_layout.json").write_text(
        json.dumps(
            {
                f"design_vector_{n_variables}": [int(v) for v in initial_design],
                "positions": [list(map(int, p)) for p in initial_positions],
                "orderliness": initial_orderliness,
                "f_move": float(initial_f[0]),
                "f_congestion": float(initial_f[1]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # booths.yaml は初期化には使わず、最適化後の比較対象としてのみ評価する。
    fixed_positions = tuple(
        tuple(int(v) for v in booth.fixed_position)
        for booth in cfg.booths
        if booth.fixed_position is not None
    )
    if len(fixed_positions) != len(cfg.booths):
        raise ValueError("比較用fixed_positionが未設定の配置物があります")
    fixed_violation = compute_violation(cfg, ctx.venue, fixed_positions)
    if fixed_violation > 0:
        raise ValueError(f"booths.yamlの比較用固定配置が制約違反です: g={fixed_violation}")
    fixed_orderliness = compute_direct_orderliness(
        cfg,
        positions_to_design(cfg, fixed_positions),
        limits=result.orderliness_limits,
    )
    if float(fixed_orderliness["total"]) > 1.0:
        print(
            "警告: 指定された受容可能性上限ではbooths.yamlが探索可能領域外です: "
            f"orderliness={float(fixed_orderliness['total']):.6g}"
        )
    else:
        print(
            "booths.yaml探索空間確認: "
            f"orderliness={float(fixed_orderliness['total']):.6g} <= 1"
        )
    fixed_result = evaluate_layout(ctx, fixed_positions)
    fixed_f = np.asarray([fixed_result.f_move, fixed_result.f_congestion], dtype=float)

    plot_pareto_front(
        F=F,
        reps=representatives,
        path=out / "pareto_front.png",
        initial_f=initial_f,
        fixed_f=fixed_f,
    )

    japanese_names = {
        "best_movement": "移動重視",
        "knee": "バランス",
        "best_congestion": "混雑重視",
    }
    output_stems = {
        "best_movement": "best_movement",
        "knee": "knee",
        "best_congestion": "best_congestion",
    }
    table_source = {}
    orderliness_diagnostics = {}
    representative_results = {}
    for name in ("best_movement", "knee", "best_congestion"):
        positions = tuple(
            tuple(int(v) for v in p)
            for p in representative_layouts[name]["positions"]
        )
        evaluated = evaluate_layout(ctx, positions, collect_heatmap=True)
        representative_results[name] = evaluated
        layout = apply_layout(cfg, ctx.venue, positions)
        plot_heatmaps(
            cfg,
            layout,
            evaluated.heatmap,
            evaluated.over_threshold_map,
            out / f"heatmap_{output_stems[name]}.png",
            title=japanese_names[name],
        )
        table_source[japanese_names[name]] = {
            **evaluated.kpis,
            "f_congestion": evaluated.f_congestion,
            "orderliness": float(result.orderliness_scores[representatives[name]]),
            **{
                key: float(result.orderliness_components[representatives[name]][key])
                for key in ORDERLINESS_COMPONENT_KEYS
            },
        }
        orderliness_diagnostics[name] = result.orderliness_components[
            representatives[name]
        ]

    table_source["ランダム初期解"] = {
        **initial_result.kpis,
        "f_congestion": initial_result.f_congestion,
        "orderliness": float(initial_orderliness["total"]),
        **{
            key: float(initial_orderliness[key])
            for key in ORDERLINESS_COMPONENT_KEYS
        },
    }
    orderliness_diagnostics["random_initial"] = initial_orderliness
    table_source["booths.yaml比較配置"] = {
        **fixed_result.kpis,
        "f_congestion": fixed_result.f_congestion,
        "orderliness": float(fixed_orderliness["total"]),
        **{
            key: float(fixed_orderliness[key])
            for key in ORDERLINESS_COMPONENT_KEYS
        },
    }
    orderliness_diagnostics["booths_yaml"] = fixed_orderliness
    (out / "orderliness_diagnostics.json").write_text(
        json.dumps(orderliness_diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    table = comparison_table(table_source, out / "comparison_table.csv")
    plot_time_series_comparison(
        initial_result=initial_result,
        representative_results=representative_results,
        congestion_path=out / "congestion_over_time.png",
        movement_path=out / "movement_burden_over_time.png",
    )

    print("\n=== Hypervolume ===")
    print(f"reference point = ({hv_reference_point[0]:.6g}, {hv_reference_point[1]:.6g})")
    print(f"final HV       = {hv_history['hypervolume'].iloc[-1]:.6g}")
    print("\n=== 代表解・初期解・比較配置 ===")
    print(table.to_string())
    print(f"\n成果物: {out}/")


if __name__ == "__main__":
    main()
