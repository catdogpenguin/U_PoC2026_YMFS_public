"""run_optimization.py — NSGA-II 最適化の実行と成果物出力(§15, §16, §19, §20)。

使用例:
    python scripts/run_optimization.py                 # フル設定(pop=64, gen=50)
    python scripts/run_optimization.py --pop 16 --gen 8 --workers 4
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pymoo.core.problem import StarmapParallelization  # noqa: E402

from src.constraints import to_integer_layout  # noqa: E402
from src.models import load_config  # noqa: E402
from src.optimization import (  # noqa: E402
    calculate_hv_history,
    run_nsga2,
    sample_valid_layout,
    save_results,
    select_representatives,
    variable_bounds,
)
from src.rng import sampling_rng
from src.parallel import build_context, default_workers, evaluate_layout, init_worker  # noqa: E402
from src.venue import apply_layout  # noqa: E402
from src.visualization import (  # noqa: E402
    comparison_table,
    plot_heatmaps,
    plot_hypervolume_history,
    plot_pareto_front,
    plot_time_series_comparison,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config"))
    ap.add_argument("--pop", type=int, default=None, help="個体数(既定: config)")
    ap.add_argument("--gen", type=int, default=None, help="世代数(既定: config)")
    ap.add_argument("--workers", type=int, default=None, help="並列プロセス数")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    workers = args.workers if args.workers is not None else default_workers()

    t0 = time.perf_counter()
    if workers > 1:
        with Pool(workers, initializer=init_worker, initargs=(cfg,)) as pool:
            runner = StarmapParallelization(pool.starmap)
            res = run_nsga2(cfg, pop_size=args.pop, n_gen=args.gen, elementwise_runner=runner)
    else:
        res = run_nsga2(cfg, pop_size=args.pop, n_gen=args.gen)
    elapsed = time.perf_counter() - t0
    print(f"\n最適化完了: {elapsed:.1f}s / パレート解 {np.atleast_2d(res.F).shape[0]} 件")

    # --- §19 成果物
    rep_layouts = save_results(cfg, res, out)
    F = np.atleast_2d(res.F)
    reps = select_representatives(F)
    
    # 最適化ログ(世代別の目的値集計は verbose 出力に委ね、最終集団を保存)
    pd.DataFrame(F, columns=["f_move", "f_congestion"]).to_csv(
        out / "optimization_log.csv", index_label="pareto_index"
    )
    
    # -------------------------------------------------------------
    # 世代ごとのHypervolume（HV）
    # -------------------------------------------------------------
    # 全世代で同じ参照点を使用する。
    # 参照点は、全世代の実行可能解の最大目的値に、
    # 目的値レンジの10%分の余白を加えて自動設定する。
    hv_history, hv_reference_point = calculate_hv_history(
        res,
        reference_margin=0.1,
    )

    # 世代ごとのHVをCSVへ保存
    hv_history.to_csv(
        out / "hv_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # HVの世代推移をresultsフォルダへ保存
    plot_hypervolume_history(
        hv_history,
        out / "hypervolume_over_generations.png",
    )

    # コンソールにも参照点と最終世代HVを表示
    print("\n=== Hypervolume ===")

    print(
        "reference point = "
        f"({hv_reference_point[0]:.6g}, "
        f"{hv_reference_point[1]:.6g})"
    )

    print(
        "final HV       = "
        f"{hv_history['hypervolume'].iloc[-1]:.6g}"
    )

    # --- §20 代表 3 案の可視化・比較表(最適化終了後に再評価)
    ctx = build_context(cfg)
    jp = {"best_movement": "移動重視", "knee": "バランス", "best_congestion": "混雑重視"}
    table_src = {}
    representative_results = {}
    for name in ("best_movement", "knee", "best_congestion"):
        positions = tuple(tuple(p) for p in rep_layouts[name]["positions"])
        r = evaluate_layout(ctx, positions, collect_heatmap=True)
        representative_results[name] = r
        layout = apply_layout(cfg, ctx.venue, positions)
        plot_heatmaps(
            cfg, layout, r.heatmap, r.over_threshold_map,
            out / f"heatmap_{name}.png", title=jp[name],
        )
        table_src[jp[name]] = {**r.kpis, "f_congestion": r.f_congestion}
        
    # =============================================================
    # 以下は，初期解と選択済み最適解の時系列比較
    # =============================================================
    # NSGA-IIの初期集団生成と同じ変数範囲
    lower_bounds, upper_bounds = variable_bounds(cfg)

    # ValidLayoutSamplingと同じ乱数系列を使用する。
    # これにより初期集団の第1有効配置を再現する。
    initial_x = sample_valid_layout(
        sampling_rng(cfg.evaluation_seed),
        lower_bounds,
        upper_bounds,
        cfg,
        ctx.venue,
    )

    # sample_valid_layout が返す初期個体は既に整数ベクトル。
    # シミュレーション用の配置タプルへ変換する。
    initial_positions = to_integer_layout(initial_x)

    # 初期解をシミュレーション
    initial_result = evaluate_layout(
        ctx,
        initial_positions,
    )
    
    # -------------------------------------------------------------
    # 初期解の目的関数値
    # -------------------------------------------------------------
    initial_f = np.array(
        [
            initial_result.f_move,
            initial_result.f_congestion,
        ],
        dtype=float,
    )

    print("\n=== パレートフロント比較用の初期解 ===")
    print(f"f_move       = {initial_f[0]:.6f}")
    print(f"f_congestion = {initial_f[1]:.6e}")

    # -------------------------------------------------------------
    # パレート解・代表解・初期解を同じグラフに表示
    # -------------------------------------------------------------
    plot_pareto_front(
        F=F,
        reps=reps,
        path=out / "pareto_front.png",
        initial_f=initial_f,
    )

    # -------------------------------------------------------------
    # 比較対象とする選択済み最適解
    # -------------------------------------------------------------
    # 選択肢：
    #   "best_movement"   : 移動重視解
    #   "knee"            : バランス解
    #   "best_congestion" : 混雑重視解
    selected_name = "knee"

    selected_result = representative_results[selected_name]

    # 2種類の時系列グラフを保存
    plot_time_series_comparison(
        initial_result=initial_result,
        representative_results=representative_results,
        congestion_path=out / "congestion_over_time.png",
        movement_path=out / "movement_burden_over_time.png",
    )
    table = comparison_table(table_src, out / "comparison_table.csv")
    print("\n=== 代表 3 案の比較(§20.3) ===")
    print(table.to_string())
    print(f"\n成果物: {out}/")


if __name__ == "__main__":  # §16.3 Windows 対応ガード
    main()
