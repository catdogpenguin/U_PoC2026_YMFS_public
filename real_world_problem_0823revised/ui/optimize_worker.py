"""optimize_worker.py — UI から起動される NSGA-II 実行プロセス。

    python -m ui.optimize_worker <run_dir>

Streamlit のプロセス内で最適化を回すと、
  - multiprocessing.Pool(Windows は spawn)がアプリスクリプトを再 import する
  - 長時間ブロックして UI が固まる
という 2 つの問題が出るため、独立プロセスとして分離する。
UI との通信は run_dir 配下のファイルのみ(run_store.py 参照)。

`scripts/run_optimization.py` と同じ直接配置方式
(`run_direct_nsga2` / `save_results` / `save_direct_metadata`)を呼ぶ。
世代ごとのスナップショット出力は `on_generation` フックとして差し込む。
"""
from __future__ import annotations

import sys
import time
import traceback
from dataclasses import replace
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pymoo.core.problem import StarmapParallelization  # noqa: E402

from src.constraints import (  # noqa: E402
    compute_violation,
    design_variable_count,
    positions_to_design,
)
from src.direct_optimization import (  # noqa: E402
    OrderlinessLimits,
    prepare_random_feasible_direct_population,
    run_direct_nsga2,
    save_direct_metadata,
)
from src.models import load_config  # noqa: E402
from src.optimization import save_results  # noqa: E402
from src.parallel import build_context, evaluate_layout, init_worker  # noqa: E402
from src.routing import warmup_routing_kernels  # noqa: E402
from ui import run_store  # noqa: E402

#: 実会場配置の評価に使うタイムアウト(秒)。
#: 最適化中の打ち切り(§17.6)とは別に、比較基準は必ず完走させたいので長くとる。
BASELINE_TIMEOUT_SECONDS = 3600.0


def _orderliness_limits(params: run_store.RunParams) -> OrderlinessLimits:
    """UI の上限値を直接配置問題の成分別制約へ変換する。"""
    return OrderlinessLimits(
        organization_nonmembership=params.max_organization_nonmembership,
        motif_complexity=params.max_motif_complexity,
        strong_structure_nonmembership=(
            params.max_strong_structure_nonmembership
        ),
        alignment_error=params.max_alignment_error,
        gap_variation=params.max_gap_variation,
        orientation=params.max_orientation_error,
        isolation=params.max_isolation,
        front_clearance=params.max_front_clearance_violation,
        wide_aisle_deficit=params.max_wide_aisle_deficit,
        path_turns=params.max_path_turns,
    )


def _run_direct(
    cfg,
    params: run_store.RunParams,
    initial_population: np.ndarray,
    on_generation,
    elementwise_runner=None,
):
    """CLI と同じ直接配置 NSGA-II の引数を一か所に集約する。"""
    return run_direct_nsga2(
        cfg,
        pop_size=params.population_size,
        n_gen=params.n_generations,
        elementwise_runner=elementwise_runner,
        seed=params.optimization_seed,
        orderliness_limits=_orderliness_limits(params),
        initial_max_trials=params.initial_max_trials,
        initial_genotypes=initial_population,
        mutation_individual_prob=params.mutation_individual_prob,
        mutation_local_weight=params.mutation_local_weight,
        mutation_meso_weight=params.mutation_meso_weight,
        mutation_macro_weight=params.mutation_macro_weight,
        exploration_generations=params.exploration_generations,
        exploration_individual_prob=params.exploration_individual_prob,
        exploration_local_weight=params.exploration_local_weight,
        exploration_meso_weight=params.exploration_meso_weight,
        exploration_macro_weight=params.exploration_macro_weight,
        donor_pool_size=params.donor_pool_size,
        donor_minimum_distance=params.donor_minimum_distance,
        donor_refresh_probability=params.donor_refresh_probability,
        minimum_layout_distance=params.minimum_layout_distance,
        memetic_interval=params.memetic_interval,
        memetic_start_generation=params.memetic_start_generation,
        memetic_parents=params.memetic_parents,
        memetic_neighbors=params.memetic_neighbors,
        memetic_max_neighbor_trials=params.memetic_max_neighbor_trials,
        on_generation=on_generation,
    )


def _population_snapshot(algorithm) -> list[dict]:
    """環境選択後の集団を、UI が扱いやすい素の list[dict] にする。

    物理制約を満たし目的関数値が有限な個体のみを残す。受容可能性制約は
    探索中に未達でも表示し、UI側で達成済み解と区別する。
    同一の設計ベクトルは 1 件にまとめる。
    """
    pop = algorithm.pop
    if pop is None or len(pop) == 0:
        return []

    X = np.asarray(pop.get("X"), dtype=np.int64)
    F = np.atleast_2d(np.asarray(pop.get("F"), dtype=float))

    G_raw = pop.get("G")
    if G_raw is None or np.size(G_raw) == 0:
        G = np.zeros((F.shape[0], 1), dtype=float)
    else:
        G = np.asarray(G_raw, dtype=float)
        if G.ndim == 1:
            G = G.reshape(-1, 1)

    rank_raw = pop.get("rank")
    rank = (
        np.full(F.shape[0], -1, dtype=float)
        if rank_raw is None
        else np.asarray(rank_raw, dtype=float).reshape(-1)
    )

    rows: list[dict] = []
    seen: set[tuple[int, ...]] = set()

    for i in range(F.shape[0]):
        if not np.all(np.isfinite(F[i])):
            continue
        if (
            G.shape[1] < 1
            or not np.isfinite(G[i, 0])
            or G[i, 0] > 0.0
        ):
            continue

        key = tuple(int(v) for v in X[i])
        if key in seen:
            continue
        seen.add(key)

        rows.append(
            {
                "x": list(key),
                "f_move": float(F[i, 0]),
                "f_congestion": float(F[i, 1]),
                "rank": (
                    int(rank[i])
                    if np.isfinite(rank[i])
                    else -1
                ),
                "acceptability_satisfied": bool(
                    G.shape[1] <= 1
                    or (
                        np.all(np.isfinite(G[i, 1:]))
                        and np.all(G[i, 1:] <= 0.0)
                    )
                ),
                "acceptability_violation": float(
                    np.sum(np.maximum(G[i, 1:], 0.0))
                    if G.shape[1] > 1 and np.all(np.isfinite(G[i, 1:]))
                    else float("inf")
                    if G.shape[1] > 1
                    else 0.0
                ),
            }
        )

    # 第1目的の昇順にしておくと UI 側でそのまま線を引ける。
    rows.sort(key=lambda r: (r["f_move"], r["f_congestion"]))
    return rows


def evaluate_fixed_layout(cfg, rd: Path) -> dict:
    """booths.yaml の fixed_position を評価し、比較基準として保存する。

    実際に使われた会場配置(調布市マルシェの実配置)を最適化解と
    同じ条件でシミュレーションし、パレート図に重ねられるようにする。
    `scripts/run_optimization.py` の `fixed_f` と同じ位置づけだが、
    こちらは制約違反や評価失敗でも例外を投げず、
    `available: false` を記録して最適化本体は続行する。
    """
    payload: dict = {"available": False}

    missing = [
        spec.booth_id
        for spec in cfg.booths
        if spec.fixed_position is None
    ]
    if missing or not cfg.booths:
        payload["reason"] = (
            "booths.yaml に fixed_position が未設定のブースがあります"
            f"(booth_id: {missing[:5]}{'…' if len(missing) > 5 else ''})"
            if missing
            else "ブースが定義されていません"
        )
        run_store.write_fixed_layout(rd, payload)
        return payload

    positions = tuple(
        tuple(int(v) for v in spec.fixed_position)
        for spec in cfg.booths
    )

    try:
        # 比較基準は必ず完走させたいのでタイムアウトだけ緩める。
        base_cfg = replace(cfg, timeout_seconds=BASELINE_TIMEOUT_SECONDS)
        ctx = build_context(base_cfg)

        violation = float(compute_violation(base_cfg, ctx.venue, positions))
        payload["violation"] = violation
        payload["feasible"] = violation <= 0.0

        if violation > 0.0:
            # 制約違反でもシミュレーション自体は行い、参考値として残す。
            payload["reason"] = (
                f"固定配置が配置制約を満たしていません(g = {violation:g})。"
                "参考値として表示します。"
            )

        result = evaluate_layout(ctx, positions)

        payload.update(
            available=True,
            label="実会場配置(booths.yaml)",
            positions=[list(p) for p in positions],
            x=[int(v) for v in positions_to_design(cfg, positions)],
            f_move=float(result.f_move),
            f_congestion=float(result.f_congestion),
            timed_out=bool(result.timed_out),
        )

    except Exception as exc:
        payload["available"] = False
        payload["reason"] = f"固定配置の評価に失敗しました: {exc}"

    run_store.write_fixed_layout(rd, payload)
    return payload


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m ui.optimize_worker <run_dir>")

    rd = Path(sys.argv[1]).resolve()
    params = run_store.RunParams.from_dict(
        run_store._read_json(rd / "params.json") or {}
    )
    params.validate()

    cfg = load_config(rd / "config")
    t0 = time.perf_counter()

    run_store.write_status(
        rd,
        state="baseline",
        generation=0,
        n_generations=int(params.n_generations),
        started_at=t0,
        elapsed_seconds=0.0,
        message="実会場配置(booths.yaml)を評価中",
    )

    # 最適化より先に比較基準を確定させる。
    # 第 1 世代のスナップショットが出た時点でパレート図に重ねられる。
    fixed = evaluate_fixed_layout(cfg, rd)
    if fixed.get("available"):
        print(
            "実会場配置(booths.yaml): "
            f"f_move = {fixed['f_move']:.6f}, "
            f"f_congestion = {fixed['f_congestion']:.6e}"
        )
    else:
        print(f"実会場配置の比較は無効: {fixed.get('reason', '不明')}")

    interval = max(1, int(params.snapshot_interval))
    last_gen = {"n": 0}

    def on_generation(algorithm) -> None:
        """毎世代: status を更新。interval 世代ごと: スナップショットを保存。"""
        gen = int(algorithm.n_gen)
        last_gen["n"] = gen
        elapsed = time.perf_counter() - t0

        # 第 1 世代と最終世代は必ず残す(初期集団と最終結果を見るため)。
        is_edge = gen == 1 or gen >= int(params.n_generations)
        if gen % interval == 0 or is_edge:
            rows = _population_snapshot(algorithm)
            run_store.write_snapshot(
                rd,
                {
                    "generation": gen,
                    "n_generations": int(params.n_generations),
                    "elapsed_seconds": elapsed,
                    "n_solutions": len(rows),
                    "solutions": rows,
                },
            )

        run_store.write_status(
            rd,
            state="running",
            generation=gen,
            n_generations=int(params.n_generations),
            elapsed_seconds=elapsed,
            message=f"第 {gen} 世代を評価しました",
        )

    try:
        workers = max(1, int(params.workers))
        limits = _orderliness_limits(params)
        run_store.write_status(
            rd,
            state="initializing",
            generation=0,
            n_generations=int(params.n_generations),
            elapsed_seconds=time.perf_counter() - t0,
            message="完全ランダムな物理制約充足初期集団を生成中",
        )
        warmup_routing_kernels()

        if workers > 1:
            with Pool(workers, initializer=init_worker, initargs=(cfg,)) as pool:
                pool_starmap = partial(pool.starmap, chunksize=1)
                initial_population = prepare_random_feasible_direct_population(
                    cfg,
                    pop_size=params.population_size,
                    seed=params.optimization_seed,
                    orderliness_limits=limits,
                    initial_max_trials=params.initial_max_trials,
                    starmap=pool_starmap,
                )
                run_store.write_status(
                    rd,
                    state="running",
                    message=(
                        f"{design_variable_count(cfg)}変数の"
                        "直接配置NSGA-IIを開始しました"
                    ),
                )
                res = _run_direct(
                    cfg,
                    params,
                    initial_population,
                    on_generation,
                    elementwise_runner=StarmapParallelization(pool_starmap),
                )
        else:
            initial_population = prepare_random_feasible_direct_population(
                cfg,
                pop_size=params.population_size,
                seed=params.optimization_seed,
                orderliness_limits=limits,
                initial_max_trials=params.initial_max_trials,
            )
            run_store.write_status(
                rd,
                state="running",
                message=(
                    f"{design_variable_count(cfg)}変数の"
                    "直接配置NSGA-IIを開始しました"
                ),
            )
            res = _run_direct(
                cfg,
                params,
                initial_population,
                on_generation,
            )

        if params.save_artifacts:
            run_store.write_status(
                rd,
                state="finalizing",
                message="成果物(パレート解 CSV・代表解 JSON)を書き出し中",
            )
            save_results(cfg, res, rd / "results")
            save_direct_metadata(res, rd / "results")

        run_store.write_status(
            rd,
            state="done",
            generation=last_gen["n"],
            elapsed_seconds=time.perf_counter() - t0,
            n_pareto=int(np.atleast_2d(res.F).shape[0]),
            method=(
                f"single_level_direct_{design_variable_count(cfg)}_layout_nsga2"
            ),
            message="直接配置NSGA-IIが完了しました",
        )

    except Exception:
        run_store.write_status(
            rd,
            state="error",
            elapsed_seconds=time.perf_counter() - t0,
            message="エラーで終了しました",
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":  # §16.3 Windows 対応ガード
    main()
