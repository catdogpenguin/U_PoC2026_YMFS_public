"""optimization.py — pymoo Problem・NSGA-II・代表解抽出(§15, §19)。

- 離散変数 + 評価時丸め方式(§15.2)
- 制約は単一の違反量 g(n_ieq_constr = 1)
- 初期集団はリジェクションサンプリングで有効配置のみ生成(§15.3)
- 交叉 SBX(eta=15)、突然変異 PM(eta=20)は pymoo 既定
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize

from .constraints import compute_violation, to_integer_layout
from .models import Config
from .parallel import evaluate_vector
from .rng import sampling_rng
from .venue import build_base_venue, position_bounds


def variable_bounds(
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """ブース座標の整数上下限を返す。

    ブース本体・受付・退出セル・待機列を含む形状全体が
    会場内に収まる範囲を使用する。
    """
    xl = []
    xu = []

    for booth_id, _spec in enumerate(cfg.booths):
        xmin, xmax, ymin, ymax = position_bounds(
            cfg,
            booth_id,
        )

        xl += [xmin, ymin]
        xu += [xmax, ymax]

    return (
        np.asarray(xl, dtype=np.int64),
        np.asarray(xu, dtype=np.int64),
    )


class LayoutProblem(ElementwiseProblem):
    """12 次元・2 目的・制約 1 本の配置最適化問題(§15.1)。"""

    def __init__(self, cfg: Config, **kwargs) -> None:
        self.cfg = cfg
        xl, xu = variable_bounds(cfg)

        super().__init__(
            n_var=len(cfg.booths) * 2,
            n_obj=2,
            n_ieq_constr=1,
            xl=xl,
            xu=xu,
            vtype=int,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f_move, f_cong, g, kpis = evaluate_vector(x, self.cfg)
        out["F"] = [f_move, f_cong]
        out["G"] = [g]
        out["kpis"] = kpis


class ValidLayoutSampling(Sampling):
    """§15.3: 有効配置のみを生成するリジェクションサンプリング。"""

    def __init__(self, cfg: Config, max_trials: int = 1000) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_trials = max_trials

    def _do(self, problem, n_samples, **kwargs):
        cfg = self.cfg
        venue = build_base_venue(cfg)
        rng = sampling_rng(cfg.evaluation_seed)

        lower = problem.xl
        upper = problem.xu

        out = np.empty(
            (n_samples, problem.n_var),
            dtype=np.int64,
        )

        for i in range(n_samples):
            out[i] = sample_valid_layout(
                rng,
                lower,
                upper,
                cfg,
                venue,
                self.max_trials,
            )

        return out


def sample_valid_layout(
    rng,
    lower_bounds,
    upper_bounds,
    cfg,
    venue,
    max_trials=1000,
):
    """整数の有効配置をリジェクションサンプリングする。"""
    lower = np.asarray(
        lower_bounds,
        dtype=np.int64,
    )
    upper = np.asarray(
        upper_bounds,
        dtype=np.int64,
    )

    for _ in range(max_trials):
        # rng.integers の上限は排他的であるため、
        # upper + 1 として上下限を両方含める。
        layout = rng.integers(
            low=lower,
            high=upper + 1,
            dtype=np.int64,
        )

        positions = to_integer_layout(layout)

        if compute_violation(
            cfg,
            venue,
            positions,
        ) == 0:
            return layout

    raise RuntimeError(
        "有効配置を生成できません。"
        "会場が狭すぎる可能性があります。"
    )


class RandomValidBoothMutation(Mutation):
    """1個のブースだけを、制約を満たす別位置へランダム移動する突然変異。

    各個体について確率 ``prob`` で突然変異を行う。

    突然変異を行う場合はブースを1個ランダムに選び、
    そのブースだけを compute_violation(...) == 0 となる
    別座標へ移動する。

    選択したブースに有効な別位置が存在しない場合は、
    元の個体を保持する。
    """

    def __init__(
        self,
        cfg: Config,
        prob: float = 0.2,
        seed: int = 1,
    ) -> None:
        # pymooのMutation基底クラスは、_doを全個体に適用した後で
        # 個体単位の突然変異確率を適用する。
        #
        # 今回は制約判定の計算を突然変異対象の個体だけに限定するため、
        # 基底クラス側の確率を1.0にし、
        # このクラス内で確率判定を行う。
        super().__init__(prob=1.0)

        if not 0.0 <= prob <= 1.0:
            raise ValueError(
                f"mutation probability must be in [0, 1]: {prob}"
            )

        self.cfg = cfg
        self.individual_prob = float(prob)

        # 制約判定に使用するベース会場。
        self.venue = build_base_venue(cfg)

        # 古いpymooなど、_doへrandom_stateが渡らない場合に
        # 使用する予備の乱数生成器。
        self._fallback_rng = np.random.default_rng(seed)

    @staticmethod
    def _randint(
        rng,
        low: int,
        high: int,
    ) -> int:
        """Generatorと旧RandomStateの両方に対応した整数乱数。"""

        if hasattr(rng, "integers"):
            return int(rng.integers(low, high))

        return int(rng.randint(low, high))

    def _do(
        self,
        problem,
        X,
        random_state=None,
        **kwargs,
    ):
        rng = (
            random_state
            if random_state is not None
            else self._fallback_rng
        )

        # SBX後は、整数値ではあるがfloat型になっていることがある。
        # 例: 3.0, 7.0
        #
        # to_integer_layout()へ渡せるように整数配列へ変換する。
        Y = np.rint(
            np.asarray(X)
        ).astype(
            np.int64,
            copy=True,
        )

        lower = np.asarray(
            problem.xl,
            dtype=np.int64,
        )

        upper = np.asarray(
            problem.xu,
            dtype=np.int64,
        )

        n_booths = len(self.cfg.booths)

        for individual_id in range(len(Y)):
            # -----------------------------------------------------
            # この個体に突然変異を適用するか判定
            # -----------------------------------------------------
            if float(rng.random()) > self.individual_prob:
                continue

            original = Y[individual_id].copy()

            # -----------------------------------------------------
            # 移動するブースを1個だけランダムに選択
            # -----------------------------------------------------
            booth_id = self._randint(
                rng,
                0,
                n_booths,
            )

            x_index = 2 * booth_id
            y_index = x_index + 1

            old_x = int(original[x_index])
            old_y = int(original[y_index])

            # -----------------------------------------------------
            # 選択したブースが取り得る全ての別座標を列挙
            # -----------------------------------------------------
            candidate_positions = [
                (x_pos, y_pos)
                for x_pos in range(
                    int(lower[x_index]),
                    int(upper[x_index]) + 1,
                )
                for y_pos in range(
                    int(lower[y_index]),
                    int(upper[y_index]) + 1,
                )
                if (x_pos, y_pos) != (old_x, old_y)
            ]

            if not candidate_positions:
                # 別位置が存在しない場合は変更しない。
                continue

            # -----------------------------------------------------
            # 候補座標をランダムな順番で確認
            # -----------------------------------------------------
            candidate_order = rng.permutation(
                len(candidate_positions)
            )

            for candidate_id_raw in candidate_order:
                candidate_id = int(candidate_id_raw)

                new_x, new_y = candidate_positions[
                    candidate_id
                ]

                # 元の個体をコピーし、選択したブースだけを移動する。
                candidate = original.copy()

                candidate[x_index] = new_x
                candidate[y_index] = new_y

                positions = to_integer_layout(candidate)

                # -------------------------------------------------
                # 配置全体が全制約を満たす場合だけ採用
                # -------------------------------------------------
                if compute_violation(
                    self.cfg,
                    self.venue,
                    positions,
                ) == 0:
                    Y[individual_id] = candidate
                    break

            # 全候補を確認しても有効な別位置がなければ、
            # Y[individual_id]は元の配置のままになる。

        return Y
    
class ObjectiveHistoryCallback(Callback):
    """各世代の環境選択後集団の目的関数値と制約値を保存する。"""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict] = []

    def notify(self, algorithm) -> None:
        # その世代の環境選択後集団を取得
        pop = algorithm.pop

        # 目的関数値
        F = np.asarray(
            pop.get("F"),
            dtype=float,
        ).copy()

        # 制約違反量
        G_raw = pop.get("G")

        if G_raw is None:
            G = None
        else:
            G = np.asarray(
                G_raw,
                dtype=float,
            ).copy()

        self.records.append(
            {
                "generation": int(algorithm.n_gen),
                "F": F,
                "G": G,
            }
        )


def _feasible_objectives(record: dict) -> np.ndarray:
    """履歴1世代分から、有限かつ実行可能な目的関数値だけを返す。"""

    F = np.atleast_2d(
        np.asarray(
            record["F"],
            dtype=float,
        )
    )

    # NaNやinfを含む目的関数値を除外
    finite = np.all(
        np.isfinite(F),
        axis=1,
    )

    G = record.get("G")

    if G is None or np.size(G) == 0:
        feasible = np.ones(
            F.shape[0],
            dtype=bool,
        )
    else:
        G = np.asarray(
            G,
            dtype=float,
        )

        if G.ndim == 1:
            G = G.reshape(-1, 1)

        # pymooではG <= 0が制約を満たす解
        feasible = np.all(
            G <= 0.0,
            axis=1,
        )

    return F[finite & feasible]


def _non_dominated_2d(F: np.ndarray) -> np.ndarray:
    """2目的・最小化問題の非劣解集合を返す。"""

    F = np.atleast_2d(
        np.asarray(
            F,
            dtype=float,
        )
    )

    if F.size == 0:
        return np.empty(
            (0, 2),
            dtype=float,
        )

    if F.shape[1] != 2:
        raise ValueError(
            "HV calculation currently supports exactly "
            f"two objectives: {F.shape[1]}"
        )

    # 同じ目的関数ベクトルを除去
    points = np.unique(
        F,
        axis=0,
    )

    # 第1目的を昇順、第1目的が同値なら第2目的を昇順に並べる
    order = np.lexsort(
        (
            points[:, 1],
            points[:, 0],
        )
    )
    points = points[order]

    # 左から走査し、第2目的の最良値を更新する点だけを残す
    nondominated = []
    best_second = np.inf

    for point in points:
        if point[1] < best_second:
            nondominated.append(point)
            best_second = point[1]

    return np.asarray(
        nondominated,
        dtype=float,
    )


def _hypervolume_2d(
    F: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    """2目的・最小化問題のHVを矩形和により計算する。"""

    ref = np.asarray(
        reference_point,
        dtype=float,
    ).reshape(-1)

    if ref.size != 2:
        raise ValueError(
            "reference_point must contain exactly two values"
        )

    F = np.atleast_2d(
        np.asarray(
            F,
            dtype=float,
        )
    )

    if F.size == 0:
        return 0.0

    # 参照点よりも悪い成分を持つ解はHVに寄与しない
    inside = np.all(
        F < ref,
        axis=1,
    )

    nd = _non_dominated_2d(
        F[inside]
    )

    if nd.size == 0:
        return 0.0

    hv = 0.0
    previous_y = float(ref[1])

    for x_value, y_value in nd:
        width = float(
            ref[0] - x_value
        )
        height = float(
            previous_y - y_value
        )

        if width > 0.0 and height > 0.0:
            hv += width * height

        previous_y = min(
            previous_y,
            float(y_value),
        )

    return float(hv)


def calculate_hv_history(
    res,
    reference_point: np.ndarray | None = None,
    reference_margin: float = 0.1,
) -> tuple[pd.DataFrame, np.ndarray]:
    """各世代の実行可能な非劣解集合に対するHVを計算する。

    全世代で同一の参照点を使用する。

    reference_pointがNoneの場合は、全世代の実行可能解から
    各目的の最大値を求め、目的値レンジのreference_margin分だけ
    悪い方向へ余白を加えた点を参照点とする。
    """

    records = getattr(
        res,
        "objective_history",
        None,
    )

    if not records:
        raise ValueError(
            "Objective history is unavailable. "
            "Run the optimizer with ObjectiveHistoryCallback enabled."
        )

    if reference_margin <= 0.0:
        raise ValueError(
            "reference_margin must be positive"
        )

    # 各世代の実行可能解
    feasible_sets = [
        _feasible_objectives(record)
        for record in records
    ]

    nonempty = [
        F
        for F in feasible_sets
        if F.size > 0
    ]

    if not nonempty:
        raise ValueError(
            "No feasible objective vectors were found in the history"
        )

    all_feasible = np.vstack(
        nonempty
    )

    if all_feasible.shape[1] != 2:
        raise ValueError(
            "HV calculation currently supports exactly "
            f"two objectives: {all_feasible.shape[1]}"
        )

    # ---------------------------------------------------------
    # 全世代共通の参照点を決定
    # ---------------------------------------------------------
    if reference_point is None:
        ideal = np.min(
            all_feasible,
            axis=0,
        )
        nadir = np.max(
            all_feasible,
            axis=0,
        )

        span = nadir - ideal

        # 目的値レンジが0の軸にも正の余白を与える
        fallback_scale = np.maximum(
            np.abs(nadir),
            1.0,
        )

        padding = np.where(
            span > 0.0,
            reference_margin * span,
            reference_margin * fallback_scale,
        )

        ref = nadir + padding

    else:
        ref = np.asarray(
            reference_point,
            dtype=float,
        ).reshape(-1)

        if ref.size != 2:
            raise ValueError(
                "reference_point must contain exactly two values"
            )

        if not np.all(
            np.isfinite(ref)
        ):
            raise ValueError(
                "reference_point contains non-finite values: "
                f"{ref}"
            )

        max_objectives = np.max(
            all_feasible,
            axis=0,
        )

        if np.any(
            ref <= max_objectives
        ):
            raise ValueError(
                "reference_point must be strictly worse than every "
                "feasible objective vector. "
                f"max objectives={max_objectives}, "
                f"reference_point={ref}"
            )

    # ---------------------------------------------------------
    # 世代ごとのHVを計算
    # ---------------------------------------------------------
    rows = []

    for record, feasible_F in zip(
        records,
        feasible_sets,
    ):
        nd = _non_dominated_2d(
            feasible_F
        )

        rows.append(
            {
                "generation": int(
                    record["generation"]
                ),
                "hypervolume": _hypervolume_2d(
                    nd,
                    ref,
                ),
                "n_feasible": int(
                    feasible_F.shape[0]
                ),
                "n_nondominated": int(
                    nd.shape[0]
                ),
                "reference_f_move": float(
                    ref[0]
                ),
                "reference_f_congestion": float(
                    ref[1]
                ),
            }
        )

    return (
        pd.DataFrame(rows),
        ref,
    )

def run_nsga2(
    cfg: Config,
    pop_size: int | None = None,
    n_gen: int | None = None,
    elementwise_runner=None,
    seed: int = 1,
    verbose: bool = True,
    mutation_prob: float = 0.2, # 突然変異率
):
    """NSGA-II を実行し pymoo の Result を返す。"""
    kwargs = {}
    if elementwise_runner is not None:
        kwargs["elementwise_runner"] = elementwise_runner
    problem = LayoutProblem(cfg, **kwargs)
    algorithm = NSGA2(
        pop_size=pop_size or cfg.population_size,
        sampling=ValidLayoutSampling(cfg),

        # SBX自体の計算は実数空間で行い、
        # 生成直後に整数へ修復する。
        crossover=SBX(
            eta=15,
            prob=0.9,
            vtype=float,
            repair=RoundingRepair(),
        ),

        # 1個のブースだけを、配置全体が制約を満たす
        # 別位置へランダムに移動する突然変異。
        mutation=RandomValidBoothMutation(
            cfg=cfg,
            prob=mutation_prob,
            seed=seed,
        ),

        # 整数化後に同じ設計ベクトルとなった個体を除去する。
        eliminate_duplicates=True,
    )

    # 各世代の目的関数値と制約値を記録する
    history_callback = ObjectiveHistoryCallback()

    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen or cfg.n_generations),
        seed=seed,
        verbose=verbose,
        callback=history_callback,
        save_history=False,
    )

    # pymooのalgorithm履歴全体は保存せず、
    # HV計算に必要なFとGだけをResultへ付加する。
    res.objective_history = history_callback.records

    return res


# ---------------------------------------------------------------- 代表解と出力(§19)
def select_representatives(F: np.ndarray) -> dict[str, int]:
    """§19.2: 移動重視・混雑重視・バランス(理想点距離最小)のインデックス。"""
    i_move = int(np.argmin(F[:, 0]))
    i_cong = int(np.argmin(F[:, 1]))
    fmin, fmax = F.min(axis=0), F.max(axis=0)
    span = np.where(fmax - fmin > 0, fmax - fmin, 1.0)
    Fn = (F - fmin) / span
    i_knee = int(np.argmin(np.sqrt((Fn**2).sum(axis=1))))
    return {"best_movement": i_move, "best_congestion": i_cong, "knee": i_knee}


def save_results(cfg: Config, res, out_dir: str | Path) -> dict:
    """§19.1 の成果物一式を書き出す。戻り値は代表解 {名前: 12 次元 list}。"""
    from .parallel import build_context, evaluate_layout  # 遅延 import

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X = np.asarray(
        np.atleast_2d(res.X),
        dtype=np.int64,
    )
    F = np.atleast_2d(res.F)

    # パレート解ごとに KPI を再取得(有効解のみが res に残る)
    ctx = build_context(cfg)
    rows = []
    for x, f in zip(X, F):
        positions = to_integer_layout(x)
        r = evaluate_layout(ctx, positions)
        row = {
            f"x{j}": int(v)
            for j, v in enumerate(x)
        }
        row.update(
            f_move=f[0],
            f_congestion=f[1],
            mean_queue_seconds=r.kpis["mean_queue_seconds"],
            mean_blocked_seconds=r.kpis["mean_blocked_seconds"],
            experience_success_rate=r.kpis["experience_success_rate"],
            give_up_total=r.kpis["give_up_total"],
            max_cell_occupancy_ratio=r.kpis["max_cell_occupancy_ratio"],
            mean_happiness=r.kpis["mean_happiness"],
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "pareto_solutions.csv", index=False)

    reps = select_representatives(F)
    name_to_file = {
        "best_movement": "best_movement_layout.json",
        "best_congestion": "best_congestion_layout.json",
        "knee": "knee_layout.json",
    }
    rep_layouts = {}
    for name, idx in reps.items():
        payload = {
            "design_vector": [
                int(v)
                for v in X[idx]
            ],
            "positions": [
                list(p)
                for p in to_integer_layout(X[idx])
            ],
            "f_move": float(F[idx, 0]),
            "f_congestion": float(F[idx, 1]),
        }
        (out / name_to_file[name]).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rep_layouts[name] = payload
    return rep_layouts
