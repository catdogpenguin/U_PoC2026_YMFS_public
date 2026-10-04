"""通常ブース3変数・テーブル2変数の直接表現による1層型NSGA-II。

通常ブースは ``(x,y,theta)``、方向を持たないテーブルは ``(x,y)`` を染色体に
保持する。本設定では28×3+18×2=120次元である。整然さは最終配置の
幾何形状から計算した制約として扱う。初期集団はbooths.yamlを参照せず、
位置・向きをランダム貪欲法で決め、幾何制約と1セル幅到達性を満たす
配置から生成する。同一仕様・同一広場のテーブルは座標順に正規化し、
識別子の交換だけで別個体になる冗長性を除く。正面2セル空間を含む物理制約は
初期化・交叉・突然変異で必ず守り、それ以外の整然さは受容可能性制約として
制約付き環境選択へ渡す。経路の方向転換回数の違反は他成分より強く扱う。
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.survival import Survival
from pymoo.optimize import minimize
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

from .constraints import (
    PlacementGeometry,
    booth_variable_indices,
    build_placement_geometry,
    canonicalize_table_design,
    compute_violation,
    compute_width_reachability_profile,
    design_variable_count,
    positions_to_design,
    tables_are_equivalent,
    to_integer_layout,
)
from .models import WALKABLE_TYPES, Config
from .optimization import (
    ObjectiveHistoryCallback,
    _booth_claimed_cells,
    _booth_fits,
    _greedy_layout_attempt,
    variable_bounds,
)
from .parallel import PENALTY_OBJECTIVE, evaluate_layout, get_context
from .routing import INF_DIST, bfs_distance_field, descend_path_indices
from .venue import (
    FRONT_CLEARANCE_DEPTH_CELLS,
    TARGET_CORRIDOR_WIDTH_CELLS,
    apply_layout,
    booth_cells_of,
    booth_exit_cell,
    build_base_venue,
    front_clearance_cells,
    oriented_booth_size,
    passable_corridor_mask,
    queue_cells_of,
    reception_cell,
)


LANE_ALIGNMENT_TOLERANCE_CELLS = 1.0
MAX_LANE_EDGE_GAP_CELLS = 6.0
MIN_STRONG_LANE_SIZE = 2
BOUNDARY_BAND_CELLS = 5.0
BOUNDARY_ALIGNMENT_TOLERANCE_CELLS = 4.0
MIN_BOUNDARY_CHAIN_SIZE = 1
MOTIF_TARGET_SIZE = 3
MIN_STRONG_MOTIF_SIZE = 3
WIDE_AISLE_WIDTH_CELLS = 3
VISUAL_DIVERSITY_BIN_CELLS = 3
VISUAL_FOOTPRINT_WEIGHT = 0.80
LAYOUT_NOVELTY_CROWDING_WEIGHT = 0.35
CONSTRAINT_CACHE_MAXSIZE = 50_000
#: path_turns を測る対象。テーブル・ゴミステーションは付帯設備なので除外し、
#: 体験ブース同士の移動だけを見る(来場者の主要動線)。
PATH_TURN_EXCLUDE_FACILITIES = True
ORDERLINESS_COMPONENT_KEYS = (
    "organization_nonmembership",
    "motif_complexity",
    "strong_structure_nonmembership",
    "alignment_error",
    "gap_variation",
    "orientation",
    "isolation",
    "front_clearance",
    "wide_aisle_deficit",
    "path_turns",
)
#: organization_nonmembership と motif_complexity は説明用の監査値として残す。
#: 前者は isolation/strong_structure と重なり、後者は同じ整然配置でも分割数に
#: 依存するため、実際の制約からは外す。正面空間も物理制約との重複を避ける。
ORDERLINESS_CONSTRAINT_KEYS = (
    "strong_structure_nonmembership",
    "alignment_error",
    "gap_variation",
    "orientation",
    "isolation",
    "wide_aisle_deficit",
    "path_turns",
)
#: 受容可能性制約に違反した個体同士では、動線の分かりやすさを強く優先する。
PATH_TURN_CONSTRAINT_WEIGHT = 3.0


@dataclass(frozen=True)
class OrderlinessLimits:
    """列・外周追従の双方を許す、説明可能な配置構造の成分別上限。"""

    organization_nonmembership: float = 0.20
    motif_complexity: float = 1.20
    #: 各区画・用途群で、3個以上の列/外周チェーンに属さない割合。
    strong_structure_nonmembership: float = 0.55
    alignment_error: float = 0.36
    gap_variation: float = 0.30
    orientation: float = 0.25
    isolation: float = 0.20
    front_clearance: float = 0.0
    #: 主要ブースのうち、入口・出口から3セル幅で到達できない割合。
    wide_aisle_deficit: float = 0.35
    #: 体験ブース間の最短経路上で許容する平均方向転換回数。
    #: 旧上限4.0では最終解が3.87～4.00へ張り付いたため、探索の自由度を
    #: 少し戻す。違反量は引き続き3倍評価するので、超過時の優先度は強い。
    path_turns: float = 4.5

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative: {value}")

    def as_dict(self) -> dict[str, float]:
        return {
            "organization_nonmembership": float(self.organization_nonmembership),
            "motif_complexity": float(self.motif_complexity),
            "strong_structure_nonmembership": float(
                self.strong_structure_nonmembership
            ),
            "alignment_error": float(self.alignment_error),
            "gap_variation": float(self.gap_variation),
            "orientation": float(self.orientation),
            "isolation": float(self.isolation),
            "front_clearance": float(self.front_clearance),
            "wide_aisle_deficit": float(self.wide_aisle_deficit),
            "path_turns": float(self.path_turns),
        }

    def scaled(self, factor: float) -> "OrderlinessLimits":
        if factor < 0.0:
            raise ValueError("orderliness limit scale must be non-negative")
        values = {key: value * factor for key, value in self.as_dict().items()}
        return OrderlinessLimits(**values)


def _visual_layout_signatures(
    cfg: Config,
    positions_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """識別子に依存しない粗視化占有形状と正面方向の署名を返す。

    同じ形状のブースを別IDへ割り当てただけの解を「別配置」と数えないため、
    3セル角のビン上で、区画×通常/テーブル別の本体占有と、区画×向き別の
    通常ブース中心を記録する。3セル未満の微調整より、列・島・空白の位置が
    変わったかを優先して測る。
    """

    positions_batch = np.asarray(positions_batch, dtype=np.int64)
    if positions_batch.ndim == 2:
        positions_batch = positions_batch[None, ...]
    n_designs = int(len(positions_batch))
    bin_size = int(VISUAL_DIVERSITY_BIN_CELLS)
    n_x_bins = int(np.ceil(cfg.grid_width / bin_size))
    n_y_bins = int(np.ceil(cfg.grid_height / bin_size))
    cells_per_layer = n_x_bins * n_y_bins
    areas = sorted({str(booth.area) for booth in cfg.booths})
    area_indices = {area: index for index, area in enumerate(areas)}
    footprints = np.zeros(
        (n_designs, len(areas) * 2 * cells_per_layer),
        dtype=np.float32,
    )
    facings = np.zeros(
        (n_designs, len(areas) * 4 * cells_per_layer),
        dtype=np.float32,
    )

    for design_id, positions in enumerate(positions_batch):
        for booth_id, booth in enumerate(cfg.booths):
            x_pos, y_pos, theta = (
                int(value) for value in positions[booth_id]
            )
            width, height = oriented_booth_size(
                cfg, booth_id, 0 if booth.is_table else theta
            )
            area_index = area_indices[str(booth.area)]
            kind_index = 1 if booth.is_table else 0
            footprint_offset = (
                (area_index * 2 + kind_index) * cells_per_layer
            )
            for cell_y in range(y_pos, y_pos + height):
                if not 0 <= cell_y < cfg.grid_height:
                    continue
                for cell_x in range(x_pos, x_pos + width):
                    if not 0 <= cell_x < cfg.grid_width:
                        continue
                    bin_index = (
                        (cell_y // bin_size) * n_x_bins
                        + cell_x // bin_size
                    )
                    footprints[design_id, footprint_offset + bin_index] = 1.0

            if booth.is_table:
                continue
            center_x = int((x_pos + (width - 1) / 2.0) // bin_size)
            center_y = int((y_pos + (height - 1) / 2.0) // bin_size)
            if not (0 <= center_x < n_x_bins and 0 <= center_y < n_y_bins):
                continue
            facing_offset = (
                (area_index * 4 + int(theta) % 4) * cells_per_layer
            )
            facings[
                design_id,
                facing_offset + center_y * n_x_bins + center_x,
            ] = 1.0
    return footprints, facings


def _pairwise_binary_jaccard(signatures: np.ndarray) -> np.ndarray:
    """0/1署名の全組合せJaccard距離を行列で返す。"""

    if len(signatures) == 0:
        return np.empty((0, 0), dtype=float)
    intersection = signatures @ signatures.T
    counts = np.diag(intersection)
    union = counts[:, None] + counts[None, :] - intersection
    similarity = np.divide(
        intersection,
        union,
        out=np.ones_like(intersection),
        where=union > 0.0,
    )
    distance = np.asarray(1.0 - similarity, dtype=float)
    np.fill_diagonal(distance, 0.0)
    return distance


def normalized_layout_distance(cfg: Config, first, second) -> float:
    """人が見る本体形状と正面方向の差を0～1へ正規化した距離。"""

    designs = np.atleast_2d(np.asarray([first, second]))
    positions = np.asarray(
        [
            to_integer_layout(canonicalize_table_design(cfg, row), cfg)
            for row in designs
        ],
        dtype=np.int64,
    )
    footprints, facings = _visual_layout_signatures(cfg, positions)
    footprint_distance = _pairwise_binary_jaccard(footprints)[0, 1]
    facing_distance = _pairwise_binary_jaccard(facings)[0, 1]
    return float(
        VISUAL_FOOTPRINT_WEIGHT * footprint_distance
        + (1.0 - VISUAL_FOOTPRINT_WEIGHT) * facing_distance
    )


def pairwise_normalized_layout_distances(cfg: Config, designs) -> np.ndarray:
    """集団内の識別子非依存な全配置距離を一括計算する。"""

    design_matrix = np.atleast_2d(np.asarray(designs))
    n_designs = int(len(design_matrix))
    if n_designs == 0:
        return np.empty((0, 0), dtype=float)
    positions = np.asarray(
        [
            to_integer_layout(canonicalize_table_design(cfg, row), cfg)
            for row in design_matrix
        ],
        dtype=np.int64,
    )
    footprints, facings = _visual_layout_signatures(cfg, positions)
    return (
        VISUAL_FOOTPRINT_WEIGHT * _pairwise_binary_jaccard(footprints)
        + (1.0 - VISUAL_FOOTPRINT_WEIGHT)
        * _pairwise_binary_jaccard(facings)
    )


class DirectDiversitySurvival(Survival):
    """NSGA-II順位を守りつつ、近似同一配置を後回しにする環境選択。"""

    def __init__(self, cfg: Config, minimum_layout_distance: float = 0.08) -> None:
        super().__init__(filter_infeasible=False)
        if not 0.0 <= minimum_layout_distance <= 1.0:
            raise ValueError("minimum_layout_distance must be in [0, 1]")
        self.cfg = cfg
        self.minimum_layout_distance = float(minimum_layout_distance)
        self.ranking = RankAndCrowding()
        self.history: list[dict] = []

    @staticmethod
    def _crowding_value(individual) -> float:
        value = individual.get("crowding")
        return float(value) if value is not None else float("-inf")

    @staticmethod
    def _novelty_scores(
        layout_distances: np.ndarray,
        indices: list[int] | np.ndarray,
        n_neighbors: int = 3,
    ) -> dict[int, float]:
        """同じ優先群における近傍配置距離の平均を返す。"""

        members = np.asarray(indices, dtype=np.int64)
        if members.size == 0:
            return {}
        if members.size == 1:
            return {int(members[0]): 1.0}
        submatrix = np.asarray(
            layout_distances[np.ix_(members, members)],
            dtype=float,
        )
        np.fill_diagonal(submatrix, np.inf)
        k = min(max(1, int(n_neighbors)), int(members.size) - 1)
        nearest = np.partition(submatrix, kth=k - 1, axis=1)[:, :k]
        return {
            int(index): float(np.mean(nearest[row]))
            for row, index in enumerate(members)
        }

    @staticmethod
    def _crowding_plus_novelty(crowding: float, novelty: float) -> float:
        """目的空間crowdingを主とし、見た目の新規性を穏やかに加える。"""

        if not np.isfinite(crowding):
            return crowding
        return float(crowding + LAYOUT_NOVELTY_CROWDING_WEIGHT * novelty)

    def _do(
        self,
        problem,
        pop,
        *args,
        n_survive=None,
        random_state=None,
        **kwargs,
    ):
        n_survive = min(int(n_survive or len(pop)), len(pop))
        if len(pop) == 0:
            return pop
        rng = random_state if random_state is not None else np.random.default_rng()
        # G[:, 0] は必須物理制約、残りは受容可能性制約である。物理的に
        # 実行可能な個体を、整然性違反の小ささにかかわらず必ず先に扱う。
        # これにより受容可能性違反が減っても物理違反を相殺できない。
        raw_cv = np.asarray(pop.get("CV"), dtype=float).reshape(-1)
        if raw_cv.size == len(pop):
            cv = raw_cv
        else:
            # 評価属性が欠落した個体を実行可能扱いにはしない。ただしCVを空のまま
            # 返すとpymooの進捗表示がmin()できないため、有限の違反値を補う。
            cv = np.full(len(pop), float(PENALTY_OBJECTIVE), dtype=float)
            for individual in pop:
                individual.set(
                    "CV",
                    np.asarray([PENALTY_OBJECTIVE], dtype=float),
                )
        raw_g = pop.get("G")
        if raw_g is None or np.size(raw_g) == 0:
            constraints = np.full(
                (len(pop), int(problem.n_ieq_constr)),
                float(PENALTY_OBJECTIVE),
                dtype=float,
            )
        else:
            constraints = np.asarray(raw_g, dtype=float)
            if constraints.ndim == 1:
                constraints = constraints.reshape(-1, 1)
            if constraints.shape[0] != len(pop):
                constraints = np.full(
                    (len(pop), int(problem.n_ieq_constr)),
                    float(PENALTY_OBJECTIVE),
                    dtype=float,
                )
        physical_violation = np.maximum(constraints[:, 0], 0.0)
        acceptability_violation = (
            np.sum(np.maximum(constraints[:, 1:], 0.0), axis=1)
            if constraints.shape[1] > 1
            else np.zeros(len(pop), dtype=float)
        )
        physical_feasible_mask = (
            np.isfinite(physical_violation) & (physical_violation <= 0.0)
        )
        feasible_mask = physical_feasible_mask & (
            np.isfinite(acceptability_violation)
            & (acceptability_violation <= 0.0)
        )
        feasible = np.where(feasible_mask)[0]
        acceptability_infeasible = np.where(
            physical_feasible_mask & ~feasible_mask
        )[0]
        physical_infeasible = np.where(~physical_feasible_mask)[0]
        raw_width4 = pop.get("width4_achievement_count")
        width4_counts = (
            np.zeros(len(pop), dtype=np.int64)
            if raw_width4 is None
            else np.rint(np.asarray(raw_width4, dtype=float).reshape(-1)).astype(np.int64)
        )
        layout_distances = pairwise_normalized_layout_distances(
            self.cfg,
            pop.get("X"),
        )

        if feasible.size:
            # 全実行可能個体のrank/crowdingを付与してから優先順を作る。
            self.ranking.do(
                problem,
                pop[feasible],
                n_survive=len(feasible),
                random_state=rng,
                **kwargs,
            )
            novelty: dict[int, float] = {}
            ranks = sorted(
                {int(pop[int(index)].get("rank")) for index in feasible}
            )
            for rank in ranks:
                rank_members = [
                    int(index)
                    for index in feasible
                    if int(pop[int(index)].get("rank")) == rank
                ]
                novelty.update(
                    self._novelty_scores(layout_distances, rank_members)
                )
            tie = {int(index): float(rng.random()) for index in feasible}
            feasible_priority = sorted(
                (int(index) for index in feasible),
                key=lambda index: (
                    int(pop[index].get("rank")),
                    -self._crowding_plus_novelty(
                        self._crowding_value(pop[index]),
                        novelty.get(index, 0.0),
                    ),
                    tie[index],
                ),
            )
        else:
            feasible_priority = []

        acceptability_tie = {
            int(index): float(rng.random()) for index in acceptability_infeasible
        }
        acceptability_priority = sorted(
            (int(index) for index in acceptability_infeasible),
            key=lambda index: (
                float(acceptability_violation[index]),
                acceptability_tie[index],
            ),
        )
        physical_tie = {
            int(index): float(rng.random()) for index in physical_infeasible
        }
        physical_infeasible_priority = sorted(
            (int(index) for index in physical_infeasible),
            key=lambda index: (
                float(physical_violation[index]),
                float(cv[index]),
                physical_tie[index],
            ),
        )

        selected: list[int] = []
        deferred: list[int] = []
        minimum_distances: dict[int, float] = {}
        feasible_groups = [
            [
                index
                for index in feasible_priority
                if int(pop[index].get("rank")) == rank
            ]
            for rank in sorted(
                {int(pop[index].get("rank")) for index in feasible_priority}
            )
        ]

        def select_tier(groups: list[list[int]]) -> None:
            """同じ実行可能性階層では、近似重複を次rankの後へ回す。"""

            tier_deferred: list[int] = []
            for group in groups:
                for index in group:
                    if len(selected) >= n_survive:
                        return
                    if not selected:
                        minimum_distance = float("inf")
                    else:
                        minimum_distance = float(
                            np.min(
                                layout_distances[
                                    int(index),
                                    np.asarray(selected, dtype=np.int64),
                                ]
                            )
                        )
                    minimum_distances[index] = minimum_distance
                    if (
                        minimum_distance + 1.0e-12
                        >= self.minimum_layout_distance
                    ):
                        selected.append(index)
                    else:
                        tier_deferred.append(index)

            # 多様な下位rankを一巡した後も空きがある場合だけ、同じ階層内の
            # 近似重複を元のrank/crowding順で戻す。これはハード制約ではない。
            deferred.extend(tier_deferred)
            for index in tier_deferred:
                if len(selected) >= n_survive:
                    return
                selected.append(index)

        # 物理制約 > 受容可能性制約 > 目的関数rank の階層は崩さない。
        # 4セル幅達成数の旧lexicographic優先は、目的関数より強い隠れ制約に
        # なっていたため撤廃し、後述の緩い3セル幅受容性制約だけを使う。
        select_tier(feasible_groups)
        if len(selected) < n_survive:
            select_tier([acceptability_priority])
        if len(selected) < n_survive:
            select_tier([physical_infeasible_priority])

        # 評価属性の欠落などがあっても、非空の候補集団から空の生存集団を返さない。
        # 通常は上のpriority_groupsだけで必ずn_survive件に達する。
        if len(selected) < n_survive:
            selected_set = set(selected)
            remaining = [index for index in range(len(pop)) if index not in selected_set]
            remaining.sort(
                key=lambda index: (
                    int(not physical_feasible_mask[index]),
                    float(physical_violation[index]),
                    float(acceptability_violation[index]),
                    float(cv[index]) if np.isfinite(cv[index]) else float("inf"),
                    int(index),
                )
            )
            selected.extend(remaining[: n_survive - len(selected)])

        survivor_indices = np.asarray(selected, dtype=np.int64)
        survivors = pop[survivor_indices]

        # 生存個体だけを母集団としてrank/crowdingを付け直す。目的空間の
        # crowdingへ小さな構造的新規性を加え、同じ見た目の非劣解だけが
        # 親として反復選択されることを防ぐ（rankと端点の保持は崩さない）。
        survivor_feasible = np.asarray(
            survivors.get("FEAS"),
            dtype=bool,
        ).reshape(-1)
        if survivor_feasible.size != len(survivors):
            survivor_cv = np.asarray(survivors.get("CV"), dtype=float).reshape(-1)
            survivor_feasible = (
                np.isfinite(survivor_cv) & (survivor_cv <= 0.0)
                if survivor_cv.size == len(survivors)
                else np.zeros(len(survivors), dtype=bool)
            )
        if np.any(survivor_feasible):
            self.ranking.do(
                problem,
                survivors[survivor_feasible],
                n_survive=int(np.count_nonzero(survivor_feasible)),
                random_state=rng,
                **kwargs,
            )

            selected_array = np.asarray(selected, dtype=np.int64)
            survivor_distances = layout_distances[
                np.ix_(selected_array, selected_array)
            ]
            feasible_local = np.where(survivor_feasible)[0]
            survivor_ranks = sorted(
                {
                    int(survivors[int(index)].get("rank"))
                    for index in feasible_local
                }
            )
            for rank in survivor_ranks:
                rank_local = [
                    int(index)
                    for index in feasible_local
                    if int(survivors[int(index)].get("rank")) == rank
                ]
                rank_novelty = self._novelty_scores(
                    survivor_distances,
                    rank_local,
                )
                for local_index in rank_local:
                    novelty_value = rank_novelty.get(local_index, 0.0)
                    individual = survivors[local_index]
                    individual.set("layout_novelty", novelty_value)
                    individual.set(
                        "crowding",
                        self._crowding_plus_novelty(
                            self._crowding_value(individual),
                            novelty_value,
                        ),
                    )

        for index in selected:
            value = minimum_distances.get(index, float("nan"))
            pop[index].set("selection_layout_distance", value)

        algorithm = kwargs.get("algorithm")
        self.history.append(
            {
                "generation": int(getattr(algorithm, "n_gen", 0) or 0),
                "candidates": int(len(pop)),
                "selected": int(len(selected)),
                "physically_feasible_candidates": int(
                    np.count_nonzero(physical_feasible_mask)
                ),
                "acceptable_candidates": int(np.count_nonzero(feasible_mask)),
                "selected_acceptable": int(
                    sum(bool(feasible_mask[index]) for index in selected)
                ),
                "selected_acceptability_violation_mean": float(
                    np.mean([acceptability_violation[index] for index in selected])
                    if selected
                    else 0.0
                ),
                "near_duplicates_deferred": int(len(deferred)),
                "minimum_layout_distance": float(self.minimum_layout_distance),
                "selected_width4_min": int(
                    min((width4_counts[index] for index in selected), default=0)
                ),
                "selected_width4_max": int(
                    max((width4_counts[index] for index in selected), default=0)
                ),
                "selected_width4_mean": float(
                    np.mean([width4_counts[index] for index in selected])
                    if selected
                    else 0.0
                ),
            }
        )
        return survivors


@dataclass(frozen=True)
class DirectLayoutGroup:
    """同じ広場・用途に属し、整列性を相互評価する配置物群。"""

    area: str
    kind: str
    booth_ids: tuple[int, ...]


@dataclass(frozen=True)
class DirectLane:
    """同じ基準線を共有する連続した水平行または垂直列。"""

    axis: str
    booth_ids: tuple[int, ...]
    baseline: float


@dataclass(frozen=True)
class DirectMotif:
    """最終座標から検出された内部列または外周追従チェーン。"""

    kind: str
    booth_ids: tuple[int, ...]
    label: str
    alignment_errors: tuple[tuple[int, float], ...]
    orientation_errors: tuple[tuple[int, float], ...]
    gap_variation: float


@dataclass(frozen=True)
class BoundarySegment:
    """区画外周について配置に依存しない前計算値。"""

    segment_id: int
    start: np.ndarray
    vector: np.ndarray
    vector_norm_sq: float
    length: float
    tangent: np.ndarray
    inward_normal: np.ndarray


# プロセスごとの厳密キャッシュ。整然性の生値は上限に依存しないため、
# キーから世代ごとの上限を外す。段階制約の更新時は保存済み生値から
# ``成分 - 現在上限`` だけを再計算する。
_CONSTRAINT_CACHE: OrderedDict[
    tuple, tuple[float, dict[str, object], dict[str, object]]
] = OrderedDict()
# 交叉は物理制約だけを必要とする。完全な整然性評価を誤って呼ばないため、
# 物理制約専用の小さなキャッシュを分離する。
_PHYSICAL_CONSTRAINT_CACHE: OrderedDict[tuple, float] = OrderedDict()
_BOUNDARY_SEGMENT_CACHE: dict[tuple[Config, str], tuple[BoundarySegment, ...]] = {}
_GROUP_CACHE: dict[tuple[Config, bool, bool], tuple[DirectLayoutGroup, ...]] = {}
_DIRECT_PROBLEM_CACHE: dict[tuple[Config, tuple], "Direct138LayoutProblem"] = {}
_MEMETIC_OPERATOR_CACHE: dict[Config, "DirectMultiScaleMutation"] = {}


def _derived_seed(*values: int) -> int:
    """並列数や実行順に依存しない、タスク単位の決定的seedを返す。"""

    sequence = np.random.SeedSequence([int(value) for value in values])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def direct_layout_groups(
    cfg: Config,
    include_tables: bool = True,
    combine_tables: bool = False,
) -> tuple[DirectLayoutGroup, ...]:
    """整列評価または構成に使う区画別グループを返す。

    ``include_tables=False`` では、来場者と相互作用しないテーブル・椅子を
    整然さの対象から除外する。``combine_tables=True`` では、同一区画の
    通常ブースとテーブル・椅子を1つの混合グループとして返す。
    """

    cache_key = (cfg, bool(include_tables), bool(combine_tables))
    cached = _GROUP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    areas = sorted({str(booth.area) for booth in cfg.booths})
    if combine_tables:
        result = tuple(
            DirectLayoutGroup(
                area,
                "mixed",
                tuple(
                    booth_id
                    for booth_id, booth in enumerate(cfg.booths)
                    if str(booth.area) == area
                ),
            )
            for area in areas
        )
        _GROUP_CACHE[cache_key] = result
        return result
    groups = []
    kinds = ("active", "table") if include_tables else ("active",)
    for kind in kinds:
        for area in areas:
            ids = tuple(
                booth_id
                for booth_id, booth in enumerate(cfg.booths)
                if str(booth.area) == area
                and ("table" if booth.is_table else "active") == kind
            )
            if ids:
                groups.append(DirectLayoutGroup(area, kind, ids))
    result = tuple(groups)
    _GROUP_CACHE[cache_key] = result
    return result


def _lane_cross_coordinate(positions: np.ndarray, booth_id: int, axis: str) -> int:
    return int(positions[booth_id, 1 if axis == "H" else 0])


def _lane_along_coordinate(positions: np.ndarray, booth_id: int, axis: str) -> int:
    return int(positions[booth_id, 0 if axis == "H" else 1])


def _edge_gap(
    cfg: Config,
    positions: np.ndarray,
    first: int,
    second: int,
    axis: str,
) -> float:
    x_first, y_first, theta_first = (int(v) for v in positions[first])
    x_second, y_second, _ = (int(v) for v in positions[second])
    width, height = oriented_booth_size(cfg, first, theta_first)
    first_end = x_first + width if axis == "H" else y_first + height
    second_start = x_second if axis == "H" else y_second
    return float(max(0, second_start - first_end))


def _claimed_cells_for_position(
    cfg: Config,
    booth_id: int,
    x_pos: int,
    y_pos: int,
    theta: int,
) -> set[tuple[int, int]]:
    """本体・受付・退出・待機列が実際に使用するセルを返す。"""

    claimed = set(booth_cells_of(cfg, booth_id, x_pos, y_pos, theta))
    for cell in (
        reception_cell(cfg, booth_id, x_pos, y_pos, theta),
        booth_exit_cell(cfg, booth_id, x_pos, y_pos, theta),
    ):
        if cell is not None:
            claimed.add(cell)
    claimed.update(queue_cells_of(cfg, booth_id, x_pos, y_pos, theta))
    return claimed


def _front_clearance_violation(
    cfg: Config,
    positions: np.ndarray,
    venue=None,
    geometry: PlacementGeometry | None = None,
) -> float:
    """全ブース正面通路セルのうち、他配置物等に塞がれた割合を返す。"""

    effective_venue = venue if venue is not None else build_base_venue(cfg)
    if geometry is None:
        claimed = [
            _claimed_cells_for_position(
                cfg,
                booth_id,
                *(int(value) for value in positions[booth_id]),
            )
            for booth_id in range(len(cfg.booths))
        ]
    else:
        claimed = [set(cells) for cells in geometry.claimed_cells]
    claimed_counts: dict[tuple[int, int], int] = {}
    for cells in claimed:
        for cell in cells:
            claimed_counts[cell] = claimed_counts.get(cell, 0) + 1
    blocked = 0
    total = 0
    height, width = cfg.grid_height, cfg.grid_width
    walkable_types = set(int(value) for value in WALKABLE_TYPES)
    for booth_id, spec in enumerate(cfg.booths):
        if spec.is_table:
            continue
        x_pos, y_pos, theta = (int(value) for value in positions[booth_id])
        clear_cells = (
            geometry.front_clearance_cells[booth_id]
            if geometry is not None
            else front_clearance_cells(cfg, booth_id, x_pos, y_pos, theta)
        )
        for cell_x, cell_y in clear_cells:
            total += 1
            if (
                not (0 <= cell_x < width and 0 <= cell_y < height)
                or int(effective_venue.base_grid[cell_y, cell_x])
                not in walkable_types
                or claimed_counts.get((cell_x, cell_y), 0)
                > int((cell_x, cell_y) in claimed[booth_id])
            ):
                blocked += 1
    return float(blocked / total) if total else 0.0


def _count_path_turns(field: np.ndarray, cell, grid_width: int) -> int | None:
    """距離場を降下する最短経路上の方向転換回数を返す。到達不能なら None。"""

    x_pos, y_pos = int(cell[0]), int(cell[1])
    if field[y_pos, x_pos] >= INF_DIST:
        return None
    flat = descend_path_indices(field, (x_pos, y_pos))
    turns = 0
    previous = None
    prev_x, prev_y = None, None
    for index in flat:
        cur_x = int(index % grid_width)
        cur_y = int(index // grid_width)
        if prev_x is not None:
            direction = (cur_x - prev_x, cur_y - prev_y)
            if previous is not None and direction != previous:
                turns += 1
            previous = direction
        prev_x, prev_y = cur_x, cur_y
    return turns


def _path_turn_violation(
    cfg: Config,
    positions: np.ndarray,
    venue=None,
    geometry: PlacementGeometry | None = None,
    layout=None,
) -> float:
    """体験ブース間の最短経路上の、平均方向転換回数を返す。

    来場者は 1 人あたり平均 2 ブース程度を回るため、動線の分かりやすさは
    「入口から各ブースまで」よりも「ブースからブースまで」で決まる。
    実会場は境界の開口部が 98 セルあり入口が事実上どこにでもあるので、
    入口起点で測ると配置による差がほとんど出ない(実測 1.3〜2.4 回)。
    ブース間で測ると手配置 3.92 回に対し最適化解 6.5〜8.2 回と差が出る。

    値は平均回数そのもの(0 以上の実数)で、OrderlinessLimits.path_turns で
    正規化される。テーブルとゴミステーションは付帯設備なので対象外。
    """

    effective_venue = venue if venue is not None else build_base_venue(cfg)
    geom = geometry or build_placement_geometry(cfg, positions)
    effective_layout = (
        layout
        if layout is not None
        else apply_layout(cfg, effective_venue, positions)
    )
    grid_width = int(cfg.grid_width)

    targets: list[int] = []
    sources: dict[int, tuple] = {}
    for booth_id, spec in enumerate(cfg.booths):
        if PATH_TURN_EXCLUDE_FACILITIES and (spec.is_table or spec.is_waste_station):
            continue
        cell = geom.receptions[booth_id]
        if cell is None:
            continue
        targets.append(booth_id)
        sources[booth_id] = (cell,)
    if len(targets) < 2:
        return 0.0

    total = 0
    count = 0
    for origin in targets:
        field = bfs_distance_field(effective_layout.walkable, sources[origin])
        for destination in targets:
            if destination <= origin:
                continue
            turns = _count_path_turns(
                field, sources[destination][0], grid_width
            )
            if turns is None:
                continue
            total += turns
            count += 1
    return float(total / count) if count else 0.0


def detect_direct_lanes(
    cfg: Config,
    positions: np.ndarray,
    group: DirectLayoutGroup,
    tolerance: float = LANE_ALIGNMENT_TOLERANCE_CELLS,
    max_edge_gap: float = MAX_LANE_EDGE_GAP_CELLS,
) -> tuple[DirectLane, ...]:
    """共有基準線と連続性を持つ2個以上の行・列候補を抽出する。"""

    lanes: dict[tuple[str, tuple[int, ...]], DirectLane] = {}
    for axis in ("H", "V"):
        seeds = sorted(
            {
                _lane_cross_coordinate(positions, booth_id, axis)
                for booth_id in group.booth_ids
            }
        )
        for seed in seeds:
            nearby = [
                booth_id
                for booth_id in group.booth_ids
                if abs(_lane_cross_coordinate(positions, booth_id, axis) - seed)
                <= tolerance
            ]
            if len(nearby) < 2:
                continue
            baseline = float(
                np.median(
                    [
                        _lane_cross_coordinate(positions, booth_id, axis)
                        for booth_id in nearby
                    ]
                )
            )
            nearby = [
                booth_id
                for booth_id in nearby
                if abs(
                    _lane_cross_coordinate(positions, booth_id, axis) - baseline
                )
                <= tolerance
            ]
            nearby.sort(
                key=lambda booth_id: _lane_along_coordinate(
                    positions, booth_id, axis
                )
            )
            segment: list[int] = []

            def register(current: list[int]) -> None:
                if len(current) < 2:
                    return
                members = tuple(int(v) for v in current)
                lane_baseline = float(
                    np.median(
                        [
                            _lane_cross_coordinate(positions, booth_id, axis)
                            for booth_id in members
                        ]
                    )
                )
                lanes[(axis, members)] = DirectLane(axis, members, lane_baseline)

            for booth_id in nearby:
                if segment and _edge_gap(
                    cfg, positions, segment[-1], booth_id, axis
                ) > max_edge_gap:
                    register(segment)
                    segment = []
                segment.append(int(booth_id))
            register(segment)
    return tuple(lanes.values())


def _gap_variation(gaps: np.ndarray) -> float:
    """外周間隔列のMAD比。2間隔未満は変動を定義できないため0とする。"""

    if len(gaps) < 2:
        return 0.0
    median = float(np.median(gaps))
    mad = float(np.median(np.abs(gaps - median)))
    return min(1.0, mad / max(1.0, median + 1.0))


def _front_direction(theta: int) -> np.ndarray:
    return (
        np.asarray((0.0, -1.0))
        if int(theta) == 0
        else np.asarray((1.0, 0.0))
        if int(theta) == 1
        else np.asarray((0.0, 1.0))
        if int(theta) == 2
        else np.asarray((-1.0, 0.0))
    )


def _booth_centers_and_sizes(
    cfg: Config, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.zeros((len(cfg.booths), 2), dtype=float)
    sizes = np.zeros((len(cfg.booths), 2), dtype=float)
    for booth_id, (x_pos, y_pos, theta) in enumerate(positions):
        width, height = oriented_booth_size(cfg, booth_id, int(theta))
        sizes[booth_id] = (width, height)
        centers[booth_id] = (float(x_pos) + width / 2.0, float(y_pos) + height / 2.0)
    return centers, sizes


def _lane_motifs(
    cfg: Config,
    positions: np.ndarray,
    group: DirectLayoutGroup,
    lanes: tuple[DirectLane, ...] | None = None,
) -> tuple[DirectMotif, ...]:
    motifs = []
    effective_lanes = lanes if lanes is not None else detect_direct_lanes(
        cfg, positions, group
    )
    for lane in effective_lanes:
        if len(lane.booth_ids) < MIN_STRONG_LANE_SIZE:
            continue
        alignment = tuple(
            (
                int(booth_id),
                min(
                    1.0,
                    abs(
                        _lane_cross_coordinate(positions, booth_id, lane.axis)
                        - lane.baseline
                    )
                    / LANE_ALIGNMENT_TOLERANCE_CELLS,
                ),
            )
            for booth_id in lane.booth_ids
        )
        desired_axis = 0 if lane.axis == "H" else 1
        orientation = tuple(
            (
                int(booth_id),
                float(int(positions[booth_id, 2]) % 2 != desired_axis),
            )
            for booth_id in lane.booth_ids
            if not cfg.booths[booth_id].is_table
        )
        ordered = sorted(
            lane.booth_ids,
            key=lambda booth_id: _lane_along_coordinate(
                positions, booth_id, lane.axis
            ),
        )
        gaps = np.asarray(
            [
                _edge_gap(cfg, positions, first, second, lane.axis)
                for first, second in zip(ordered, ordered[1:])
            ],
            dtype=float,
        )
        motifs.append(
            DirectMotif(
                kind="lane",
                booth_ids=tuple(int(value) for value in lane.booth_ids),
                label=f"{lane.axis}@{lane.baseline:g}",
                alignment_errors=alignment,
                orientation_errors=orientation,
                gap_variation=_gap_variation(gaps),
            )
        )
    return tuple(motifs)


def _boundary_segments(cfg: Config, area: str) -> tuple[BoundarySegment, ...]:
    """区画外周の接線・内向き法線をプロセス内で再利用する。"""

    cache_key = (cfg, str(area))
    cached = _BOUNDARY_SEGMENT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    polygon_source = next(
        (polygon for name, polygon in cfg.venue_areas if str(name) == str(area)),
        None,
    )
    if polygon_source is None:
        _BOUNDARY_SEGMENT_CACHE[cache_key] = ()
        return ()
    polygon = np.asarray(polygon_source, dtype=float) / float(cfg.cell_size_meters)
    centroid = np.mean(polygon, axis=0)
    segments = []
    for segment_id, (start_raw, end_raw) in enumerate(
        zip(polygon, np.roll(polygon, -1, axis=0))
    ):
        start = np.asarray(start_raw, dtype=float)
        vector = np.asarray(end_raw - start_raw, dtype=float)
        vector_norm_sq = float(np.dot(vector, vector))
        length = float(np.sqrt(vector_norm_sq))
        if length <= 1.0e-9:
            continue
        tangent = vector / length
        normal = np.asarray((-tangent[1], tangent[0]), dtype=float)
        midpoint = start + vector / 2.0
        if float(np.dot(centroid - midpoint, normal)) < 0.0:
            normal = -normal
        segments.append(
            BoundarySegment(
                segment_id=int(segment_id),
                start=start,
                vector=vector,
                vector_norm_sq=vector_norm_sq,
                length=length,
                tangent=tangent,
                inward_normal=normal,
            )
        )
    result = tuple(segments)
    _BOUNDARY_SEGMENT_CACHE[cache_key] = result
    return result


def _boundary_motifs(
    cfg: Config,
    positions: np.ndarray,
    group: DirectLayoutGroup,
    centers: np.ndarray | None = None,
    sizes: np.ndarray | None = None,
) -> tuple[DirectMotif, ...]:
    """多角形外周の各辺に沿う連続チェーンを最終座標から検出する。"""

    segments = _boundary_segments(cfg, group.area)
    if not segments:
        return ()
    if centers is None or sizes is None:
        centers, sizes = _booth_centers_and_sizes(cfg, positions)
    motifs: list[DirectMotif] = []
    for segment in segments:
        start = segment.start
        vector = segment.vector
        tangent = segment.tangent
        normal = segment.inward_normal
        eligible = []
        for booth_id in group.booth_ids:
            center = centers[booth_id]
            relative = center - start
            parameter = float(np.dot(relative, vector) / segment.vector_norm_sq)
            projection_parameter = float(np.clip(parameter, 0.0, 1.0))
            projection = start + projection_parameter * vector
            distance = float(np.linalg.norm(center - projection))
            if not (-0.05 <= parameter <= 1.05) or distance > BOUNDARY_BAND_CELLS:
                continue
            half_extent = float(
                (
                    abs(tangent[0]) * sizes[booth_id, 0]
                    + abs(tangent[1]) * sizes[booth_id, 1]
                )
                / 2.0
            )
            along = float(np.dot(center - start, tangent))
            eligible.append(
                (along, int(booth_id), distance, half_extent)
            )
        eligible.sort(key=lambda value: value[0])
        chains: list[list[tuple[float, int, float, float]]] = []
        chain = []
        for item in eligible:
            if chain:
                edge_gap = max(
                    0.0,
                    item[0] - item[3] - chain[-1][0] - chain[-1][3],
                )
                if edge_gap > MAX_LANE_EDGE_GAP_CELLS:
                    if len(chain) >= MIN_BOUNDARY_CHAIN_SIZE:
                        chains.append(chain)
                    chain = []
            chain.append(item)
        if len(chain) >= MIN_BOUNDARY_CHAIN_SIZE:
            chains.append(chain)
        for chain in chains:
            distances = np.asarray([item[2] for item in chain], dtype=float)
            median_distance = float(np.median(distances))
            alignment = tuple(
                (
                    item[1],
                    min(
                        1.0,
                        abs(item[2] - median_distance)
                        / BOUNDARY_ALIGNMENT_TOLERANCE_CELLS,
                    ),
                )
                for item in chain
            )
            orientation = tuple(
                (
                    item[1],
                    float(
                        np.dot(
                            _front_direction(int(positions[item[1], 2])), normal
                        )
                        < 0.35
                    ),
                )
                for item in chain
                if not cfg.booths[item[1]].is_table
            )
            gaps = np.asarray(
                [
                    max(
                        0.0,
                        second[0] - second[3] - first[0] - first[3],
                    )
                    for first, second in zip(chain, chain[1:])
                ],
                dtype=float,
            )
            motifs.append(
                DirectMotif(
                    kind="boundary",
                    booth_ids=tuple(item[1] for item in chain),
                    label=f"edge:{segment.segment_id}",
                    alignment_errors=alignment,
                    orientation_errors=orientation,
                    gap_variation=_gap_variation(gaps),
                )
            )
    return tuple(motifs)


def detect_direct_motifs(
    cfg: Config,
    positions: np.ndarray,
    group: DirectLayoutGroup,
    lanes: tuple[DirectLane, ...] | None = None,
    centers: np.ndarray | None = None,
    sizes: np.ndarray | None = None,
) -> tuple[DirectMotif, ...]:
    """内部列と外周追従チェーンを、配置後の座標だけから検出する。"""

    candidates = [
        *_lane_motifs(cfg, positions, group, lanes=lanes),
        *_boundary_motifs(
            cfg, positions, group, centers=centers, sizes=sizes
        ),
    ]
    unique: dict[tuple[str, tuple[int, ...]], DirectMotif] = {}
    for motif in candidates:
        key = (motif.kind, tuple(sorted(motif.booth_ids)))
        current = unique.get(key)
        current_error = (
            float(np.mean([value for _, value in current.alignment_errors]))
            if current is not None
            else float("inf")
        )
        candidate_error = float(
            np.mean([value for _, value in motif.alignment_errors])
        )
        if current is None or candidate_error < current_error:
            unique[key] = motif
    return tuple(unique.values())


def _select_covering_motifs(
    booth_ids: tuple[int, ...], motifs: tuple[DirectMotif, ...]
) -> tuple[DirectMotif, ...]:
    """少ないモチーフで多くを説明する決定的な貪欲集合被覆。"""

    uncovered = set(int(value) for value in booth_ids)
    selected: list[DirectMotif] = []
    while True:
        options = []
        for index, motif in enumerate(motifs):
            new_members = uncovered.intersection(motif.booth_ids)
            if len(new_members) < 2:
                continue
            mean_error = float(
                np.mean([value for _, value in motif.alignment_errors])
            )
            options.append(
                (
                    len(new_members),
                    len(motif.booth_ids),
                    -mean_error,
                    -index,
                    motif,
                )
            )
        if not options:
            break
        motif = max(options, key=lambda value: value[:-1])[-1]
        selected.append(motif)
        uncovered.difference_update(motif.booth_ids)
    return tuple(selected)


def _group_orderliness(
    cfg: Config,
    positions: np.ndarray,
    group: DirectLayoutGroup,
    centers: np.ndarray | None = None,
    sizes: np.ndarray | None = None,
) -> dict[str, object]:
    lanes = detect_direct_lanes(cfg, positions, group)
    motifs = detect_direct_motifs(
        cfg,
        positions,
        group,
        lanes=lanes,
        centers=centers,
        sizes=sizes,
    )
    selected = _select_covering_motifs(group.booth_ids, motifs)
    supported = {
        booth_id for motif in selected for booth_id in motif.booth_ids
    }
    strong_supported = {
        booth_id
        for motif in selected
        if len(motif.booth_ids) >= MIN_STRONG_MOTIF_SIZE
        for booth_id in motif.booth_ids
    }
    # 孤立判定では、強い列に届かない2個組も「近傍あり」とみなす。
    weak_lane_ids = {
        booth_id
        for lane in lanes
        for booth_id in lane.booth_ids
    }
    paired = weak_lane_ids.union(
        booth_id for motif in motifs for booth_id in motif.booth_ids
    )
    alignment_maps = [dict(motif.alignment_errors) for motif in selected]
    orientation_maps = [dict(motif.orientation_errors) for motif in selected]
    alignment_errors = []
    orientation_errors = []
    for booth_id in group.booth_ids:
        member_indices = [
            index
            for index, motif in enumerate(selected)
            if booth_id in motif.booth_ids
        ]
        alignment_errors.append(
            min(
                (
                    alignment_maps[index].get(booth_id, 1.0)
                    for index in member_indices
                ),
                default=1.0,
            )
        )
        if not cfg.booths[booth_id].is_table:
            orientation_errors.append(
                min(
                    (
                        orientation_maps[index].get(booth_id, 1.0)
                        for index in member_indices
                    ),
                    default=1.0,
                )
            )
    n_members = len(group.booth_ids)
    target_motifs = max(1, int(np.ceil(n_members / MOTIF_TARGET_SIZE)))
    return {
        "area": group.area,
        "kind": group.kind,
        "n_members": int(n_members),
        "n_motifs": int(len(selected)),
        "n_lane_motifs": int(sum(motif.kind == "lane" for motif in selected)),
        "n_boundary_motifs": int(
            sum(motif.kind == "boundary" for motif in selected)
        ),
        "organization_nonmembership": float(1.0 - len(supported) / n_members),
        "motif_complexity": float(len(selected) / target_motifs),
        "strong_structure_nonmembership": float(
            1.0 - len(strong_supported) / n_members
        ),
        "alignment_error": float(np.mean(alignment_errors)),
        "gap_variation": (
            float(np.mean([motif.gap_variation for motif in selected]))
            if selected
            else 1.0
        ),
        "orientation": (
            float(np.mean(orientation_errors)) if orientation_errors else 0.0
        ),
        "isolation": float(1.0 - len(paired) / n_members),
        "selected_motifs": [
            {
                "kind": motif.kind,
                "label": motif.label,
                "booth_ids": list(motif.booth_ids),
                "gap_variation": float(motif.gap_variation),
            }
            for motif in selected
        ],
    }


def _wide_aisle_deficit(
    cfg: Config,
    positions: np.ndarray,
    venue=None,
    geometry: PlacementGeometry | None = None,
    width_profile: dict[str, object] | None = None,
    layout=None,
) -> float:
    """主要ブースのうち3セル幅で往復到達できない割合を返す。"""

    active_ids = [
        booth_id
        for booth_id, booth in enumerate(cfg.booths)
        if not booth.is_table and not booth.is_waste_station
    ]
    if not active_ids:
        return 0.0

    # 将来、幅プロファイルが3セルまで持つ設定になった場合は再計算を省く。
    profile_widths = (
        set((width_profile or {}).get("achieved_counts", {}).keys())
        if width_profile is not None
        else set()
    )
    booth_widths = tuple(
        int(value) for value in (width_profile or {}).get("booth_widths", ())
    )
    if (
        WIDE_AISLE_WIDTH_CELLS in profile_widths
        and len(booth_widths) == len(cfg.booths)
    ):
        return float(
            np.mean(
                [
                    booth_widths[booth_id] < WIDE_AISLE_WIDTH_CELLS
                    for booth_id in active_ids
                ]
            )
        )

    effective_venue = venue if venue is not None else build_base_venue(cfg)
    geom = geometry or build_placement_geometry(cfg, positions)
    effective_layout = (
        layout
        if layout is not None
        else apply_layout(cfg, effective_venue, positions)
    )
    wide_walkable = passable_corridor_mask(
        effective_layout.raw_walkable,
        WIDE_AISLE_WIDTH_CELLS,
    )
    entrance_field = bfs_distance_field(
        wide_walkable,
        effective_venue.entrance_cells,
    )
    exit_field = bfs_distance_field(
        wide_walkable,
        effective_venue.exit_cells,
    )
    reachable = 0
    for booth_id in active_ids:
        reception = geom.receptions[booth_id]
        booth_exit = geom.booth_exits[booth_id]
        if reception is None or booth_exit is None:
            continue
        reception_x, reception_y = reception
        exit_x, exit_y = booth_exit
        if (
            entrance_field[reception_y, reception_x] < INF_DIST
            and exit_field[exit_y, exit_x] < INF_DIST
        ):
            reachable += 1
    return float(1.0 - reachable / len(active_ids))


def compute_direct_orderliness(
    cfg: Config,
    x,
    limits: OrderlinessLimits | None = None,
    venue=None,
    geometry: PlacementGeometry | None = None,
    width_profile: dict[str, object] | None = None,
    include_path_turns: bool = True,
    include_wide_aisle: bool = True,
) -> dict[str, object]:
    """列と外周追従を同格に扱う、最終表現上の受容可能性制約を計算する。"""

    effective_limits = limits or OrderlinessLimits()
    canonical = canonicalize_table_design(cfg, np.asarray(x))
    positions = np.asarray(to_integer_layout(canonical, cfg), dtype=np.int64)
    effective_venue = venue if venue is not None else build_base_venue(cfg)
    shared_layout = (
        apply_layout(cfg, effective_venue, positions)
        if include_path_turns or include_wide_aisle
        else None
    )
    centers, sizes = _booth_centers_and_sizes(cfg, positions)
    mixed_groups = [
        _group_orderliness(
            cfg, positions, group, centers=centers, sizes=sizes
        )
        for group in direct_layout_groups(cfg, combine_tables=True)
        if len(group.booth_ids) >= 2
    ]
    separate_groups = [
        _group_orderliness(
            cfg, positions, group, centers=centers, sizes=sizes
        )
        for group in direct_layout_groups(cfg, include_tables=True)
        if len(group.booth_ids) >= 2
    ]
    groups = [*mixed_groups, *separate_groups]
    components = {
        name: max(float(group[name]) for group in groups) if groups else 0.0
        for name in (
            "organization_nonmembership",
            "motif_complexity",
            "strong_structure_nonmembership",
            "alignment_error",
            "gap_variation",
            "orientation",
            "isolation",
        )
    }
    components["front_clearance"] = _front_clearance_violation(
        cfg, positions, venue=effective_venue, geometry=geometry
    )
    components["wide_aisle_deficit"] = (
        _wide_aisle_deficit(
            cfg,
            positions,
            venue=effective_venue,
            geometry=geometry,
            width_profile=width_profile,
            layout=shared_layout,
        )
        if include_wide_aisle
        else 0.0
    )
    # path_turns は本問題の受容可能性制約に含める。ただし整然配置ドナーの
    # 構成中だけは、列・外周モチーフを先に獲得できるよう計算から外せる。
    # ドナーを母集団へ導入した後は、他の個体と同様に必ず方向転換制約を評価する。
    components["path_turns"] = (
        _path_turn_violation(
            cfg,
            positions,
            venue=effective_venue,
            geometry=geometry,
            layout=shared_layout,
        )
        if include_path_turns
        else 0.0
    )
    return _materialize_orderliness(
        {**components, "groups": groups},
        effective_limits,
    )


def _raw_orderliness(orderliness: dict[str, object]) -> dict[str, object]:
    """上限に依存しない整然性成分だけを抽出する。"""

    return {
        **{
            name: float(orderliness[name])
            for name in ORDERLINESS_COMPONENT_KEYS
        },
        "groups": orderliness.get("groups", []),
    }


def _materialize_orderliness(
    raw: dict[str, object],
    limits: OrderlinessLimits,
) -> dict[str, object]:
    """保存済み生値へ現在の上限を適用し、従来形式の辞書を返す。"""

    components = {
        name: float(raw[name])
        for name in ORDERLINESS_COMPONENT_KEYS
    }
    limit_values = limits.as_dict()
    normalized = {
        name: (
            value / limit_values[name]
            if limit_values[name] > 0.0
            else (0.0 if value <= 0.0 else float("inf"))
        )
        for name, value in components.items()
    }
    return {
        "total": float(
            max(
                (normalized[name] for name in ORDERLINESS_CONSTRAINT_KEYS),
                default=0.0,
            )
        ),
        **components,
        "limits": limit_values,
        "groups": raw.get("groups", []),
    }


def _orderliness_constraint_vector(
    orderliness: dict[str, object],
    limits: OrderlinessLimits,
) -> np.ndarray:
    """整然性成分を、比較可能な受容可能性制約違反量へ変換する。

    各成分は ``値 / 上限 - 1 <= 0`` となるよう正規化する。上限0の成分は
    0以下だけを許す。方向転換回数だけは、制約未達個体同士の選択で強く
    改善されるよう違反量を3倍する。物理制約とは別ベクトルのため、整然性
    違反を減らすことで物理違反が相殺されることはない。
    """

    limit_values = limits.as_dict()
    constraints = []
    for name in ORDERLINESS_CONSTRAINT_KEYS:
        value = float(orderliness[name])
        limit = float(limit_values[name])
        violation = (
            value / limit - 1.0
            if limit > 0.0
            else (0.0 if value <= 0.0 else value)
        )
        weight = PATH_TURN_CONSTRAINT_WEIGHT if name == "path_turns" else 1.0
        constraints.append(float(weight * violation))
    return np.asarray(constraints, dtype=float)


class Direct138LayoutProblem(ElementwiseProblem):
    """物理制約を必須、整然性を受容可能性制約とする直接配置問題。"""

    def __init__(
        self,
        cfg: Config,
        orderliness_limits: OrderlinessLimits | None = None,
        max_orderliness: float | None = None,
        **kwargs,
    ) -> None:
        self.cfg = cfg
        self.orderliness_limits = orderliness_limits or OrderlinessLimits()
        # 旧CLI/API互換。0.25を従来既定値とし、全成分上限を一様に拡縮する。
        if max_orderliness is not None:
            if not 0.0 <= max_orderliness <= 1.0:
                raise ValueError("max_orderliness must be in [0, 1]")
            self.orderliness_limits = self.orderliness_limits.scaled(
                float(max_orderliness) / 0.25
            )
        self.final_orderliness_limits = self.orderliness_limits
        self.venue = build_base_venue(cfg)
        xl, xu = variable_bounds(cfg)
        super().__init__(
            n_var=design_variable_count(cfg),
            n_obj=2,
            n_ieq_constr=1 + len(ORDERLINESS_CONSTRAINT_KEYS),
            xl=xl,
            xu=xu,
            vtype=int,
            **kwargs,
        )

    def set_orderliness_limits(self, limits: OrderlinessLimits) -> None:
        """受容可能性制約の固定上限を更新する。"""
        self.orderliness_limits = limits

    def _positions_and_cache_key(self, x) -> tuple[tuple, tuple]:
        canonical = canonicalize_table_design(self.cfg, np.asarray(x))
        positions = to_integer_layout(canonical, self.cfg)
        design_key = tuple(value for triple in positions for value in triple)
        return positions, (self.cfg, design_key)

    @staticmethod
    def _trim_cache(cache: OrderedDict) -> None:
        while len(cache) > CONSTRAINT_CACHE_MAXSIZE:
            cache.popitem(last=False)

    def physical_violation(self, x) -> float:
        """交叉用に、整然性を計算せず物理制約だけを返す。"""

        positions, cache_key = self._positions_and_cache_key(x)
        complete = _CONSTRAINT_CACHE.get(cache_key)
        if complete is not None:
            _CONSTRAINT_CACHE.move_to_end(cache_key)
            return float(complete[0])

        cached = _PHYSICAL_CONSTRAINT_CACHE.get(cache_key)
        if cached is not None:
            _PHYSICAL_CONSTRAINT_CACHE.move_to_end(cache_key)
            return float(cached)

        geometry = build_placement_geometry(self.cfg, positions)
        physical = float(
            compute_violation(
                self.cfg,
                self.venue,
                positions,
                geometry=geometry,
            )
        )
        _PHYSICAL_CONSTRAINT_CACHE[cache_key] = physical
        _PHYSICAL_CONSTRAINT_CACHE.move_to_end(cache_key)
        self._trim_cache(_PHYSICAL_CONSTRAINT_CACHE)
        return physical

    def constraint_components(self, x):
        positions, cache_key = self._positions_and_cache_key(x)
        cached = _CONSTRAINT_CACHE.get(cache_key)
        if cached is not None:
            _CONSTRAINT_CACHE.move_to_end(cache_key)
            physical, width_profile, raw_orderliness = cached
            orderliness = _materialize_orderliness(
                raw_orderliness,
                self.orderliness_limits,
            )
            return (
                float(physical),
                width_profile,
                _orderliness_constraint_vector(
                    orderliness,
                    self.orderliness_limits,
                ),
                orderliness,
            )

        geometry = build_placement_geometry(self.cfg, positions)
        physical_cached = _PHYSICAL_CONSTRAINT_CACHE.get(cache_key)
        if physical_cached is None:
            physical = float(
                compute_violation(
                    self.cfg,
                    self.venue,
                    positions,
                    geometry=geometry,
                )
            )
        else:
            _PHYSICAL_CONSTRAINT_CACHE.move_to_end(cache_key)
            physical = float(physical_cached)
        width_profile = (
            compute_width_reachability_profile(
                self.cfg,
                self.venue,
                positions,
                geometry=geometry,
            )
            if physical <= 0.0
            else {
                # 幅のキーは TARGET_CORRIDOR_WIDTH_CELLS から生成する。
                # 4 を直接書くと、目標幅を変えたときに KeyError になる。
                "target_width": int(TARGET_CORRIDOR_WIDTH_CELLS),
                "required_booth_count": len(self.cfg.booths),
                "booth_widths": tuple(0 for _ in self.cfg.booths),
                "achieved_counts": {
                    width: 0
                    for width in range(TARGET_CORRIDOR_WIDTH_CELLS, 0, -1)
                },
                "exact_counts": {
                    width: 0
                    for width in range(TARGET_CORRIDOR_WIDTH_CELLS, 0, -1)
                },
                "four_cell_achievement_count": 0,
                "all_booths_one_cell_reachable": False,
            }
        )
        orderliness = compute_direct_orderliness(
            self.cfg,
            positions_to_design(self.cfg, positions),
            limits=self.orderliness_limits,
            venue=self.venue,
            geometry=geometry,
            width_profile=width_profile,
        )
        raw_orderliness = _raw_orderliness(orderliness)
        _CONSTRAINT_CACHE[cache_key] = (
            physical,
            width_profile,
            raw_orderliness,
        )
        _CONSTRAINT_CACHE.move_to_end(cache_key)
        self._trim_cache(_CONSTRAINT_CACHE)
        return (
            physical,
            width_profile,
            _orderliness_constraint_vector(
                orderliness,
                self.orderliness_limits,
            ),
            orderliness,
        )

    def is_orderly(self, x) -> bool:
        """全受容可能性制約を満たすかを返す。"""

        _, _, order_constraints, _ = self.constraint_components(x)
        return bool(np.all(np.asarray(order_constraints) <= 0.0))

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        physical, width_profile, order_constraints, orderliness = (
            self.constraint_components(x)
        )
        all_constraints = [float(physical), *np.asarray(order_constraints).tolist()]
        out["width4_achievement_count"] = float(
            width_profile["four_cell_achievement_count"]
        )
        out["width4_required_count"] = float(width_profile["required_booth_count"])
        out["booth_access_widths"] = width_profile["booth_widths"]
        # 整然性違反個体も目的関数を評価して探索候補に残す。シミュレーションを
        # 省略するのは、必須である物理制約に違反した場合だけとする。
        if physical > 0.0:
            out["F"] = [PENALTY_OBJECTIVE, PENALTY_OBJECTIVE]
            out["G"] = all_constraints
            out["kpis"] = {}
            out["orderliness"] = orderliness
            return

        positions = to_integer_layout(
            canonicalize_table_design(self.cfg, np.asarray(x)),
            self.cfg,
        )
        simulation = evaluate_layout(
            get_context(self.cfg),
            positions,
            collect_time_series=False,
        )
        objectives = np.asarray(
            [simulation.f_move, simulation.f_congestion],
            dtype=float,
        )
        if simulation.timed_out or not np.all(np.isfinite(objectives)):
            out["F"] = [PENALTY_OBJECTIVE, PENALTY_OBJECTIVE]
        else:
            out["F"] = objectives.tolist()
        out["G"] = all_constraints
        out["kpis"] = simulation.kpis
        out["orderliness"] = orderliness


def _cached_direct_problem(
    cfg: Config,
    limits: OrderlinessLimits,
) -> Direct138LayoutProblem:
    """初期化・近傍生成ワーカー内で問題オブジェクトを再利用する。"""

    key = (cfg, tuple(limits.as_dict().items()))
    problem = _DIRECT_PROBLEM_CACHE.get(key)
    if problem is None:
        problem = Direct138LayoutProblem(cfg, orderliness_limits=limits)
        _DIRECT_PROBLEM_CACHE[key] = problem
    return problem


def _random_lane_partition(booth_ids: tuple[int, ...], rng) -> list[list[int]]:
    """1モチーフ3～5個を基本とするランダム分割を返す。"""

    n_members = len(booth_ids)
    min_lanes = int(np.ceil(n_members / 3))
    # 見た目の点在を防ぐため、1モチーフ当たり最大5個を基準にする。
    # モチーフは後段で内部列または外周追従として配置される。
    n_lanes = min_lanes
    sizes = [MIN_STRONG_LANE_SIZE] * n_lanes
    remaining = n_members - sum(sizes)
    while remaining > 0:
        lane_id = int(rng.integers(0, n_lanes))
        if sizes[lane_id] < 3:
            sizes[lane_id] += 1
            remaining -= 1
    shuffled = [int(v) for v in rng.permutation(booth_ids)]
    result = []
    start = 0
    for size in sizes:
        result.append(shuffled[start : start + size])
        start += size
    return result


def _try_place_boundary_chain(
    cfg: Config,
    venue,
    group: DirectLayoutGroup,
    members: list[int],
    occupied: np.ndarray,
    front_reserved: np.ndarray,
    rng,
    trials: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]] | None:
    """会場形状だけを使い、指定配置物を外周の1辺へ等間隔配置する。"""

    polygon_source = next(
        (polygon for area, polygon in cfg.venue_areas if str(area) == group.area),
        None,
    )
    if polygon_source is None:
        return None
    polygon = np.asarray(polygon_source, dtype=float) / float(cfg.cell_size_meters)
    centroid = np.mean(polygon, axis=0)
    segments = []
    for segment_id, (start, end) in enumerate(
        zip(polygon, np.roll(polygon, -1, axis=0))
    ):
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length <= 1.0e-9:
            continue
        tangent = vector / length
        normal = np.asarray((-tangent[1], tangent[0]), dtype=float)
        if float(np.dot(centroid - (start + end) / 2.0, normal)) < 0.0:
            normal = -normal
        segments.append((segment_id, start, length, tangent, normal))
    if not segments:
        return None
    height, width = cfg.grid_height, cfg.grid_width
    walkable_types = set(int(value) for value in WALKABLE_TYPES)
    for _ in range(max(1, int(trials))):
        _, start, length, tangent, normal = segments[
            int(rng.integers(0, len(segments)))
        ]
        theta_candidates = [
            theta
            for theta in range(4)
            if float(np.dot(_front_direction(theta), normal)) >= 0.35
        ]
        if not theta_candidates:
            continue
        thetas = [
            (
                0
                if cfg.booths[booth_id].is_table
                else int(rng.choice(theta_candidates))
            )
            for booth_id in members
        ]
        half_extents = []
        for booth_id, theta in zip(members, thetas):
            booth_width, booth_height = oriented_booth_size(cfg, booth_id, theta)
            half_extents.append(
                float(
                    (
                        abs(tangent[0]) * booth_width
                        + abs(tangent[1]) * booth_height
                    )
                    / 2.0
                )
            )
        base_gap = int(rng.integers(2, 6))
        total_extent = float(2.0 * sum(half_extents) + base_gap * (len(members) - 1))
        if total_extent > length - 1.0:
            continue
        offset = float(rng.uniform(1.5, 4.0))
        cursor = float(rng.uniform(0.5, max(0.500001, length - total_extent)))
        temporary = occupied.copy()
        temporary_front = front_reserved.copy()
        states: list[tuple[int, int, int, int]] = []
        valid = True
        for member_index, (booth_id, theta, half_extent) in enumerate(
            zip(members, thetas, half_extents)
        ):
            center_along = cursor + half_extent
            target_center = start + tangent * center_along + normal * offset
            booth_width, booth_height = oriented_booth_size(cfg, booth_id, theta)
            base_x = int(round(float(target_center[0]) - booth_width / 2.0))
            base_y = int(round(float(target_center[1]) - booth_height / 2.0))
            jitters = [
                (dx, dy)
                for radius in range(3)
                for dx in range(-radius, radius + 1)
                for dy in range(-radius, radius + 1)
                if max(abs(dx), abs(dy)) == radius
            ]
            if float(rng.random()) < 0.5:
                jitters = list(reversed(jitters))
            placed = False
            for dx, dy in jitters:
                x_pos, y_pos = base_x + dx, base_y + dy
                if not _booth_fits(
                    cfg,
                    venue,
                    temporary,
                    booth_id,
                    x_pos,
                    y_pos,
                    theta,
                ):
                    continue
                claimed_cells = _booth_claimed_cells(
                    cfg, booth_id, x_pos, y_pos, theta
                )
                if any(
                    temporary[cell_y, cell_x]
                    or temporary_front[cell_y, cell_x]
                    for cell_x, cell_y in claimed_cells
                ):
                    continue
                clear_cells = front_clearance_cells(
                    cfg, booth_id, x_pos, y_pos, theta
                )
                if any(
                    not (0 <= cell_x < width and 0 <= cell_y < height)
                    or int(venue.base_grid[cell_y, cell_x]) not in walkable_types
                    or temporary[cell_y, cell_x]
                    for cell_x, cell_y in clear_cells
                ):
                    continue
                states.append((booth_id, x_pos, y_pos, theta))
                for cell_x, cell_y in claimed_cells:
                    temporary[cell_y, cell_x] = True
                for cell_x, cell_y in clear_cells:
                    temporary_front[cell_y, cell_x] = True
                placed = True
                break
            if not placed:
                valid = False
                break
            cursor += 2.0 * half_extent
            if member_index < len(members) - 1:
                cursor += base_gap
        if valid:
            return temporary, temporary_front, states
    return None


def construct_random_orderly_layout(
    cfg: Config,
    rng,
    max_restarts: int = 500,
    lane_trials: int = 400,
    style: str | None = None,
) -> np.ndarray:
    """内部列・外周追従・混成から物理制約充足設計ベクトルを構成する。

    fixed_positionや人手配置は参照しない。会場多角形だけを使って外周配置も
    生成し、染色体は直接表現のまま保持する。最適化後の修復ではない。
    """

    if style not in (None, "lane", "perimeter", "hybrid"):
        raise ValueError("style must be lane, perimeter, hybrid, or None")

    venue = build_base_venue(cfg)
    height, width = cfg.grid_height, cfg.grid_width
    walkable_types = set(int(value) for value in WALKABLE_TYPES)
    # activeとtableを用途別に3～5個の説明可能なモチーフへ構成する。
    groups = direct_layout_groups(cfg, include_tables=True, combine_tables=False)
    for _ in range(int(max_restarts)):
        restart_style = style or str(
            rng.choice(("lane", "perimeter", "hybrid"), p=(0.50, 0.20, 0.30))
        )
        occupied = np.zeros((height, width), dtype=bool)
        front_reserved = np.zeros((height, width), dtype=bool)
        layout = np.full(len(cfg.booths) * 3, -1, dtype=np.int64)
        succeeded = True
        for group in groups:
            for members in _random_lane_partition(group.booth_ids, rng):
                boundary_probability = (
                    0.70
                    if restart_style == "perimeter"
                    else 0.50
                    if restart_style == "hybrid"
                    else 0.0
                )
                prefer_boundary = float(rng.random()) < boundary_probability
                if prefer_boundary:
                    boundary_result = _try_place_boundary_chain(
                        cfg,
                        venue,
                        group,
                        members,
                        occupied,
                        front_reserved,
                        rng,
                        trials=max(20, int(lane_trials) // 4),
                    )
                    if boundary_result is not None:
                        occupied, front_reserved, boundary_states = boundary_result
                        for booth_id, x_pos, y_pos, theta_value in boundary_states:
                            theta_value = 0 if cfg.booths[booth_id].is_table else theta_value
                            base = 3 * booth_id
                            layout[base : base + 3] = (
                                x_pos,
                                y_pos,
                                theta_value,
                            )
                        continue
                lane_placed = False
                for _ in range(int(lane_trials)):
                    axis = "H" if float(rng.random()) < 0.65 else "V"
                    active_theta = int(
                        rng.choice((0, 2) if axis == "H" else (1, 3))
                    )
                    thetas = [
                        (
                            0 if cfg.booths[booth_id].is_table else active_theta
                        )
                        for booth_id in members
                    ]
                    dimensions = [
                        oriented_booth_size(cfg, booth_id, theta_value)
                        for booth_id, theta_value in zip(members, thetas)
                    ]
                    extents = [
                        size[0] if axis == "H" else size[1]
                        for size in dimensions
                    ]
                    base_gap = int(rng.integers(2, 6))
                    # 構成的初期化では等間隔を明示的に保証する。GAはこの後、
                    # 制約範囲内で局所的な疎密を自由に探索できる。
                    gaps = [base_gap for _ in range(len(members) - 1)]
                    total_extent = int(sum(extents) + sum(gaps))
                    along_limit = width if axis == "H" else height
                    cross_limit = height if axis == "H" else width
                    if total_extent > along_limit:
                        continue
                    along = int(rng.integers(0, along_limit - total_extent + 1))
                    cross = int(rng.integers(0, cross_limit))
                    temporary = occupied.copy()
                    temporary_front = front_reserved.copy()
                    candidate_states = []
                    cursor = along
                    valid = True
                    for member_index, (booth_id, theta_value, extent) in enumerate(
                        zip(members, thetas, extents)
                    ):
                        x_pos, y_pos = (
                            (cursor, cross) if axis == "H" else (cross, cursor)
                        )
                        if not _booth_fits(
                            cfg,
                            venue,
                            temporary,
                            booth_id,
                            x_pos,
                            y_pos,
                            theta_value,
                        ):
                            valid = False
                            break
                        claimed_cells = _booth_claimed_cells(
                            cfg, booth_id, x_pos, y_pos, theta_value
                        )
                        # 既存ブースの正面通路へ本体・受付・待機列を置かない。
                        if any(
                            temporary[cell_y, cell_x]
                            or temporary_front[cell_y, cell_x]
                            for cell_x, cell_y in claimed_cells
                        ):
                            valid = False
                            break
                        clear_cells = front_clearance_cells(
                            cfg, booth_id, x_pos, y_pos, theta_value
                        )
                        if any(
                            not (0 <= cell_x < width and 0 <= cell_y < height)
                            or int(venue.base_grid[cell_y, cell_x])
                            not in walkable_types
                            or temporary[cell_y, cell_x]
                            for cell_x, cell_y in clear_cells
                        ):
                            valid = False
                            break
                        candidate_states.append(
                            (booth_id, x_pos, y_pos, theta_value)
                        )
                        for cell_x, cell_y in _booth_claimed_cells(
                            cfg, booth_id, x_pos, y_pos, theta_value
                        ):
                            temporary[cell_y, cell_x] = True
                        for cell_x, cell_y in clear_cells:
                            temporary_front[cell_y, cell_x] = True
                        cursor += extent
                        if member_index < len(gaps):
                            cursor += gaps[member_index]
                    if valid:
                        occupied = temporary
                        front_reserved = temporary_front
                        for booth_id, x_pos, y_pos, theta_value in candidate_states:
                            theta_value = 0 if cfg.booths[booth_id].is_table else theta_value
                            base = 3 * booth_id
                            layout[base : base + 3] = (
                                x_pos,
                                y_pos,
                                theta_value,
                            )
                        lane_placed = True
                        break
                if not lane_placed:
                    succeeded = False
                    break
            if not succeeded:
                break
        if succeeded and np.all(layout >= 0):
            positions = to_integer_layout(layout)
            design = positions_to_design(cfg, positions)
            geometry = build_placement_geometry(cfg, positions)
            if (
                compute_violation(
                    cfg,
                    venue,
                    positions,
                    geometry=geometry,
                )
                <= 0.0
                and compute_direct_orderliness(
                    cfg,
                    design,
                    venue=venue,
                    geometry=geometry,
                    include_path_turns=False,
                    include_wide_aisle=False,
                )["total"]
                <= 1.0
            ):
                return canonicalize_table_design(cfg, design)
    raise RuntimeError(
        "ランダムな内部列/外周追従配置を構成できません。"
        "initial_max_trialsまたはlane_trialsを増やしてください。"
    )


class RandomFeasibleDirectSampling(Sampling):
    """数理最適化を使わず、幾何＋1セル幅到達性を満たす初期集団を作る。"""

    def __init__(
        self,
        cfg: Config,
        seed: int = 1,
        max_trials_per_individual: int = 5000,
        position_trials: int = 1000,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.seed = int(seed)
        self.max_trials_per_individual = int(max_trials_per_individual)
        self.position_trials = int(position_trials)

    def _do(self, problem: Direct138LayoutProblem, n_samples: int, **kwargs):
        output = np.empty((int(n_samples), problem.n_var), dtype=np.int64)
        restarts = max(8, min(32, self.max_trials_per_individual // 200))
        trials_per_booth = max(100, min(400, self.position_trials))
        for row in range(int(n_samples)):
            rng = np.random.default_rng(
                _derived_seed(self.seed, row, 7_000_001)
            )
            accepted = None
            for _ in range(restarts):
                candidate = _greedy_layout_attempt(
                    rng,
                    self.cfg,
                    problem.venue,
                    problem.xl,
                    problem.xu,
                    trials_per_booth,
                )
                if candidate is not None and problem.physical_violation(candidate) <= 0.0:
                    accepted = canonicalize_table_design(
                        self.cfg,
                        np.asarray(candidate, dtype=np.int64),
                    )
                    break
            if accepted is None:
                raise RuntimeError(
                    "ランダム貪欲法で幾何制約と1セル幅到達性を満たす"
                    "初期個体を生成できませんでした"
                )
            output[row] = accepted
        return output


def _initial_individual_task(
    cfg: Config,
    limits: OrderlinessLimits,
    base_seed: int,
    individual_index: int,
    retry_index: int,
    max_trials_per_individual: int,
    position_trials: int,
) -> tuple[int, np.ndarray | None, str | None]:
    """並列数に依存しない乱数系列で初期個体を1つ生成する。"""

    task_seed = _derived_seed(
        int(base_seed),
        int(individual_index),
        int(retry_index),
        1_000_003,
    )
    problem = _cached_direct_problem(cfg, limits)
    sampler = RandomFeasibleDirectSampling(
        cfg,
        seed=task_seed,
        max_trials_per_individual=max_trials_per_individual,
        position_trials=position_trials,
    )
    try:
        return int(individual_index), sampler._do(problem, 1)[0], None
    except RuntimeError as exc:
        # 1ワーカーの失敗で全初期化をRemoteTraceback終了させず、親側で
        # 失敗個体だけ決定的にseedを変えて再試行する。
        return int(individual_index), None, str(exc)


class DirectTwoPointCrossover(Crossover):
    """ブース遺伝子ブロックを分断しない2点交叉。"""

    def __init__(self, cfg: Config, prob: float = 0.90) -> None:
        super().__init__(2, 2, prob=prob)
        self.cfg = cfg
        self.booth_slices = tuple(
            slice(
                booth_variable_indices(cfg, booth_id)[0],
                booth_variable_indices(cfg, booth_id)[0]
                + (2 if booth.is_table else 3),
            )
            for booth_id, booth in enumerate(cfg.booths)
        )

    def _do(self, problem, X, random_state=None, **kwargs):
        rng = random_state if random_state is not None else np.random.default_rng()
        _, n_matings, n_var = X.shape
        offspring = np.empty((2, n_matings, n_var), dtype=np.int64)
        for mating in range(n_matings):
            first = canonicalize_table_design(
                self.cfg,
                np.rint(X[0, mating]).astype(np.int64),
            )
            second = canonicalize_table_design(
                self.cfg,
                np.rint(X[1, mating]).astype(np.int64),
            )
            child_first, child_second = first.copy(), second.copy()

            n_booths = len(self.booth_slices)
            if n_booths >= 3:
                # 切断点はブース間に置く。通常ブース(x,y,theta)や
                # テーブル(x,y)の途中で親を混ぜない。
                cut_first, cut_second = sorted(
                    int(value)
                    for value in rng.choice(
                        np.arange(1, n_booths), size=2, replace=False
                    )
                )
            elif n_booths == 2:
                # 内部境界が1つしかない縮小問題では、1ブース区間を交換する。
                cut_first = int(rng.integers(0, 2))
                cut_second = cut_first + 1
            else:
                cut_first = cut_second = 0

            for sl in self.booth_slices[cut_first:cut_second]:
                child_first[sl] = second[sl]
                child_second[sl] = first[sl]

            child_first = canonicalize_table_design(self.cfg, child_first)
            child_second = canonicalize_table_design(self.cfg, child_second)

            # 交叉では物理的に配置可能かだけを確認する。整然性制約は
            # 第1世代から固定上限で評価し、制約付き環境選択へ委ねる。
            for child, parent in (
                (child_first, first),
                (child_second, second),
            ):
                physical = problem.physical_violation(child)
                if physical > 0.0:
                    child[:] = parent
            offspring[0, mating] = child_first
            offspring[1, mating] = child_second
        return offspring


class DirectMultiScaleMutation(Mutation):
    """可変ブロック直接表現用の局所・中規模・大域突然変異。"""

    def __init__(
        self,
        cfg: Config,
        individual_prob: float = 0.20,
        local_weight: float = 0.55,
        meso_weight: float = 0.30,
        macro_weight: float = 0.15,
        max_position_trials: int = 30,
        exploration_generations: int = 30,
        exploration_individual_prob: float = 0.40,
        exploration_local_weight: float = 0.20,
        exploration_meso_weight: float = 0.30,
        exploration_macro_weight: float = 0.50,
        donor_pool_size: int = 32,
        donor_minimum_distance: float = 0.05,
        donor_refresh_probability: float = 0.25,
    ) -> None:
        super().__init__(prob=1.0)
        if not 0.0 <= individual_prob <= 1.0:
            raise ValueError("individual_prob must be in [0, 1]")
        weights = np.asarray([local_weight, meso_weight, macro_weight], dtype=float)
        if np.any(weights < 0.0) or not np.isclose(float(weights.sum()), 1.0):
            raise ValueError("three mutation weights must sum to 1")
        exploration_weights = np.asarray(
            [
                exploration_local_weight,
                exploration_meso_weight,
                exploration_macro_weight,
            ],
            dtype=float,
        )
        if not 0.0 <= exploration_individual_prob <= 1.0:
            raise ValueError("exploration_individual_prob must be in [0, 1]")
        if np.any(exploration_weights < 0.0) or not np.isclose(
            float(exploration_weights.sum()), 1.0
        ):
            raise ValueError("exploration mutation weights must sum to 1")
        if int(exploration_generations) < 0:
            raise ValueError("exploration_generations must be non-negative")
        if int(donor_pool_size) < 1:
            raise ValueError("donor_pool_size must be positive")
        if not 0.0 <= donor_minimum_distance <= 1.0:
            raise ValueError("donor_minimum_distance must be in [0, 1]")
        if not 0.0 <= donor_refresh_probability <= 1.0:
            raise ValueError("donor_refresh_probability must be in [0, 1]")
        self.cfg = cfg
        self.individual_prob = float(individual_prob)
        self.weights = weights
        self.exploration_generations = int(exploration_generations)
        self.exploration_individual_prob = float(exploration_individual_prob)
        self.exploration_weights = exploration_weights
        self.max_position_trials = int(max_position_trials)
        self.venue = build_base_venue(cfg)
        # 中規模演算も通常ブースとテーブル・椅子の混合レーンへ適用する。
        self.groups = direct_layout_groups(cfg, combine_tables=True)
        self.areas = tuple(sorted({str(booth.area) for booth in cfg.booths}))
        self.donor_pool_size = int(donor_pool_size)
        self.donor_minimum_distance = float(donor_minimum_distance)
        self.donor_refresh_probability = float(donor_refresh_probability)
        self._donor_pool: list[np.ndarray] = []
        self.mutation_history: list[dict] = []

    def settings_for_generation(self, generation: int) -> tuple[float, np.ndarray]:
        """初期30世代は大域探索を強め、以後は通常設定へ戻す。"""
        if int(generation) <= self.exploration_generations:
            return self.exploration_individual_prob, self.exploration_weights
        return self.individual_prob, self.weights

    def _is_feasible(self, problem, x) -> bool:
        return bool(problem.physical_violation(x) <= 0.0)

    def _is_acceptable(self, problem, candidate, current) -> bool:
        """整然性・到達幅の増減を問わず、必須物理制約を守る候補を受理する。"""
        return bool(problem.physical_violation(candidate) <= 0.0)

    def _mutate_local(self, problem, x, rng) -> None:
        count = int(rng.integers(1, min(4, len(self.cfg.booths)) + 1))
        booth_ids = rng.choice(len(self.cfg.booths), size=count, replace=False)
        current = x.copy()
        for booth_raw in booth_ids:
            booth_id = int(booth_raw)
            x_index, y_index, theta_index = booth_variable_indices(
                self.cfg, booth_id
            )
            accepted = False
            for _ in range(self.max_position_trials):
                candidate = current.copy()
                operation = int(rng.integers(0, 2 if theta_index is None else 3))
                if operation < 2:
                    gene = x_index if operation == 0 else y_index
                    # 選択済み配置モチーフの許容幅内で局所差を探索する。
                    step = int(rng.choice([-1, 1]))
                    candidate[gene] = int(
                        np.clip(
                            candidate[gene] + step,
                            problem.xl[gene],
                            problem.xu[gene],
                        )
                    )
                else:
                    # 角度変更の大きさは90度または180度を等確率で選ぶ。
                    # 90度の場合は時計回り・反時計回りも等確率とする。
                    turn = int(rng.choice((1, 2)))
                    if turn == 1 and float(rng.random()) < 0.5:
                        turn = -1
                    candidate[theta_index] = (
                        int(candidate[theta_index]) + turn
                    ) % 4
                if np.array_equal(candidate, current):
                    continue
                if self._is_acceptable(problem, candidate, current):
                    current = candidate
                    accepted = True
                    break
            if not accepted:
                continue
        x[:] = canonicalize_table_design(self.cfg, current)

    def _aligned_subset(self, x, rng) -> list[int]:
        positions = np.asarray(to_integer_layout(x, self.cfg), dtype=np.int64)
        available: list[DirectMotif] = []
        for group in self.groups:
            available.extend(
                detect_direct_motifs(self.cfg, positions, group)
            )
        if available:
            motif = available[int(rng.integers(0, len(available)))]
            return [int(v) for v in motif.booth_ids]
        group = self.groups[int(rng.integers(0, len(self.groups)))]
        return [int(v) for v in group.booth_ids]

    def _meso_swap(self, problem, x, rng) -> bool:
        groups = [group for group in self.groups if len(group.booth_ids) >= 2]
        for _ in range(self.max_position_trials):
            group = groups[int(rng.integers(0, len(groups)))]
            first, second = (
                int(v) for v in rng.choice(group.booth_ids, size=2, replace=False)
            )
            # 同一仕様・同一広場のテーブルは入れ替えても配置が変わらない。
            # 無意味な遺伝子置換を突然変異として数えないよう選び直す。
            if tables_are_equivalent(self.cfg, first, second):
                continue
            candidate = x.copy()
            first_x, first_y, first_theta = booth_variable_indices(self.cfg, first)
            second_x, second_y, second_theta = booth_variable_indices(self.cfg, second)
            first_xy = candidate[[first_x, first_y]].copy()
            candidate[[first_x, first_y]] = candidate[[second_x, second_y]]
            candidate[[second_x, second_y]] = first_xy
            if first_theta is not None and second_theta is not None:
                candidate[first_theta], candidate[second_theta] = (
                    candidate[second_theta], candidate[first_theta]
                )
            if self._is_acceptable(problem, candidate, x):
                x[:] = candidate
                return True
        return False

    def _meso_translate(self, problem, x, rng) -> bool:
        for _ in range(self.max_position_trials):
            members = self._aligned_subset(x, rng)
            candidate = x.copy()
            distance = int(rng.integers(2, 7))
            dx, dy = ((distance, 0) if float(rng.random()) < 0.5 else (0, distance))
            if float(rng.random()) < 0.5:
                dx, dy = -dx, -dy
            for booth_id in members:
                x_index, y_index, _ = booth_variable_indices(self.cfg, booth_id)
                candidate[x_index] = int(
                    np.clip(candidate[x_index] + dx, problem.xl[x_index], problem.xu[x_index])
                )
                candidate[y_index] = int(
                    np.clip(
                        candidate[y_index] + dy,
                        problem.xl[y_index],
                        problem.xu[y_index],
                    )
                )
            if not np.array_equal(candidate, x) and self._is_acceptable(problem, candidate, x):
                x[:] = candidate
                return True
        return False

    def _meso_rotate(self, problem, x, rng) -> bool:
        for _ in range(self.max_position_trials):
            members = self._aligned_subset(x, rng)
            candidate = x.copy()
            positions = np.asarray(to_integer_layout(candidate, self.cfg), dtype=np.int64)
            center_x = float(np.mean(positions[members, 0]))
            center_y = float(np.mean(positions[members, 1]))
            direction = -1 if float(rng.random()) < 0.5 else 1
            for booth_id in members:
                x_index, y_index, theta_index = booth_variable_indices(
                    self.cfg, booth_id
                )
                rel_x = int(candidate[x_index]) - center_x
                rel_y = int(candidate[y_index]) - center_y
                candidate[x_index] = int(
                    np.clip(
                        round(center_x - direction * rel_y),
                        problem.xl[x_index],
                        problem.xu[x_index],
                    )
                )
                candidate[y_index] = int(
                    np.clip(
                        round(center_y + direction * rel_x),
                        problem.xl[y_index],
                        problem.xu[y_index],
                    )
                )
                if theta_index is not None:
                    candidate[theta_index] = (
                        int(candidate[theta_index]) + direction
                    ) % 4
            if self._is_acceptable(problem, candidate, x):
                x[:] = candidate
                return True
        return False

    def _mutate_meso(self, problem, x, rng) -> None:
        operation = int(rng.integers(0, 3))
        if operation == 0:
            succeeded = self._meso_swap(problem, x, rng)
        elif operation == 1:
            succeeded = self._meso_translate(problem, x, rng)
        else:
            succeeded = self._meso_rotate(problem, x, rng)
        if not succeeded:
            self._mutate_local(problem, x, rng)

    def _random_donor(self, problem, rng) -> np.ndarray | None:
        """大域変異用の実行可能ドナーを返す。

        ドナー候補の不成立は通常の突然変異失敗であり、最適化全体を停止させる
        エラーではない。計算時間が暴走しない範囲で再試行し、全候補が不成立なら
        ``None`` を返して呼び出し元に元個体を維持させる。
        """

        donor_attempts = max(2, min(16, self.max_position_trials))
        donor_restarts = max(50, min(500, self.max_position_trials * 10))
        should_generate = (
            len(self._donor_pool) < self.donor_pool_size
            or float(rng.random()) < self.donor_refresh_probability
        )
        if should_generate:
            for _ in range(donor_attempts):
                try:
                    donor = construct_random_orderly_layout(
                        self.cfg,
                        rng,
                        max_restarts=donor_restarts,
                    )
                except RuntimeError:
                    continue
                if not self._is_feasible(problem, donor):
                    continue
                if self._donor_pool:
                    nearest = min(
                        normalized_layout_distance(self.cfg, donor, existing)
                        for existing in self._donor_pool
                    )
                    if nearest + 1.0e-12 < self.donor_minimum_distance:
                        continue
                if len(self._donor_pool) >= self.donor_pool_size:
                    replace_index = int(rng.integers(0, len(self._donor_pool)))
                    self._donor_pool[replace_index] = donor.copy()
                else:
                    self._donor_pool.append(donor.copy())
                return donor.copy()

        if self._donor_pool:
            donor_id = int(rng.integers(0, len(self._donor_pool)))
            return self._donor_pool[donor_id].copy()
        return None

    def _macro_area_reseed(self, problem, x, rng) -> bool:
        donor = self._random_donor(problem, rng)
        if donor is None:
            return False
        area = str(rng.choice(self.areas))
        candidate = x.copy()
        for booth_id, booth in enumerate(self.cfg.booths):
            if str(booth.area) == area:
                x_index, _, theta_index = booth_variable_indices(self.cfg, booth_id)
                sl = slice(x_index, x_index + (2 if theta_index is None else 3))
                candidate[sl] = donor[sl]
        x[:] = candidate if self._is_acceptable(problem, candidate, x) else donor
        return True

    def _macro_transform(self, problem, x, rng, mirror: bool) -> bool:
        for _ in range(self.max_position_trials):
            area = str(rng.choice(self.areas))
            members = [
                booth_id
                for booth_id, booth in enumerate(self.cfg.booths)
                if str(booth.area) == area
            ]
            candidate = x.copy()
            positions = np.asarray(to_integer_layout(candidate, self.cfg), dtype=np.int64)
            if mirror:
                axis_x = float(np.mean(positions[members, 0]))
                horizontal = float(rng.random()) < 0.5
                axis_y = float(np.mean(positions[members, 1]))
                for booth_id in members:
                    x_index, y_index, theta_index = booth_variable_indices(
                        self.cfg, booth_id
                    )
                    if horizontal:
                        candidate[x_index] = int(
                            np.clip(
                                round(2 * axis_x - candidate[x_index]),
                                problem.xl[x_index],
                                problem.xu[x_index],
                            )
                        )
                        if theta_index is not None:
                            candidate[theta_index] = (-int(candidate[theta_index])) % 4
                    else:
                        candidate[y_index] = int(
                            np.clip(
                                round(2 * axis_y - candidate[y_index]),
                                problem.xl[y_index],
                                problem.xu[y_index],
                            )
                        )
                        if theta_index is not None:
                            candidate[theta_index] = (2 - int(candidate[theta_index])) % 4
            else:
                distance = int(rng.integers(4, 11))
                dx, dy = (
                    (distance, 0) if float(rng.random()) < 0.5 else (0, distance)
                )
                if float(rng.random()) < 0.5:
                    dx, dy = -dx, -dy
                for booth_id in members:
                    x_index, y_index, _ = booth_variable_indices(self.cfg, booth_id)
                    candidate[x_index] = int(
                        np.clip(candidate[x_index] + dx, problem.xl[x_index], problem.xu[x_index])
                    )
                    candidate[y_index] = int(
                        np.clip(
                            candidate[y_index] + dy,
                            problem.xl[y_index],
                            problem.xu[y_index],
                        )
                    )
            if not np.array_equal(candidate, x) and self._is_acceptable(problem, candidate, x):
                x[:] = candidate
                return True
        return False

    def _mutate_macro(self, problem, x, rng) -> None:
        operation = int(rng.integers(0, 3))
        if operation == 0:
            succeeded = self._macro_area_reseed(problem, x, rng)
        elif operation == 1:
            succeeded = self._macro_transform(problem, x, rng, mirror=True)
            if not succeeded:
                succeeded = self._macro_area_reseed(problem, x, rng)
        else:
            succeeded = self._macro_transform(problem, x, rng, mirror=False)
            if not succeeded:
                succeeded = self._macro_area_reseed(problem, x, rng)
        # 厳しい制約下でドナーも変換も不成立なら、実行可能な局所変異で
        # 完全な不発を避ける。通常は上の大域操作が成立する。
        if not succeeded:
            self._mutate_local(problem, x, rng)

    def _do(self, problem, X, random_state=None, **kwargs):
        rng = random_state if random_state is not None else np.random.default_rng()
        output = np.asarray(
            [
                canonicalize_table_design(self.cfg, row)
                for row in np.rint(np.asarray(X)).astype(np.int64, copy=True)
            ],
            dtype=np.int64,
        )
        algorithm = kwargs.get("algorithm")
        generation = int(getattr(algorithm, "n_gen", 1) or 1)
        individual_probability, weights = self.settings_for_generation(generation)
        attempted = np.zeros(3, dtype=np.int64)
        changed = np.zeros(3, dtype=np.int64)
        for row in range(len(output)):
            if float(rng.random()) >= individual_probability:
                continue
            scale = int(rng.choice(3, p=weights))
            before = output[row].copy()
            attempted[scale] += 1
            if scale == 0:
                self._mutate_local(problem, output[row], rng)
            elif scale == 1:
                self._mutate_meso(problem, output[row], rng)
            else:
                self._mutate_macro(problem, output[row], rng)
            if not np.array_equal(before, output[row]):
                changed[scale] += 1
            # 各スケールの操作は、実行可能性を確認した候補だけを反映する。
            # ここで同じ設計変数を再判定する必要はない。
            output[row] = canonicalize_table_design(self.cfg, output[row])
        self.mutation_history.append(
            {
                "generation": generation,
                "individual_probability": float(individual_probability),
                "local_weight": float(weights[0]),
                "meso_weight": float(weights[1]),
                "macro_weight": float(weights[2]),
                "local_attempted": int(attempted[0]),
                "meso_attempted": int(attempted[1]),
                "macro_attempted": int(attempted[2]),
                "local_changed": int(changed[0]),
                "meso_changed": int(changed[1]),
                "macro_changed": int(changed[2]),
                "donor_pool_size": int(len(self._donor_pool)),
            }
        )
        return output


def _memetic_neighbor_task(
    cfg: Config,
    limits: OrderlinessLimits,
    parent_x: np.ndarray,
    known_layouts: tuple[tuple[int, ...], ...],
    task_seed: int,
    n_neighbors: int,
    max_neighbor_trials: int,
) -> list[np.ndarray]:
    """1親の実行可能局所近傍を、独立した決定的乱数系列で生成する。"""

    problem = _cached_direct_problem(cfg, limits)
    operator = _MEMETIC_OPERATOR_CACHE.get(cfg)
    if operator is None:
        operator = DirectMultiScaleMutation(
            cfg,
            individual_prob=1.0,
            local_weight=1.0,
            meso_weight=0.0,
            macro_weight=0.0,
            max_position_trials=50,
        )
        _MEMETIC_OPERATOR_CACHE[cfg] = operator
    rng = np.random.default_rng(int(task_seed))
    parent = np.asarray(parent_x, dtype=np.int64)
    known = set(known_layouts)
    neighbors: list[np.ndarray] = []
    for _ in range(int(max_neighbor_trials)):
        candidate = parent.copy()
        operator._mutate_local(problem, candidate, rng)
        key = tuple(int(value) for value in candidate)
        if np.array_equal(candidate, parent) or key in known:
            continue
        # _mutate_localは実行可能な変更だけをcandidateへ反映する。
        neighbors.append(candidate)
        known.add(key)
        if len(neighbors) >= int(n_neighbors):
            break
    return neighbors


class DirectMemeticNSGA2(NSGA2):
    """一定世代ごとに非劣解の実行可能近傍を追加評価するNSGA-II。"""

    def __init__(
        self,
        cfg: Config,
        local_search_interval: int = 5,
        local_search_start_generation: int = 30,
        n_nondominated_parents: int = 10,
        n_neighbors_per_parent: int = 10,
        max_neighbor_trials: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if local_search_interval < 1:
            raise ValueError("local_search_interval must be at least 1")
        if local_search_start_generation < 1:
            raise ValueError("local_search_start_generation must be at least 1")
        if n_nondominated_parents < 1 or n_neighbors_per_parent < 1:
            raise ValueError("memetic parent and neighbor counts must be positive")
        if max_neighbor_trials < n_neighbors_per_parent:
            raise ValueError("max_neighbor_trials is too small")
        self.cfg = cfg
        self.local_search_interval = int(local_search_interval)
        self.local_search_start_generation = int(local_search_start_generation)
        self.n_nondominated_parents = int(n_nondominated_parents)
        self.n_neighbors_per_parent = int(n_neighbors_per_parent)
        self.max_neighbor_trials = int(max_neighbor_trials)
        self.memetic_history: list[dict[str, int]] = []
        self._memetic_rng: np.random.Generator | None = None

    def _rng(self) -> np.random.Generator:
        """pymoo 0.6.1.x の細かなAPI差を避けた専用乱数器を返す。"""

        if self._memetic_rng is None:
            seed = getattr(self, "seed", None)
            self._memetic_rng = np.random.default_rng(
                1 if seed is None else int(seed) + 104729
            )
        return self._memetic_rng

    def _post_advance(self) -> None:
        if (
            self.pop is not None
            and self.n_gen is not None
            and int(self.n_gen) >= self.local_search_start_generation
            and int(self.n_gen) % self.local_search_interval == 0
        ):
            self._apply_direct_local_search()
        # callbackとoptimumには、近傍を統合した後の集団を渡す。
        super()._post_advance()

    def _select_nondominated_indices(self) -> np.ndarray:
        """受容可能解、未獲得時は物理実行可能な低違反解から親を選ぶ。"""

        if self.pop is None or len(self.pop) == 0:
            return np.empty(0, dtype=np.int64)
        rank_raw = self.pop.get("rank")
        rank = (
            np.full(len(self.pop), np.inf, dtype=float)
            if rank_raw is None
            else np.asarray(rank_raw, dtype=float).reshape(-1)
        )
        if rank.size != len(self.pop):
            rank = np.full(len(self.pop), np.inf, dtype=float)
        objectives = np.atleast_2d(np.asarray(self.pop.get("F"), dtype=float))
        feasible_raw = self.pop.get("FEAS")
        feasible = (
            np.ones(len(self.pop), dtype=bool)
            if feasible_raw is None
            else np.asarray(feasible_raw, dtype=bool).reshape(-1)
        )
        pool = np.where(
            feasible
            & (rank == 0)
            & np.all(np.isfinite(objectives), axis=1)
        )[0]
        # 整然性制約を全て満たす個体がまだ無い段階でも局所探索を止めない。
        # 必須物理制約を満たす個体のうち、重み付き受容可能性違反が小さい順に
        # 親候補とする。Gのpath_turns成分は既に3倍されているため、ここでも
        # 方向転換の改善が強く優先される。
        if pool.size == 0:
            raw_g = self.pop.get("G")
            constraints = (
                np.empty((len(self.pop), 0), dtype=float)
                if raw_g is None
                else np.asarray(raw_g, dtype=float)
            )
            if constraints.ndim == 1:
                constraints = constraints.reshape(-1, 1)
            if constraints.shape[0] == len(self.pop) and constraints.shape[1] >= 1:
                physical_feasible = (
                    np.isfinite(constraints[:, 0])
                    & (constraints[:, 0] <= 0.0)
                    & np.all(np.isfinite(objectives), axis=1)
                )
                acceptability_violation = np.sum(
                    np.maximum(constraints[:, 1:], 0.0),
                    axis=1,
                )
                ordered = np.lexsort(
                    (
                        self._rng().random(len(self.pop)),
                        acceptability_violation,
                    )
                )
                pool = ordered[physical_feasible[ordered]][
                    : self.n_nondominated_parents
                ]
        count = min(self.n_nondominated_parents, int(pool.size))
        if count == 0:
            return np.empty(0, dtype=np.int64)
        return np.asarray(
            self._rng().choice(pool, size=count, replace=False),
            dtype=np.int64,
        )

    def _generate_local_neighbors(
        self,
        parent_x: np.ndarray,
        known_layouts: set[tuple[int, ...]],
    ) -> list[np.ndarray]:
        """1～4配置物の±1セル移動・90/180度回転から実行可能近傍を作る。"""

        task_seed = int(
            self._rng().integers(0, np.iinfo(np.int64).max, dtype=np.int64)
        )
        neighbors = _memetic_neighbor_task(
            self.cfg,
            self.problem.orderliness_limits,
            np.asarray(parent_x, dtype=np.int64),
            tuple(sorted(known_layouts)),
            task_seed,
            self.n_neighbors_per_parent,
            self.max_neighbor_trials,
        )
        known_layouts.update(
            tuple(int(value) for value in candidate)
            for candidate in neighbors
        )
        return neighbors

    def _parallel_neighbor_batches(
        self,
        parent_indices: np.ndarray,
        current_x: np.ndarray,
        known_layouts: set[tuple[int, ...]],
    ) -> list[np.ndarray]:
        """親単位の決定的タスクを、評価用Poolがあれば並列実行する。"""

        parent_ids = [int(value) for value in parent_indices]
        collected: list[list[np.ndarray]] = [[] for _ in parent_ids]
        remaining = [self.n_neighbors_per_parent for _ in parent_ids]
        runner = getattr(self.problem, "elementwise_runner", None)
        starmap = getattr(runner, "starmap", None)
        base_seed = int(getattr(self, "seed", 1) or 1)
        generation = int(self.n_gen or 0)

        # 通常は1回で10近傍揃う。親間で同一近傍が発生した場合だけ再生成する。
        for round_index in range(8):
            active = [index for index, count in enumerate(remaining) if count > 0]
            if not active:
                break
            known_snapshot = tuple(sorted(known_layouts))
            tasks = [
                (
                    self.cfg,
                    self.problem.orderliness_limits,
                    current_x[parent_ids[index]],
                    known_snapshot,
                    _derived_seed(
                        base_seed,
                        generation,
                        parent_ids[index],
                        round_index,
                        2_000_003,
                    ),
                    remaining[index],
                    self.max_neighbor_trials,
                )
                for index in active
            ]
            if starmap is not None and len(tasks) > 1:
                generated = list(starmap(_memetic_neighbor_task, tasks))
            else:
                generated = [_memetic_neighbor_task(*task) for task in tasks]

            progress = 0
            for local_index, candidates in zip(active, generated):
                for candidate in candidates:
                    key = tuple(int(value) for value in candidate)
                    if key in known_layouts:
                        continue
                    collected[local_index].append(candidate)
                    known_layouts.add(key)
                    remaining[local_index] -= 1
                    progress += 1
                    if remaining[local_index] <= 0:
                        break
            if progress == 0:
                break
        return [candidate for batch in collected for candidate in batch]

    def _apply_direct_local_search(self) -> None:
        """選んだ非劣解の近傍を評価し、通常のNSGA-II環境選択へ統合する。"""

        parent_indices = self._select_nondominated_indices()
        if parent_indices.size == 0:
            return
        current_x = np.asarray(self.pop.get("X"), dtype=np.int64)
        known_layouts = {
            tuple(int(value) for value in row)
            for row in current_x
        }
        neighbor_vectors = self._parallel_neighbor_batches(
            parent_indices,
            current_x,
            known_layouts,
        )
        self.memetic_history.append(
            {
                "generation": int(self.n_gen),
                "selected_parents": int(parent_indices.size),
                "evaluated_neighbors": int(len(neighbor_vectors)),
            }
        )
        if not neighbor_vectors:
            return
        local_population = Population.new(
            "X",
            np.asarray(neighbor_vectors, dtype=np.int64),
        )
        self.evaluator.eval(
            self.problem,
            local_population,
            algorithm=self,
        )
        merged = Population.merge(self.pop, local_population)
        self.pop = self.survival.do(
            self.problem,
            merged,
            n_survive=self.pop_size,
            algorithm=self,
        )


def run_direct_nsga2(
    cfg: Config,
    pop_size: int | None = None,
    n_gen: int | None = None,
    elementwise_runner=None,
    seed: int = 1,
    verbose: bool = True,
    orderliness_limits: OrderlinessLimits | None = None,
    max_orderliness: float | None = None,
    initial_max_trials: int = 5000,
    initial_genotypes: np.ndarray | None = None,
    mutation_individual_prob: float = 0.20,
    mutation_local_weight: float = 0.55,
    mutation_meso_weight: float = 0.30,
    mutation_macro_weight: float = 0.15,
    exploration_generations: int = 30,
    exploration_individual_prob: float = 0.40,
    exploration_local_weight: float = 0.20,
    exploration_meso_weight: float = 0.30,
    exploration_macro_weight: float = 0.50,
    donor_pool_size: int = 32,
    donor_minimum_distance: float = 0.05,
    donor_refresh_probability: float = 0.25,
    minimum_layout_distance: float = 0.08,
    memetic_interval: int = 5,
    memetic_start_generation: int = 30,
    memetic_parents: int = 10,
    memetic_neighbors: int = 10,
    memetic_max_neighbor_trials: int = 1000,
    on_generation=None,
):
    """通常ブース3変数・テーブル2変数の直接表現NSGA-IIを実行する。"""

    kwargs = {}
    if elementwise_runner is not None:
        kwargs["elementwise_runner"] = elementwise_runner
    problem = Direct138LayoutProblem(
        cfg,
        orderliness_limits=orderliness_limits,
        max_orderliness=max_orderliness,
        **kwargs,
    )
    effective_n_gen = int(n_gen or cfg.n_generations)
    # 指定された整然性上限を第1世代から最終世代まで固定して使用する。
    fixed_limits = problem.orderliness_limits
    if initial_genotypes is None:
        sampling = prepare_random_feasible_direct_population(
            cfg,
            pop_size=int(pop_size or cfg.population_size),
            seed=seed,
            orderliness_limits=fixed_limits,
            initial_max_trials=initial_max_trials,
        )
    else:
        sampling = np.asarray(
            [
                canonicalize_table_design(cfg, row)
                for row in np.atleast_2d(
                    np.asarray(initial_genotypes, dtype=np.int64)
                )
            ],
            dtype=np.int64,
        )
        expected = int(pop_size or cfg.population_size)
        if sampling.shape != (expected, problem.n_var):
            raise ValueError(
                "initial_genotypes shape mismatch: "
                f"expected {(expected, problem.n_var)}, got {sampling.shape}"
            )
        invalid_physical = [
            index
            for index, row in enumerate(sampling)
            if problem.physical_violation(row) > 0.0
        ]
        if invalid_physical:
            raise ValueError(
                "initial_genotypes contains physical constraint violations: "
                f"indices={invalid_physical[:10]}"
            )

    algorithm = DirectMemeticNSGA2(
        cfg=cfg,
        local_search_interval=memetic_interval,
        local_search_start_generation=memetic_start_generation,
        n_nondominated_parents=memetic_parents,
        n_neighbors_per_parent=memetic_neighbors,
        max_neighbor_trials=memetic_max_neighbor_trials,
        pop_size=int(pop_size or cfg.population_size),
        sampling=sampling,
        survival=DirectDiversitySurvival(
            cfg,
            minimum_layout_distance=minimum_layout_distance,
        ),
        crossover=DirectTwoPointCrossover(cfg, prob=0.90),
        mutation=DirectMultiScaleMutation(
            cfg,
            individual_prob=mutation_individual_prob,
            local_weight=mutation_local_weight,
            meso_weight=mutation_meso_weight,
            macro_weight=mutation_macro_weight,
            exploration_generations=exploration_generations,
            exploration_individual_prob=exploration_individual_prob,
            exploration_local_weight=exploration_local_weight,
            exploration_meso_weight=exploration_meso_weight,
            exploration_macro_weight=exploration_macro_weight,
            donor_pool_size=donor_pool_size,
            donor_minimum_distance=donor_minimum_distance,
            donor_refresh_probability=donor_refresh_probability,
        ),
        eliminate_duplicates=True,
        # 受容可能性制約を満たす解が探索途中にまだ無い場合も、物理制約を
        # 満たす中で違反が最小の個体を返せるようにする。
        return_least_infeasible=True,
    )
    callback = ObjectiveHistoryCallback(on_generation=on_generation)
    result = minimize(
        problem,
        algorithm,
        ("n_gen", effective_n_gen),
        seed=seed,
        verbose=verbose,
        callback=callback,
        save_history=False,
    )
    result.objective_history = callback.records
    if result.X is None:
        raise RuntimeError("直接表現GAから実行可能なパレート解を取得できませんでした")

    designs = np.asarray(
        [
            canonicalize_table_design(cfg, row)
            for row in np.atleast_2d(np.asarray(result.X, dtype=np.int64))
        ],
        dtype=np.int64,
    )
    result.X = designs[0] if np.asarray(result.X).ndim == 1 else designs
    result.width_reachability_profiles = [
        compute_width_reachability_profile(
            cfg,
            problem.venue,
            to_integer_layout(x, cfg),
        )
        for x in designs
    ]
    components = [
        compute_direct_orderliness(
            cfg,
            x,
            limits=fixed_limits,
            venue=problem.venue,
            width_profile=width_profile,
        )
        for x, width_profile in zip(
            designs,
            result.width_reachability_profiles,
        )
    ]
    result.orderliness_scores = np.asarray(
        [item["total"] for item in components], dtype=float
    )
    result.orderliness_components = components
    result.orderliness_constraint_vectors = [
        _orderliness_constraint_vector(item, fixed_limits).tolist()
        for item in components
    ]
    result.acceptability_satisfied = np.asarray(
        [
            np.all(np.asarray(vector, dtype=float) <= 0.0)
            for vector in result.orderliness_constraint_vectors
        ],
        dtype=bool,
    )
    result.orderliness_limits = fixed_limits
    result.mutation_settings = {
        "exploration_phase": {
            "through_generation": int(exploration_generations),
            "individual_probability": float(exploration_individual_prob),
            "local_weight": float(exploration_local_weight),
            "meso_weight": float(exploration_meso_weight),
            "macro_weight": float(exploration_macro_weight),
        },
        "standard_phase": {
            "individual_probability": float(mutation_individual_prob),
            "local_weight": float(mutation_local_weight),
            "meso_weight": float(mutation_meso_weight),
            "macro_weight": float(mutation_macro_weight),
        },
        "donor_pool": {
            "maximum_size": int(donor_pool_size),
            "minimum_layout_distance": float(donor_minimum_distance),
            "refresh_probability": float(donor_refresh_probability),
        },
    }
    result.memetic_settings = {
        "start_generation": int(memetic_start_generation),
        "interval_generations": int(memetic_interval),
        "nondominated_parents": int(memetic_parents),
        "neighbors_per_parent": int(memetic_neighbors),
        "max_neighbor_trials": int(memetic_max_neighbor_trials),
    }
    # minimize() は既定で algorithm を deepcopy して実行するため、実行履歴は
    # result.algorithm から回収する（元の algorithm は空のまま）。
    result.memetic_history = list(result.algorithm.memetic_history)
    result.orderliness_constraint_mode = (
        "physical_hard_plus_orderliness_acceptability_constraints"
    )
    result.diversity_settings = {
        "minimum_layout_distance": float(minimum_layout_distance),
        "distance": (
            "identifier_independent_coarse_footprint_jaccard_80pct_"
            "plus_facing_pattern_jaccard_20pct"
        ),
        "coarse_bin_cells": int(VISUAL_DIVERSITY_BIN_CELLS),
        "novelty_crowding_weight": float(
            LAYOUT_NOVELTY_CROWDING_WEIGHT
        ),
        "selection": (
            "physical_then_acceptability_then_rank_crowding_plus_"
            "visual_layout_novelty"
        ),
    }
    result.diversity_history = list(result.algorithm.survival.history)
    executed_mutation = result.algorithm.mating.mutation
    result.mutation_history = list(executed_mutation.mutation_history)
    result.donor_pool_final_size = int(len(executed_mutation._donor_pool))
    return result


def prepare_random_feasible_direct_population(
    cfg: Config,
    pop_size: int,
    seed: int = 1,
    orderliness_limits: OrderlinessLimits | None = None,
    max_orderliness: float | None = None,
    initial_max_trials: int = 5000,
    starmap=None,
) -> np.ndarray:
    """決定的な個体別乱数系列で、物理制約充足初期集団を生成する。

    ``starmap`` の有無にかかわらず各個体のseedは同じなので、ワーカー数を
    変更しても同じ初期集団を返す。重複時だけその個体のretry番号を進める。
    """

    problem = Direct138LayoutProblem(
        cfg,
        orderliness_limits=orderliness_limits,
        max_orderliness=max_orderliness,
    )
    limits = problem.orderliness_limits
    n_samples = int(pop_size)
    output = np.empty((n_samples, problem.n_var), dtype=np.int64)
    retry_counts = np.zeros(n_samples, dtype=np.int64)
    pending = list(range(n_samples))
    known: set[tuple[int, ...]] = set()
    while pending:
        tasks = [
            (
                cfg,
                limits,
                int(seed),
                int(index),
                int(retry_counts[index]),
                int(initial_max_trials),
                250,
            )
            for index in pending
        ]
        if starmap is None:
            generated = [_initial_individual_task(*task) for task in tasks]
        else:
            generated = list(starmap(_initial_individual_task, tasks))
        next_pending = []
        for index, candidate, error in sorted(generated, key=lambda item: item[0]):
            if candidate is None:
                retry_counts[index] += 1
                if retry_counts[index] >= 12:
                    raise RuntimeError(
                        "幾何制約と1セル幅到達性を満たす初期個体を"
                        f"再試行内に生成できませんでした: {error or '原因不明'}"
                    )
                next_pending.append(int(index))
                continue
            key = tuple(int(value) for value in candidate)
            if key in known:
                retry_counts[index] += 1
                if retry_counts[index] >= 12:
                    raise RuntimeError(
                        "重複しないランダム制約充足初期個体を生成できません"
                    )
                next_pending.append(int(index))
                continue
            output[index] = candidate
            known.add(key)
        pending = next_pending
    return output


def save_direct_metadata(result, out_dir: str | Path) -> None:
    """直接設計値、整然さ内訳、最適化方式を保存する。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    designs = np.atleast_2d(np.asarray(result.X, dtype=np.int64))
    rows = []
    for index, (design, components, width_profile) in enumerate(
        zip(
            designs,
            result.orderliness_components,
            result.width_reachability_profiles,
        )
    ):
        row = {"pareto_index": index}
        row.update({f"x{gene}": int(value) for gene, value in enumerate(design)})
        for key in ("total", *ORDERLINESS_COMPONENT_KEYS):
            row[f"order_{key}"] = float(components[key])
        constraint_vector = np.asarray(
            result.orderliness_constraint_vectors[index],
            dtype=float,
        )
        for key, value in zip(ORDERLINESS_CONSTRAINT_KEYS, constraint_vector):
            row[f"constraint_{key}"] = float(value)
        row["acceptability_satisfied"] = bool(
            result.acceptability_satisfied[index]
        )
        row["width4_achievement_count"] = int(
            width_profile["four_cell_achievement_count"]
        )
        # 幅の列も TARGET_CORRIDOR_WIDTH_CELLS から生成する(4 直書きだと
        # 目標幅を下げたとき exact_counts に 4 が無く KeyError になる)。
        for width in range(TARGET_CORRIDOR_WIDTH_CELLS, 0, -1):
            row[f"exact_width_{width}_booths"] = int(
                width_profile["exact_counts"].get(width, 0)
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        output / f"direct_{designs.shape[1]}_designs.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(result.memetic_history).to_csv(
        output / "memetic_history.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(result.mutation_history).to_csv(
        output / "mutation_history.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(result.diversity_history).to_csv(
        output / "layout_diversity_history.csv", index=False, encoding="utf-8-sig"
    )
    payload = {
        "method": f"single_level_direct_{designs.shape[1]}_layout_nsga2",
        "n_genes": int(designs.shape[1]),
        "encoding": "active booth=(x,y,theta), table=(x,y)",
        "equivalent_tables": {
            "equivalence": "same_area_and_same_simulation_specification",
            "canonicalization": "coordinates_sorted_by_y_then_x_and_assigned_to_ascending_booth_id",
            "equivalent_table_swap_mutation": "forbidden",
        },
        "crossover": {
            "type": "booth_block_two_point",
            "probability": 0.90,
            "cut_points": "between_booths",
            "preserves_booth_gene_blocks": True,
            "invalid_offspring": "revert_to_corresponding_parent",
        },
        "orderliness_limits": result.orderliness_limits.as_dict(),
        "initialization": "random_greedy_geometry_plus_one_cell_reachability",
        "uses_fixed_position_for_initialization": False,
        "post_optimization_repair": False,
        "reachability_constraints": {
            "initialization_and_operator_hard_constraint": "geometry_plus_one_cell_width",
            "environmental_selection": (
                "three_cell_active_booth_deficit_is_a_loose_acceptability_constraint"
            ),
            "simulation_walkable": "widest_available_path_per_target_in_order_4_3_2_1",
            "final_pareto_requirement": "all_booths_at_least_one_cell_reachable",
        },
        "objectives": {
            "f_move": "mean_free_walk_seconds_from_attached_metrics",
            "f_congestion": "booth_near_congestion_from_attached_metrics",
        },
        "orderliness": {
            "role": (
                "front_clearance_is_physical_hard_constraint_"
                "seven_nonredundant_metrics_are_acceptability_constraints"
            ),
            "aggregation": "seven_independent_constraints_without_compensation",
            "constraint_keys": list(ORDERLINESS_CONSTRAINT_KEYS),
            "diagnostic_only_keys": [
                "organization_nonmembership",
                "motif_complexity",
                "front_clearance",
            ],
            "path_turn_constraint_weight": float(PATH_TURN_CONSTRAINT_WEIGHT),
            "grouping": "area_x_mixed_plus_active_and_table_checks",
            "motifs": ["interior_horizontal_or_vertical_lane", "venue_boundary_chain"],
            "tables": "same_motif_rules_as_active_booths_without_orientation_check",
            "minimum_strong_lane_size": MIN_STRONG_LANE_SIZE,
            "minimum_strong_motif_size": MIN_STRONG_MOTIF_SIZE,
            "minimum_boundary_chain_size": MIN_BOUNDARY_CHAIN_SIZE,
            "wide_aisle_width_cells": WIDE_AISLE_WIDTH_CELLS,
            "boundary_band_cells": BOUNDARY_BAND_CELLS,
            "boundary_alignment_tolerance_cells": BOUNDARY_ALIGNMENT_TOLERANCE_CELLS,
            "motif_target_size": MOTIF_TARGET_SIZE,
            "lane_alignment_tolerance_cells": LANE_ALIGNMENT_TOLERANCE_CELLS,
            "maximum_lane_edge_gap_cells": MAX_LANE_EDGE_GAP_CELLS,
            "front_clearance_depth_cells": FRONT_CLEARANCE_DEPTH_CELLS,
            "weighted_compensation": False,
        },
        "mutation": {
            **result.mutation_settings,
            "local_orientation_change": {
                "magnitude_degrees": [90, 180],
                "magnitude_probability": [0.5, 0.5],
                "ninety_degree_direction": "clockwise_or_counterclockwise_with_equal_probability",
            },
        },
        "memetic": result.memetic_settings,
        "orderliness_constraint_application": {
            "mode": result.orderliness_constraint_mode,
            "limits": result.orderliness_limits.as_dict(),
            "physical_constraint_index": 0,
            "acceptability_constraint_indices": list(
                range(1, 1 + len(ORDERLINESS_CONSTRAINT_KEYS))
            ),
            "selection_priority": [
                "physically_feasible",
                "all_acceptability_constraints_satisfied",
                "minimum_weighted_acceptability_violation",
            ],
        },
        "layout_diversity": result.diversity_settings,
        "donor_pool_final_size": int(result.donor_pool_final_size),
        "exact_speedups": {
            "constraint_cache": "bounded exact integer-genotype cache per process",
            "shared_geometry": "physical and orderliness checks reuse identical placement geometry",
            "shared_motif_features": "centers, sizes, lanes, groups, and boundary segments are reused",
            "redundant_feasibility_checks_removed": True,
            "parallel_initialization": "bounded random-greedy tasks with deterministic per-individual seeds",
            "parallel_memetic_generation": "deterministic per-parent tasks on the evaluation process pool",
            "surrogate_or_approximation": False,
            "evaluation_budget_changed": False,
        },
    }
    (output / f"direct_{designs.shape[1]}_method.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
