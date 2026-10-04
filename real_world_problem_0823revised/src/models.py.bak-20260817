"""models.py — 定数・dataclass 群・設定ローダ(仕様書 v2 §2, §3, §5, §6)。

座標系(§3.5):
- グリッドは NumPy 配列 grid[y, x]
- x は列インデックス(0 が左端、右へ増加)
- y は行インデックス(0 が上端、下へ増加)
- 位置タプルは (x, y) の順で統一
- ブース基準座標はブースの左上セル
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------- セル種別(§4.2)
EMPTY = 0
WALL = 1
FORBIDDEN = 2  # ブース配置不可・通行は可(既定)
BOOTH = 3
QUEUE = 4
ENTRANCE = 5
EXIT = 6

#: 通行可能セル種別(§4.2【v2】)。FORBIDDEN は「配置不可だが通行可」を既定とする。
WALKABLE_TYPES = (EMPTY, FORBIDDEN, QUEUE, ENTRANCE, EXIT)

#: 人が通行できる通路の物理的な最小幅。設定値がこれを下回っても、
#: シミュレーションと配置制約では 4 セルを下限として扱う。
MIN_PEDESTRIAN_CORRIDOR_WIDTH_CELLS = 4

# ---------------------------------------------------------------- 来場者状態(§6.2)
NOT_ARRIVED = 0
CHOOSING = 1
WALKING = 2
QUEUING = 3
EXPERIENCING = 4
EXITING = 5
FINISHED = 6
RESTING = 7

#: BFS/A* の到達不能を表す距離値
INF_DIST = np.int32(2**30)


# ================================================================ 設定 dataclass
@dataclass(frozen=True)
class BoothSpec:
    """ブース仕様(§5)。実行時状態は BoothRuntime に分離する。

    kind:
        booth       : 通常の体験ブース(受付・待機列あり)
        kitchen_car : キッチンカー(挙動はブースと同じ。色分け用の属性)
        table       : 方向・専用出入口を持たないテーブル・椅子。周囲4辺から
                      着席して疲労を回復する。capacity は同時着席可能人数。
        waste_station: ゴミステーション。通常時は低確率、飲食後は高確率で
                       立ち寄る付帯設備で、体験ブース数には数えない。
    theme:
        normal / music(音楽祭関連)。色分け用の属性。
    area:
        所属広場(venue_shape.areas の name)。空なら制約なし。
        指定時はブース本体セルがその広場内に収まることが配置制約になる。
    fixed_position:
        最適化前の固定配置パターン。通常ブースは (x,y,theta)、テーブルは (x,y)。
        theta は 0=北, 1=東, 2=南, 3=西(venue.py の THETA_*)。
        最適化実行時は使用されない(座標・向きは最適化変数)。
    """

    booth_id: int
    width: int
    height: int
    category: str
    attractiveness: float  # [0,1] 正規化済み(§8)
    capacity: int
    mean_service_seconds: float
    max_queue_length: int
    name: str = ""
    kind: str = "booth"
    theme: str = "normal"
    area: str = ""
    fixed_position: tuple | None = None

    @property
    def is_table(self) -> bool:
        """休憩・飲食に利用できるテーブル（ベンチ）か。"""
        return self.kind == "table"

    @property
    def is_waste_station(self) -> bool:
        """来場者がごみを捨てるために利用するゴミステーションか。

        新規設定では ``kind: waste_station`` を推奨する。既存の booths.yaml
        との互換性のため、kind/category/theme/name に従来使われていた表記が
        含まれる場合もゴミステーションとして認識する。
        """
        kind = self.kind.strip().casefold()
        if kind in {
            "waste_station",
            "garbage_station",
            "trash_station",
            "recycling_station",
        }:
            return True
        description = " ".join(
            (self.category, self.theme, self.name)
        ).strip().casefold()
        return any(
            marker in description
            for marker in (
                "waste station",
                "waste_station",
                "garbage",
                "trash",
                "recycling",
                "ゴミステーション",
                "ごみステーション",
                "ゴミ捨て",
                "ごみ捨て",
            )
        )


@dataclass(frozen=True)
class ProfileSpec:
    """来場者プロファイル(§6.3)。"""

    name: str
    ratio: float
    speed_mps: float
    congestion_avoidance: float  # λ
    queue_tolerance_seconds: float
    max_visit_seconds: float
    distance_weight_factor: float
    interest_mode: str  # uniform / focused / dispersed
    #: True ならブースに立ち寄らず会場を通り抜けるだけの通行人(駅前広場を想定)。
    passerby: bool = False


@dataclass(frozen=True)
class Config:
    """全設定の一元管理(§26)。YAML から生成し、ワーカーへ pickle 可能。"""

    # 空間(§3.3, §4.1)
    grid_width: int = 20
    grid_height: int = 16
    cell_size_meters: float = 1.0
    # 会場形状(実会場)。venue_areas が空なら従来どおり全面矩形。
    # 各要素は (name, ((x, y), ...)) の組。頂点はメートル単位、y は下向き。
    # venue_areas: ブース配置可能区域(広場)。name は BoothSpec.area から参照される。
    # venue_walkways: 歩行可能・ブース配置不可の区域(道路)。
    #                 グリッド境界に達する道路セルは「領域外への開口部」となり、
    #                 来場者の入退場セル集合として扱う。
    # venue_obstacles: 場内の通行不可障害物(樹木等)。
    venue_areas: tuple = ()
    venue_walkways: tuple = ()
    venue_obstacles: tuple = ()
    # 時間(§3.1, §3.2)
    dt_seconds: float = 1.0
    duration_seconds: float = 400.0
    max_speed_mps: float = 1.0
    # 容量(§4.4)
    cap_default: int = 3
    cap_queue: int = 5
    cap_entrance: int = 10
    cap_exit: int = 10
    # 入口・出口(§4.5)
    entrance_clearance: int = 0
    entrance_side: str = "bottom"
    exit_side: str = "top"
    entrance_exit_span: int = 2
    # 入口・出口セルの明示指定 ((x, y), ...)。省略時は venue_shape があれば
    # 境界開口部(venue.build_base_venue で導出)、なければ辺中央の span を使う。
    entrance_cells_override: tuple = ()
    exit_cells_override: tuple = ()
    # ブース周辺形状
    reception_position: str = "bottom_left"
    booth_exit_position: str = "top_left"
    queue_pattern: str = "bottom_then_right"
    queue_capacity_per_cell: int = 5
    queue_walkable: bool = False
    # シミュレーション(§2.3, §7, §9.3, §12.2, §17.6)
    n_visitors: int = 8000
    evaluation_seed: int = 0
    safe_occupancy_ratio: float = 0.67
    timeout_seconds: float = 5.0
    replan_interval_steps: int = 10
    blocked_replan_count: int = 3
    blocked_congestion_weight: float = 0.00005
    congestion_travel_time_weight: float = 1.0
    # テーブル利用・疲労
    table_rest_seconds: float = 45.0
    table_after_food_probability: float = 0.65
    table_after_categories: tuple = ("food",)
    fatigue_rest_threshold: float = 0.65
    fatigue_walk_per_second: float = 0.0025
    fatigue_wait_per_second: float = 0.0008
    fatigue_experience_per_second: float = 0.0004
    fatigue_recovery_per_second: float = 0.015
    table_occupancy_weight: float = 0.5
    # ゴミステーション利用。通常来場者は低確率、飲食後は高確率で利用する。
    waste_station_general_probability: float = 0.05
    waste_station_after_food_probability: float = 0.85
    waste_station_after_categories: tuple = ("food",)
    # 効用(§8)
    softmax_temperature: float = 0.3
    q_norm_seconds: float = 600.0
    giveup_utility_penalty: float = 0.5
    utility_min_threshold: float = 0.0
    w_interest: float = 1.0
    w_attract: float = 0.5
    w_distance: float = 1.0
    w_queue: float = 0.8
    w_crowd: float = 0.6
    # 到着プロファイル(§7): ((start, end, rate), ...)
    arrival_profile: tuple = ((0.0, 400.0, 1.0),)
    # 制約(§14)
    min_corridor_width: int = MIN_PEDESTRIAN_CORRIDOR_WIDTH_CELLS
    # 幸福度(§13)
    hap_alpha: float = 0.05
    hap_beta: float = 0.1
    hap_gamma: float = 0.3
    hap_rest_bonus: float = 0.15
    hap_fatigue_penalty: float = 0.5
    # 最適化(§15)
    population_size: int = 100
    n_generations: int = 100
    # ブース・プロファイル
    booths: tuple = ()      # tuple[BoothSpec, ...]
    profiles: tuple = ()    # tuple[ProfileSpec, ...]
    categories: tuple = ()  # 興味ベクトルのカテゴリ順

    # ------------------------------------------------------------ 派生値
    @property
    def n_steps(self) -> int:
        """§3.2: シミュレーション総ステップ数。"""
        return int(self.duration_seconds / self.dt_seconds)

    @property
    def n_substeps(self) -> int:
        """§3.4【v2】: 1 ステップ内の固定サブステップ数。"""
        return math.ceil(self.max_speed_mps * self.dt_seconds / self.cell_size_meters)

    @property
    def d_max(self) -> float:
        """§8: 会場対角距離 D_max = sqrt(W^2 + H^2)。"""
        return math.hypot(self.grid_width, self.grid_height)

    @property
    def entrance_cells(self) -> tuple[tuple[int, int], ...]:
        """入口セル。明示指定があればそれを、なければ指定辺の中央を使う。"""
        if self.entrance_cells_override:
            return self.entrance_cells_override
        return self._boundary_cells(self.entrance_side)

    @property
    def exit_cells(self) -> tuple[tuple[int, int], ...]:
        """出口セル。明示指定があればそれを、なければ指定辺の中央を使う。"""
        if self.exit_cells_override:
            return self.exit_cells_override
        return self._boundary_cells(self.exit_side)

    def _boundary_cells(self, side: str) -> tuple[tuple[int, int], ...]:
        """指定辺の中央 ``entrance_exit_span`` セルを (x, y) で返す。"""
        span = self.entrance_exit_span
        if span < 1:
            raise ValueError("entrance_exit.span_cells must be at least 1")
        if side in ("top", "bottom"):
            if span > self.grid_width:
                raise ValueError("entrance_exit.span_cells exceeds grid width")
            start = (self.grid_width - span) // 2
            y = 0 if side == "top" else self.grid_height - 1
            return tuple((x, y) for x in range(start, start + span))
        if side in ("left", "right"):
            if span > self.grid_height:
                raise ValueError("entrance_exit.span_cells exceeds grid height")
            start = (self.grid_height - span) // 2
            x = 0 if side == "left" else self.grid_width - 1
            return tuple((x, y) for y in range(start, start + span))
        raise ValueError(f"unsupported entrance/exit side: {side!r}")

    def category_index(self, category: str) -> int:
        return self.categories.index(category)


def load_config(config_dir: str | Path) -> Config:
    """config ディレクトリの YAML 3 ファイルから Config を構築する。"""
    d = Path(config_dir)
    main = yaml.safe_load((d / "default_config.yaml").read_text(encoding="utf-8"))
    booth_raw = yaml.safe_load((d / "booths.yaml").read_text(encoding="utf-8"))["booths"]
    prof_raw = yaml.safe_load((d / "visitor_profiles.yaml").read_text(encoding="utf-8"))["profiles"]

    def _fixed_position(item) -> tuple | None:
        raw = item.get("fixed_position")
        if raw is None:
            return None
        values = tuple(int(value) for value in raw)
        if str(item.get("kind", "booth")) == "table":
            if len(values) != 2:
                raise ValueError("table fixed_position must be [x, y]")
            return values + (0,)
        if len(values) != 3:
            raise ValueError("booth fixed_position must be [x, y, theta]")
        return values

    def _is_waste_station_item(item: dict) -> bool:
        kind = str(item.get("kind", "booth")).strip().casefold()
        if kind in {
            "waste_station",
            "garbage_station",
            "trash_station",
            "recycling_station",
        }:
            return True
        description = " ".join(
            str(item.get(key, "")) for key in ("category", "theme", "name")
        ).casefold()
        return any(
            marker in description
            for marker in (
                "waste station",
                "waste_station",
                "garbage",
                "trash",
                "recycling",
                "ゴミステーション",
                "ごみステーション",
                "ゴミ捨て",
                "ごみ捨て",
            )
        )

    booths = tuple(
        BoothSpec(
            booth_id=int(b["booth_id"]),
            width=int(b["width"]),
            height=int(b["height"]),
            category=str(b.get("category", "none")),
            attractiveness=float(b.get("attractiveness", 0.0)),
            capacity=(
                max(1, int(b.get("capacity", 1)))
                if _is_waste_station_item(b)
                else int(
                    b.get(
                        "capacity",
                        4 if str(b.get("kind", "booth")) == "table" else 0,
                    )
                )
            ),
            mean_service_seconds=float(b.get("mean_service_seconds", 0.0)),
            max_queue_length=int(b.get("max_queue_length", 0)),
            name=str(b.get("name", "")),
            kind=str(b.get("kind", "booth")),
            theme=str(b.get("theme", "normal")),
            area=str(b.get("area", "")),
            fixed_position=_fixed_position(b),
        )
        for b in booth_raw
    )
    profiles = tuple(
        ProfileSpec(
            name=str(p["name"]),
            ratio=float(p["ratio"]),
            speed_mps=float(p["speed_mps"]),
            congestion_avoidance=float(p["congestion_avoidance"]),
            queue_tolerance_seconds=float(p["queue_tolerance_seconds"]),
            max_visit_seconds=float(p["max_visit_seconds"]),
            distance_weight_factor=float(p["distance_weight_factor"]),
            interest_mode=str(p["interest_mode"]),
            passerby=bool(p.get("passerby", False)),  # 未記載なら通常の来場者
        )
        for p in prof_raw
    )
    # カテゴリ順は booths.yaml の出現順(重複除去)。テーブルと
    # ゴミステーションは効用で選ぶ体験ブースではないため対象外。
    categories: list[str] = []
    for b in booths:
        if not b.is_table and not b.is_waste_station and b.category not in categories:
            categories.append(b.category)

    shape = main.get("venue_shape") or {}

    def _polygons(key: str) -> tuple:
        return tuple(
            (
                str(item.get("name", f"{key}_{i}")),
                tuple((float(x), float(y)) for x, y in item["polygon"]),
            )
            for i, item in enumerate(shape.get(key) or [])
        )

    def _explicit_cells(key: str) -> tuple:
        return tuple((int(x), int(y)) for x, y in (main["entrance_exit"].get(key) or []))

    uw = main["utility_weights"]
    resting = main.get("resting") or {}
    waste_disposal = main.get("waste_disposal") or {}
    happiness_cfg = main.get("happiness") or {}
    cfg = Config(
        grid_width=int(main["grid"]["width"]),
        grid_height=int(main["grid"]["height"]),
        cell_size_meters=float(main["grid"]["cell_size_meters"]),
        venue_areas=_polygons("areas"),
        venue_walkways=_polygons("walkways"),
        venue_obstacles=_polygons("obstacles"),
        entrance_cells_override=_explicit_cells("entrance_cells"),
        exit_cells_override=_explicit_cells("exit_cells"),
        dt_seconds=float(main["time"]["dt_seconds"]),
        duration_seconds=float(main["time"]["duration_seconds"]),
        max_speed_mps=float(main["time"]["max_speed_mps"]),
        cap_default=int(main["capacities"]["default"]),
        cap_queue=int(main["capacities"]["queue"]),
        cap_entrance=int(main["capacities"]["entrance"]),
        cap_exit=int(main["capacities"]["exit"]),
        entrance_clearance=int(main["entrance_exit"]["clearance_cells"]),
        entrance_side=str(main["entrance_exit"].get("entrance_side", "left")),
        exit_side=str(main["entrance_exit"].get("exit_side", "right")),
        entrance_exit_span=int(main["entrance_exit"].get("span_cells", 3)),
        reception_position=str(
            main.get("booth_geometry", {}).get("reception_position", "bottom_center")
        ),
        booth_exit_position=str(
            main.get("booth_geometry", {}).get("exit_position", "reception")
        ),
        queue_pattern=str(
            main.get("booth_geometry", {}).get("queue_pattern", "straight_down")
        ),
        queue_capacity_per_cell=int(
            main.get("booth_geometry", {}).get("queue_capacity_per_cell", 1)
        ),
        queue_walkable=bool(
            main.get("booth_geometry", {}).get("queue_walkable", True)
        ),
        n_visitors=int(main["simulation"]["n_visitors"]),
        evaluation_seed=int(main["simulation"]["evaluation_seed"]),
        safe_occupancy_ratio=float(main["simulation"]["safe_occupancy_ratio"]),
        timeout_seconds=float(main["simulation"]["timeout_seconds"]),
        replan_interval_steps=int(main["simulation"]["replan_interval_steps"]),
        blocked_replan_count=int(main["simulation"]["blocked_replan_count"]),
        blocked_congestion_weight=float(main["simulation"].get("blocked_congestion_weight", 0.00005)),
        congestion_travel_time_weight=float(
            main["simulation"].get("congestion_travel_time_weight", 1.0)
        ),
        table_rest_seconds=float(resting.get("rest_seconds", 45.0)),
        table_after_food_probability=float(
            resting.get("after_food_probability", 0.65)
        ),
        table_after_categories=tuple(
            str(value) for value in resting.get("after_categories", ["food"])
        ),
        fatigue_rest_threshold=float(resting.get("fatigue_threshold", 0.65)),
        fatigue_walk_per_second=float(resting.get("fatigue_walk_per_second", 0.0025)),
        fatigue_wait_per_second=float(resting.get("fatigue_wait_per_second", 0.0008)),
        fatigue_experience_per_second=float(
            resting.get("fatigue_experience_per_second", 0.0004)
        ),
        fatigue_recovery_per_second=float(
            resting.get("fatigue_recovery_per_second", 0.015)
        ),
        table_occupancy_weight=float(resting.get("occupancy_weight", 0.5)),
        waste_station_general_probability=float(
            waste_disposal.get("general_probability", 0.05)
        ),
        waste_station_after_food_probability=float(
            waste_disposal.get("after_food_probability", 0.85)
        ),
        waste_station_after_categories=tuple(
            str(value)
            for value in waste_disposal.get(
                "after_categories",
                resting.get("after_categories", ["food"]),
            )
        ),
        softmax_temperature=float(main["simulation"]["softmax_temperature"]),
        q_norm_seconds=float(main["simulation"]["q_norm_seconds"]),
        giveup_utility_penalty=float(main["simulation"]["giveup_utility_penalty"]),
        utility_min_threshold=float(main["simulation"]["utility_min_threshold"]),
        w_interest=float(uw["interest"]),
        w_attract=float(uw["attract"]),
        w_distance=float(uw["distance"]),
        w_queue=float(uw["queue"]),
        w_crowd=float(uw["crowd"]),
        arrival_profile=tuple(tuple(float(v) for v in row) for row in main["arrival_profile"]),
        min_corridor_width=max(
            MIN_PEDESTRIAN_CORRIDOR_WIDTH_CELLS,
            int(
                (main.get("constraints") or {}).get(
                    "min_corridor_width",
                    MIN_PEDESTRIAN_CORRIDOR_WIDTH_CELLS,
                )
            ),
        ),
        hap_alpha=float(happiness_cfg["alpha"]),
        hap_beta=float(happiness_cfg["beta"]),
        hap_gamma=float(happiness_cfg["gamma"]),
        hap_rest_bonus=float(happiness_cfg.get("rest_bonus", 0.15)),
        hap_fatigue_penalty=float(happiness_cfg.get("fatigue_penalty", 0.5)),
        population_size=int(main["optimization"]["population_size"]),
        n_generations=int(main["optimization"]["n_generations"]),
        booths=booths,
        profiles=profiles,
        categories=tuple(categories),
    )
    if cfg.queue_capacity_per_cell < 1:
        raise ValueError("booth_geometry.queue_capacity_per_cell must be at least 1")
    if cfg.congestion_travel_time_weight < 0.0:
        raise ValueError("simulation.congestion_travel_time_weight must be non-negative")
    if not 0.0 <= cfg.table_after_food_probability <= 1.0:
        raise ValueError("resting.after_food_probability must be in [0, 1]")
    if not 0.0 <= cfg.waste_station_general_probability < 1.0:
        raise ValueError("waste_disposal.general_probability must be in [0, 1)")
    if not 0.0 <= cfg.waste_station_after_food_probability < 1.0:
        raise ValueError("waste_disposal.after_food_probability must be in [0, 1)")
    if (
        cfg.waste_station_after_food_probability
        <= cfg.waste_station_general_probability
    ):
        raise ValueError(
            "waste_disposal.after_food_probability must exceed general_probability"
        )
    if cfg.table_rest_seconds <= 0.0:
        raise ValueError("resting.rest_seconds must be positive")
    if not 0.0 <= cfg.fatigue_rest_threshold <= 1.0:
        raise ValueError("resting.fatigue_threshold must be in [0, 1]")
    if any(b.is_table and b.capacity < 1 for b in cfg.booths):
        raise ValueError("table capacity must be at least 1")
    if any(b.is_waste_station and b.capacity < 1 for b in cfg.booths):
        raise ValueError("waste station capacity must be at least 1")
    return cfg


# ================================================================ 実行時 dataclass
@dataclass
class BoothRuntime:
    """ブース実行時状態(§5)。"""

    booth_id: int
    queue: deque = field(default_factory=deque)        # 待機中 visitor_id(先頭=受付側)
    active_users: dict = field(default_factory=dict)   # visitor_id -> 退場予定ステップ
    exit_waiting: deque = field(default_factory=deque)  # 体験終了・受付セル空き待ち(§11.4)


@dataclass
class Visitor:
    """来場者エージェント(§6.1)。"""

    visitor_id: int
    spawn_time_step: int
    position: tuple | None  # (x, y)。体験中・入場前・退出後は None
    speed_mps: float
    congestion_avoidance: float          # λ(§9.2)
    queue_tolerance_seconds: float
    max_visit_seconds: float
    interests: np.ndarray                # カテゴリ別興味 [0,1]
    distance_weight_factor: float
    profile_name: str
    table_after_food_propensity: float = 1.0
    general_waste_propensity: float = 1.0
    food_waste_propensity: float = 1.0
    #: 素通り客(ブースへ行かず別の開口部へ抜ける通行人)。
    is_passerby: bool = False
    #: 素通り客が目指す開口部(連結成分)の番号。None なら最寄りの開口部から退出する。
    exit_goal_group: int | None = None
    state: int = NOT_ARRIVED
    target_booth_id: int | None = None
    path: list = field(default_factory=list)   # list[tuple[int,int]](現在セルは含まない)
    path_index: int = 0
    move_credit: float = 0.0             # セル進入に使う移動時間残高
    blocked_count: int = 0               # 連続移動失敗回数【v2】
    visited_booths: set = field(default_factory=set)   # 体験済み【v2】
    given_up_booths: set = field(default_factory=set)  # 諦め済み(効用減点対象)
    # --- 集計用(補助KPI) ---
    moved_cells: int = 0                 # 実際に移動したセル数【v2】
    blocked_seconds: float = 0.0         # 移動できず待機した時間【v2】
    queue_seconds: float = 0.0
    give_up_count: int = 0
    entry_delay_seconds: float = 0.0     # 入場遅延(§4.5【v2】)
    current_queue_wait: float = 0.0      # 現在の待ち行列での待機秒数
    steps_since_plan: int = 0
    actual_spawn_step: int = -1          # 実際に入場したステップ(遅延込み)
    # --- 疲労・テーブル利用 ---
    fatigue: float = 0.0                 # 0=疲労なし, 1=最大疲労
    peak_fatigue: float = 0.0
    table_requested_after_food: bool = False
    table_requests_after_food: int = 0
    table_visit_count: int = 0
    food_table_visit_count: int = 0
    table_seconds: float = 0.0
    # --- ゴミステーション利用 ---
    general_waste_decided: bool = False
    waste_station_requested: bool = False
    waste_requested_after_food: bool = False
    waste_station_request_count: int = 0
    waste_station_visit_count: int = 0
    waste_station_give_up_count: int = 0
    food_waste_opportunity_count: int = 0
    food_waste_request_count: int = 0
    food_waste_visit_count: int = 0