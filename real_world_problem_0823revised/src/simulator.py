"""simulator.py — メインループ・サブステップ移動・一括競合解決・ブース体験。

仕様書 v2 の §3(離散化), §8(目的地選択), §9(経路), §10(競合解決),
§11(体験・諦め), §12(目的関数の材料収集)を実装する。

1 ステップの処理順序:
    1. 入場処理(スポーン遅延込み, §4.5)
    2. ブース更新(体験完了→受付セル再出現→キュー先頭入場, §11)
    3. 待ち行列の待機時間更新と諦め判定(§11.3)
    4. 意思決定(CHOOSING の目的地選択+経路計画, §8)
    5. 移動(固定 3 サブステップ×一括競合解決, §3.4/§10)+到着処理
    6. ステップ末の占有スナップショットで混雑を集計(§12.2)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .models import (
    CHOOSING,
    EXITING,
    EXPERIENCING,
    FINISHED,
    NOT_ARRIVED,
    QUEUING,
    RESTING,
    WALKING,
    EXIT,
    INF_DIST,
    BoothRuntime,
    Config,
    Visitor,
)
from .rng import decision_rng, softmax_choice, tiebreak_rng
from .routing import astar, bfs_distance_field, descend_path_indices
from .venue import (
    Layout,
    Venue,
    booth_cells_of,
)
from .visitors import instantiate_visitors


def _connected_cell_groups(cells) -> tuple:
    """セル集合を 4 近傍で連結成分に分ける(開口部を「1 つの出入口」単位にする)。

    戻り値は frozenset のタプル。走査順は入力順に依存させないため、
    各成分の最小セル座標でソートして決定的にする。
    """
    remaining = set(tuple(int(v) for v in c) for c in cells)
    groups: list[frozenset] = []
    while remaining:
        seed = min(remaining)
        remaining.discard(seed)
        component = {seed}
        stack = [seed]
        while stack:
            x, y = stack.pop()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) in remaining:
                    remaining.discard((nx, ny))
                    component.add((nx, ny))
                    stack.append((nx, ny))
        groups.append(frozenset(component))
    groups.sort(key=min)
    return tuple(groups)


@dataclass
class SimResult:
    """1 配置評価の結果。"""

    f_move: float
    f_congestion: float
    kpis: dict
    timed_out: bool = False
    heatmap: np.ndarray | None = None            # 累積滞在人数
    over_threshold_map: np.ndarray | None = None  # 混雑閾値超過時間(秒)
    frames: list = field(default_factory=list)    # 可視化用スナップショット
    
    # 以下は，混雑度と移動距離の推移をプロットするための履歴データ
    time_steps: list[int] = field(default_factory=list)
    congestion_history: list[float] = field(default_factory=list)
    movement_history: list[float] = field(default_factory=list)


class Simulation:
    """1 配置・1 シミュレーション(§2.3)。"""

    def __init__(
        self,
        cfg: Config,
        venue: Venue,
        layout: Layout,
        visitor_records: tuple,
        collect_heatmap: bool = False,
        collect_time_series: bool = True,
        record_frames: bool = False,
        frame_interval: int = 1,
    ) -> None:
        self.cfg = cfg
        self.venue = venue
        self.layout = layout
        self.collect_heatmap = collect_heatmap
        self.collect_time_series = collect_time_series
        self.record_frames = record_frames
        if frame_interval < 1:
            raise ValueError("frame_interval must be at least 1")
        self.frame_interval = frame_interval

        H, W = cfg.grid_height, cfg.grid_width
        self.walkable = layout.walkable
        self.capacity = layout.capacity.astype(np.int16)
        self.occupancy = np.zeros((H, W), dtype=np.int16)
        # 占有率計算用の1次元ビュー。occupancyはin-place更新されるため、
        # このビューにも常に最新値が反映される。
        self._occupancy_flat = self.occupancy.reshape(-1)
        self._capacity_flat_float = self.capacity.reshape(-1).astype(np.float64)
        self._crowd_path_cache: dict[
            tuple[int, tuple[int, int]], np.ndarray
        ] = {}

        # §9.4 静的距離場(配置ごとに 1 回、体験ブース数+1 回の BFS)。
        # 体験ブースは受付、テーブルは周囲4辺への静的距離場を持つ。
        self.booth_fields = []
        for b, spec in enumerate(cfg.booths):
            sources = (
                layout.table_access_cells[b]
                if spec.is_table
                else ((layout.receptions[b],) if layout.receptions[b] is not None else ())
            )
            self.booth_fields.append(
                bfs_distance_field(self.walkable, sources) if sources else None
            )
        self.exit_field = bfs_distance_field(self.walkable, venue.exit_cells)
        self.exit_set = set(venue.exit_cells)
        # 開口部(出入口)を連結成分ごとにまとめる。素通り客は「入ってきた
        # 開口部とは別の開口部」を目的地にするため、その候補集合として使う。
        self.exit_groups = _connected_cell_groups(venue.exit_cells)
        self._exit_group_of = {
            cell: index
            for index, group in enumerate(self.exit_groups)
            for cell in group
        }
        # 開口部ごとの静的距離場は、素通り客が実際に向かった分だけ遅延生成する。
        self._exit_group_fields: dict[int, np.ndarray] = {}

        # 来場者(テーブルから毎回新規生成)・ブース実行時状態
        self.visitors: list[Visitor] = instantiate_visitors(visitor_records)
        self._visitor_by_id = {v.visitor_id: v for v in self.visitors}
        self._visitor_index_by_id = {
            v.visitor_id: index for index, v in enumerate(self.visitors)
        }
        if len(self._visitor_by_id) != len(self.visitors):
            raise ValueError("visitor_id must be unique")
        # Pythonで全来場者を毎ステップ走査しないための状態別インデックス。
        # np.flatnonzeroは元のself.visitors順を保つので、意思決定・乱数消費順は
        # 従来実装と同一になる。
        self._state_masks = {
            state: np.zeros(len(self.visitors), dtype=bool)
            for state in (
                NOT_ARRIVED,
                CHOOSING,
                WALKING,
                QUEUING,
                EXPERIENCING,
                EXITING,
                FINISHED,
                RESTING,
            )
        }
        for index, visitor in enumerate(self.visitors):
            self._state_masks[int(visitor.state)][index] = True
        self._moving_mask = (
            self._state_masks[WALKING] | self._state_masks[EXITING]
        )
        self._active_mask = ~(
            self._state_masks[NOT_ARRIVED] | self._state_masks[FINISHED]
        )
        self._spawned_mask = ~self._state_masks[NOT_ARRIVED].copy()
        self._finished_count = int(np.count_nonzero(self._state_masks[FINISHED]))
        self.booths = [BoothRuntime(booth_id=b.booth_id) for b in cfg.booths]
        self.service_steps = [
            max(1, round(b.mean_service_seconds / cfg.dt_seconds)) for b in cfg.booths
        ]
        self.table_rest_steps = max(1, round(cfg.table_rest_seconds / cfg.dt_seconds))

        # 乱数(§2.3.2): 系統2 は来場者ごとに遅延生成、系統3 はタイブレーク専用
        self._decision_rngs: dict[int, np.random.Generator] = {}
        self._tiebreak = tiebreak_rng(cfg.evaluation_seed)

        # スポーン管理(spawn_time_step 昇順前提)
        self._pending = list(self.visitors)  # 未入場(先頭から順に処理)
        self._pending_idx = 0
        self._delayed: list[Visitor] = []    # 入口満杯で繰り延べ中(§4.5)

        # 集計
        self.congestion_sum = 0.0
        self.weighted_congestion_sum = 0.0
        self.max_occ_ratio = 0.0
        self.n_walkable = int(np.sum(self.walkable))
        self.congestion_weights = self._make_congestion_weights()
        self.congestion_weight_sum = float(np.sum(self.congestion_weights[self.walkable]))
        # 毎ステップ不変な混雑集計用配列は初期化時に一度だけ切り出す。
        self._walkable_flat_indices = np.flatnonzero(self.walkable.reshape(-1))
        self._walkable_capacity = self._capacity_flat_float[
            self._walkable_flat_indices
        ]
        self._walkable_weights = self.congestion_weights.reshape(-1)[
            self._walkable_flat_indices
        ]
        self._walkable_congestion_denom = np.maximum(
            1.0,
            self._walkable_capacity - 1.0,
        )
        self.heatmap = np.zeros((H, W), dtype=np.int64) if collect_heatmap else None
        self.over_map = np.zeros((H, W), dtype=np.float64) if collect_heatmap else None
        self.frames: list = []
        self.time_steps: list[int] = [] # 時系列グラフ用
        self.congestion_history: list[float] = [] # 時系列グラフ用
        self.movement_history: list[float] = [] # 時系列グラフ用
        self.experiences_completed = 0
        self.executed_steps = 0

    def _visitor(self, vid: int):
        """visitor_id からエージェントをO(1)で引く。"""
        try:
            return self._visitor_by_id[vid]
        except KeyError:
            raise KeyError(vid) from None

    def _set_state(self, visitor: Visitor, state: int) -> None:
        """Visitor.stateと状態別インデックスを一括更新する。"""

        previous = int(visitor.state)
        state = int(state)
        if previous == state:
            return
        index = self._visitor_index_by_id[visitor.visitor_id]
        self._state_masks[previous][index] = False
        self._state_masks[state][index] = True
        self._moving_mask[index] = state in (WALKING, EXITING)
        self._active_mask[index] = state not in (NOT_ARRIVED, FINISHED)
        if state != NOT_ARRIVED:
            self._spawned_mask[index] = True
        if previous == FINISHED:
            self._finished_count -= 1
        if state == FINISHED:
            self._finished_count += 1
        visitor.state = state

    def _visitors_from_mask(self, mask: np.ndarray) -> list[Visitor]:
        """マスク対象を、従来と同じself.visitors順のスナップショットで返す。"""

        return [
            self.visitors[int(index)]
            for index in np.flatnonzero(mask)
        ]

    def _make_congestion_weights(self) -> np.ndarray:
        """ブース周辺セルだけを混雑度計算の対象にするマスクを作る。

        重みは歩行可能セルに対してだけ最終的に使う。
        - ブース関連セルからマンハッタン距離 target_radius 以内: 1.0
        - それ以外: 0.0

        ここでいうブース関連セルは、ブース本体・受付セル・ブース出口セル・
        待機列セルを含む。これにより、f_congestion は会場全体の平均混雑度
        ではなく「ブース周辺混雑度」として計算される。
        """
        cfg = self.cfg
        H, W = cfg.grid_height, cfg.grid_width

        # 0.0 = 評価対象外、1.0 = 評価対象
        weights = np.zeros((H, W), dtype=np.float64)

        booth_related: set[tuple[int, int]] = set()

        def add_cell(cell) -> None:
            if cell is None:
                return
            x, y = int(cell[0]), int(cell[1])
            if 0 <= x < W and 0 <= y < H:
                booth_related.add((x, y))

        # 受付セル・ブース出口セル・待機列セルをブース周辺の代表点として使う。
        for cell in getattr(self.layout, "receptions", []):
            add_cell(cell)
        for cell in getattr(self.layout, "booth_exits", []):
            add_cell(cell)
        for qcells in getattr(self.layout, "queue_cells", []):
            for cell in qcells:
                add_cell(cell)

        # 角度を反映したブース本体セルも代表点に含める。
        for b, pos in enumerate(
            getattr(
                self.layout,
                "positions",
                [],
            )
        ):
            if b >= len(cfg.booths):
                break

            bx, by, theta = (
                int(v)
                for v in pos
            )

            for x, y in booth_cells_of(
                cfg,
                b,
                bx,
                by,
                theta,
            ):
                if (
                    0 <= x < W
                    and 0 <= y < H
                ):
                    booth_related.add(
                        (x, y)
                    )

        if not booth_related:
            return weights

        # ブース周辺と見なす範囲。まずは半径3セルを推奨。
        target_radius = 3
        # 各セルから全ブース関連セルまでの距離を総当たりで求める代わりに、
        # 関連セルマスクをマンハッタン半径分だけ平行移動してORする。
        # 障害物を無視したマンハッタン距離という元の定義と厳密に同値。
        related = np.zeros((H, W), dtype=bool)
        for x, y in booth_related:
            related[y, x] = True

        near = np.zeros((H, W), dtype=bool)
        for dy in range(-target_radius, target_radius + 1):
            max_dx = target_radius - abs(dy)
            for dx in range(-max_dx, max_dx + 1):
                src_y0 = max(0, -dy)
                src_y1 = min(H, H - dy)
                src_x0 = max(0, -dx)
                src_x1 = min(W, W - dx)
                dst_y0 = src_y0 + dy
                dst_y1 = src_y1 + dy
                dst_x0 = src_x0 + dx
                dst_x1 = src_x1 + dx
                near[dst_y0:dst_y1, dst_x0:dst_x1] |= related[
                    src_y0:src_y1, src_x0:src_x1
                ]

        weights[near & self.walkable] = 1.0

        return weights

    # ------------------------------------------------------------ 乱数
    def _rng_of(self, vid: int) -> np.random.Generator:
        rng = self._decision_rngs.get(vid)
        if rng is None:
            rng = decision_rng(self.cfg.evaluation_seed, vid)
            self._decision_rngs[vid] = rng
        return rng

    # ------------------------------------------------------------ メインループ
    def run(self) -> SimResult:
        cfg = self.cfg
        start_wall = time.perf_counter()
        timed_out = False
        for t in range(cfg.n_steps):
            self.t = t
            self._spawn_step(t)
            self._booth_step(t)
            self._queue_step(t)
            self._decision_step(t)
            self._movement_step(t)
            self._fatigue_step()
            self._metrics_step(t)
            self.executed_steps = t + 1

            if self.record_frames and t % self.frame_interval == 0:
                self._capture_frame(t)

            # §17.6 早期終了・タイムアウト
            if t % 50 == 0 and time.perf_counter() - start_wall > cfg.timeout_seconds:
                timed_out = True
                break
            if (
                self._pending_idx >= len(self._pending)
                and not self._delayed
                and self._finished_count == len(self.visitors)
            ):
                break
        return self._finalize(timed_out)

    # ------------------------------------------------------------ 1. 入場(§4.5)
    def _spawn_step(self, t: int) -> None:
        due: list[Visitor] = []
        # 繰り延べ中を先に(FIFO)、その後に新規到着分
        for v in self._delayed:
            v.entry_delay_seconds += self.cfg.dt_seconds
            due.append(v)
        self._delayed = []
        while self._pending_idx < len(self._pending) and self._pending[
            self._pending_idx
        ].spawn_time_step <= t:
            due.append(self._pending[self._pending_idx])
            self._pending_idx += 1

        for v in due:
            cell = self._free_entrance_cell()
            if cell is None:
                self._delayed.append(v)  # 次ステップ以降へ繰り延べ
                continue
            x, y = cell
            self.occupancy[y, x] += 1
            v.position = cell
            self._set_state(v, CHOOSING)
            v.actual_spawn_step = t
            if v.is_passerby:
                # 素通り客は入場した時点で反対側の開口部へ歩き出す(§8 の
                # 目的地選択は行わない)。
                self._start_passing_through(v)

    def _free_entrance_cell(self) -> tuple | None:
        candidates = [
            (x, y)
            for (x, y) in self.venue.entrance_cells
            if self.occupancy[y, x] < self.capacity[y, x]
        ]
        if not candidates:
            return None
        idx = int(self._tiebreak.integers(len(candidates)))
        return candidates[idx]

    # ------------------------------------------------------------ 2. ブース更新(§11)
    def _booth_step(self, t: int) -> None:
        for rt in self.booths:
            spec = self.cfg.booths[rt.booth_id]
            if spec.is_table:
                self._table_step(rt, t)
                continue
            ex, ey = self.layout.booth_exits[rt.booth_id]
            separate_exit = self.layout.booth_exits[rt.booth_id] != self.layout.receptions[rt.booth_id]

            # 体験完了 → 退出待ち(入場順=挿入順で決定的)
            done = [vid for vid, end in rt.active_users.items() if end <= t]
            for vid in done:
                del rt.active_users[vid]
                rt.exit_waiting.append(vid)

            # 受付セルへ再出現(§11.4)。空きがなければブース内で待機
            while rt.exit_waiting and self.occupancy[ey, ex] < self.capacity[ey, ex]:
                vid = rt.exit_waiting.popleft()
                v = self._visitor(vid)
                self.occupancy[ey, ex] += 1
                v.position = (ex, ey)
                if spec.is_waste_station:
                    self._complete_waste_station_visit(v)
                else:
                    v.visited_booths.add(rt.booth_id)
                    self._schedule_post_experience_actions(v, spec)
                # max_visit 超過時は意思決定で EXITING へ(§6.2)
                self._set_state(v, CHOOSING)

            # 退出待ちがいない場合のみキュー先頭を入場(§11.4)
            while (
                (separate_exit or not rt.exit_waiting)
                and len(rt.active_users) < spec.capacity
                and rt.queue
            ):
                vid = rt.queue.popleft()
                v = self._visitor(vid)
                qx, qy = v.position
                self.occupancy[qy, qx] -= 1
                v.position = None
                self._set_state(v, EXPERIENCING)
                rt.active_users[vid] = t + self.service_steps[rt.booth_id]
                self._shift_queue(rt)

    def _schedule_post_experience_actions(self, v: Visitor, spec) -> None:
        """体験後のテーブル・ゴミステーション利用を予約する。

        飲食後はまずテーブル利用を判定し、ゴミ廃棄の判定は独立に行う。
        両方を希望した場合、意思決定側が必ずテーブルを先に選ぶため、
        ``飲食→テーブル→ゴミステーション`` の順になる。
        """
        cfg = self.cfg
        if spec.category in cfg.table_after_categories:
            wants_table = (
                v.table_after_food_propensity
                < cfg.table_after_food_probability
            )
            v.table_requested_after_food = wants_table
            if wants_table:
                v.table_requests_after_food += 1

        if spec.category in cfg.waste_station_after_categories:
            # 飲食利用者には通常利用の低確率判定を重ねず、高確率の専用判定を使う。
            v.general_waste_decided = True
            v.food_waste_opportunity_count += 1
            wants_waste_station = (
                v.food_waste_propensity
                < cfg.waste_station_after_food_probability
            )
            if wants_waste_station:
                v.waste_station_requested = True
                v.waste_requested_after_food = True
                v.waste_station_request_count += 1
                v.food_waste_request_count += 1
        else:
            self._request_general_waste_station(v)

    def _request_general_waste_station(self, v: Visitor) -> bool:
        """通常来場者について一度だけ低確率の廃棄行動を判定する。"""
        if v.general_waste_decided:
            return v.waste_station_requested
        v.general_waste_decided = True
        if (
            v.general_waste_propensity
            < self.cfg.waste_station_general_probability
        ):
            v.waste_station_requested = True
            v.waste_requested_after_food = False
            v.waste_station_request_count += 1
        return v.waste_station_requested

    @staticmethod
    def _complete_waste_station_visit(v: Visitor) -> None:
        """ゴミステーション利用を記録し、現在の廃棄要求を完了する。"""
        v.waste_station_visit_count += 1
        if v.waste_requested_after_food:
            v.food_waste_visit_count += 1
        v.waste_station_requested = False
        v.waste_requested_after_food = False

    def _table_step(self, rt: BoothRuntime, t: int) -> None:
        """休憩完了者を周囲4辺の空きセルへ戻す。待機列は作らない。"""
        accesses = self.layout.table_access_cells[rt.booth_id]
        done = [vid for vid, end in rt.active_users.items() if end <= t]
        for vid in done:
            del rt.active_users[vid]
            rt.exit_waiting.append(vid)
        while rt.exit_waiting:
            available = [
                (x, y)
                for x, y in accesses
                if self.occupancy[y, x] < self.capacity[y, x]
            ]
            if not available:
                break
            min_occupancy = min(self.occupancy[y, x] for x, y in available)
            best = [
                cell for cell in available
                if self.occupancy[cell[1], cell[0]] == min_occupancy
            ]
            access = best[int(self._tiebreak.integers(len(best)))]
            ax, ay = access
            vid = rt.exit_waiting.popleft()
            v = self._visitor(vid)
            self.occupancy[ay, ax] += 1
            v.position = access
            v.target_booth_id = None
            self._set_state(v, CHOOSING)

    def _shift_queue(self, rt: BoothRuntime) -> None:
        """待機者を先頭(受付側)から詰め直す(§5.2)。"""
        cells = self.layout.queue_cells[rt.booth_id]
        per_cell = self.cfg.queue_capacity_per_cell
        for slot, vid in enumerate(rt.queue):
            v = self._visitor(vid)
            new_cell = cells[slot // per_cell]
            if v.position != new_cell:
                ox, oy = v.position
                nx, ny = new_cell
                self.occupancy[oy, ox] -= 1
                self.occupancy[ny, nx] += 1
                v.position = new_cell

    # ------------------------------------------------------------ 3. 待ち行列(§11.3)
    def _queue_step(self, t: int) -> None:
        dt = self.cfg.dt_seconds
        for rt in self.booths:
            leavers = []
            for vid in rt.queue:
                v = self._visitor(vid)
                v.queue_seconds += dt
                v.current_queue_wait += dt
                if v.current_queue_wait > v.queue_tolerance_seconds:
                    leavers.append(vid)
            for vid in leavers:
                rt.queue.remove(vid)
                v = self._visitor(vid)
                self._give_up_target(v, rt.booth_id)
                # その場(現在の行列セル)から再選択
                self._set_state(v, CHOOSING)
                v.current_queue_wait = 0.0
            if leavers:
                self._shift_queue(rt)

    # ------------------------------------------------------------ 4. 意思決定(§8)
    def _decision_step(self, t: int) -> None:
        cfg = self.cfg
        choosing = self._visitors_from_mask(self._state_masks[CHOOSING])
        for v in choosing:
            if v.position is None:
                continue
            elapsed = (t - v.actual_spawn_step) * cfg.dt_seconds
            if elapsed > v.max_visit_seconds:
                self._start_exiting(v)
                continue
            self._choose_target(v, t)

    def _choose_target(self, v: Visitor, t: int) -> None:
        """効用 + softmax によるブース選択(§8)。候補なしなら EXITING。"""
        cfg = self.cfg
        # 飲食後は「テーブル or ゴミステーション」。両方を利用する人は
        # テーブルを先にし、離席後も残る waste_station_requested に従って
        # ゴミステーションへ向かう。
        if v.table_requested_after_food:
            if self._choose_table(v, t):
                return
            # テーブルが存在しない・全席満席・到達不能なら廃棄行動へ進む。
            v.table_requested_after_food = False
        if v.waste_station_requested and self._choose_waste_station(v, t):
            return
        if (
            v.fatigue >= cfg.fatigue_rest_threshold
            and self._choose_table(v, t)
        ):
            return
        x, y = v.position
        cand_ids: list[int] = []
        utils: list[float] = []
        for b, spec in enumerate(cfg.booths):
            if spec.is_table or spec.is_waste_station or b in v.visited_booths:
                continue  # 付帯設備と体験済みブースは除外
            field_b = self.booth_fields[b]
            if field_b[y, x] >= INF_DIST:
                continue
            u = self._utility(v, b, field_b[y, x])
            if b in v.given_up_booths:
                u -= cfg.giveup_utility_penalty  # 諦めブースは減点して候補に残す
            cand_ids.append(b)
            utils.append(u)

        if not cand_ids or max(utils) < cfg.utility_min_threshold:
            # まだ通常利用判定をしていない来場者は、退出直前にも一度だけ
            # ペットボトル等を捨てる低確率の機会を持つ。
            if (
                self._request_general_waste_station(v)
                and self._choose_waste_station(v, t)
            ):
                return
            self._start_exiting(v)
            return

        u_arr = np.array(utils, dtype=float)
        z = (u_arr - u_arr.max()) / cfg.softmax_temperature
        p = np.exp(z)
        p /= p.sum()
        target = softmax_choice(self._rng_of(v.visitor_id), cand_ids, p)

        v.target_booth_id = target
        goal = self.layout.receptions[target]
        if v.position == goal:
            v.path, v.path_index = [], 0
            self._set_state(v, WALKING)
            self._arrive_at_reception(v, t)
            return
        path = astar(
            self.walkable,
            self.occupancy,
            self.capacity,
            v.position,
            {goal},
            v.congestion_avoidance,
            self.booth_fields[target],
        )
        if path is None:  # 静的距離が有限なら理論上到達可能(安全側フォールバック)
            v.given_up_booths.add(target)
            return
        v.path, v.path_index = path, 0
        v.steps_since_plan = 0
        v.blocked_count = 0
        self._set_state(v, WALKING)

    def _choose_table(self, v: Visitor, t: int) -> bool:
        """空席があり到達可能なテーブルを、距離と満席度から選択する。"""
        cfg = self.cfg
        x, y = v.position
        candidates: list[int] = []
        utilities: list[float] = []
        for b, spec in enumerate(cfg.booths):
            if not spec.is_table:
                continue
            field = self.booth_fields[b]
            if field is None or field[y, x] >= INF_DIST:
                continue
            occupied = len(self.booths[b].active_users)
            if occupied >= spec.capacity:
                continue
            distance = float(field[y, x]) / cfg.d_max
            occupancy_ratio = occupied / spec.capacity
            candidates.append(b)
            utilities.append(-distance - cfg.table_occupancy_weight * occupancy_ratio)
        if not candidates:
            return False

        values = np.asarray(utilities, dtype=float)
        z = (values - values.max()) / cfg.softmax_temperature
        probabilities = np.exp(z)
        probabilities /= probabilities.sum()
        target = softmax_choice(
            self._rng_of(v.visitor_id), candidates, probabilities
        )
        v.target_booth_id = target
        goal_set = set(self.layout.table_access_cells[target])
        if v.position in goal_set:
            v.path, v.path_index = [], 0
            self._set_state(v, WALKING)
            self._arrive_at_reception(v, t)
            return True
        path = astar(
            self.walkable,
            self.occupancy,
            self.capacity,
            v.position,
            goal_set,
            v.congestion_avoidance,
            self.booth_fields[target],
        )
        if path is None:
            return False
        v.path, v.path_index = path, 0
        v.steps_since_plan = 0
        v.blocked_count = 0
        self._set_state(v, WALKING)
        return True

    def _choose_waste_station(self, v: Visitor, t: int) -> bool:
        """到達可能なゴミステーションを距離と待ち時間から選択する。"""
        cfg = self.cfg
        x, y = v.position
        candidates: list[int] = []
        utilities: list[float] = []
        for b, spec in enumerate(cfg.booths):
            if not spec.is_waste_station:
                continue
            field = self.booth_fields[b]
            if field is None or field[y, x] >= INF_DIST:
                continue
            runtime = self.booths[b]
            queue_capacity = (
                len(self.layout.queue_cells[b])
                * cfg.queue_capacity_per_cell
            )
            if (
                len(runtime.active_users) >= spec.capacity
                and len(runtime.queue) >= queue_capacity
            ):
                continue
            distance = float(field[y, x]) / cfg.d_max
            estimated_wait = (
                len(runtime.queue)
                * spec.mean_service_seconds
                / max(spec.capacity, 1)
                / max(cfg.q_norm_seconds, 1.0)
            )
            candidates.append(b)
            utilities.append(-distance - cfg.w_queue * estimated_wait)
        if not candidates:
            return False

        values = np.asarray(utilities, dtype=float)
        z = (values - values.max()) / cfg.softmax_temperature
        probabilities = np.exp(z)
        probabilities /= probabilities.sum()
        target = softmax_choice(
            self._rng_of(v.visitor_id), candidates, probabilities
        )
        v.target_booth_id = target
        goal = self.layout.receptions[target]
        if v.position == goal:
            v.path, v.path_index = [], 0
            self._set_state(v, WALKING)
            self._arrive_at_reception(v, t)
            return True
        path = astar(
            self.walkable,
            self.occupancy,
            self.capacity,
            v.position,
            {goal},
            v.congestion_avoidance,
            self.booth_fields[target],
        )
        if path is None:
            return False
        v.path, v.path_index = path, 0
        v.steps_since_plan = 0
        v.blocked_count = 0
        self._set_state(v, WALKING)
        return True

    def _give_up_target(self, v: Visitor, booth_id: int | None) -> None:
        """通常ブースとゴミステーションの諦めを別々に集計する。"""
        if booth_id is None:
            return
        if self.cfg.booths[booth_id].is_waste_station:
            v.waste_station_give_up_count += 1
            # 捨てずに持ち歩く選択として、この廃棄要求はここで終了する。
            v.waste_station_requested = False
            v.waste_requested_after_food = False
        else:
            v.give_up_count += 1
            v.given_up_booths.add(booth_id)

    def _utility(self, v: Visitor, b: int, static_dist: int) -> float:
        """§8 の正規化済み効用。"""
        cfg = self.cfg
        spec = cfg.booths[b]
        rt = self.booths[b]
        interest = float(v.interests[cfg.category_index(spec.category)])
        d_norm = float(static_dist) / cfg.d_max
        q_hat = len(rt.queue) * spec.mean_service_seconds / spec.capacity
        q_norm = min(1.0, q_hat / cfg.q_norm_seconds)
        c_norm = self._crowd_on_path(v.position, b)
        return (
            cfg.w_interest * interest
            + cfg.w_attract * spec.attractiveness
            - cfg.w_distance * v.distance_weight_factor * d_norm
            - cfg.w_queue * q_norm
            - cfg.w_crowd * c_norm
        )

    def _crowd_on_path(self, pos: tuple, b: int) -> float:
        """C̃: 静的最短経路上の通行可能セルの占有率平均(評価時点スナップショット)。"""
        cache_key = (int(b), (int(pos[0]), int(pos[1])))
        flat_indices = self._crowd_path_cache.get(cache_key)
        if flat_indices is None:
            flat_indices = descend_path_indices(self.booth_fields[b], pos)
            self._crowd_path_cache[cache_key] = flat_indices
        if flat_indices.size == 0:
            return 0.0
        # 経路自体は静的なのでインデックスを再利用し、占有率の加算だけを
        # NumPyで実行する。占有値は毎時点の最新スナップショットを参照する。
        mean_ratio = float(
            np.sum(
                self._occupancy_flat[flat_indices]
                / self._capacity_flat_float[flat_indices]
            )
            / flat_indices.size
        )
        return min(1.0, mean_ratio)

    def _exit_goal_set(self, v: Visitor) -> set:
        """退出先セル集合。素通り客だけ「入口とは別の開口部」に限定される。"""
        if v.exit_goal_group is None:
            return self.exit_set
        return self.exit_groups[v.exit_goal_group]

    def _exit_heuristic(self, v: Visitor) -> np.ndarray:
        """退出先までの静的距離場。A* は距離場が 0 のセルを目標とみなす。"""
        if v.exit_goal_group is None:
            return self.exit_field
        field_g = self._exit_group_fields.get(v.exit_goal_group)
        if field_g is None:
            # 素通り客が実際に使う開口部の分だけ BFS する(通常は数回)。
            field_g = bfs_distance_field(
                self.walkable, tuple(sorted(self.exit_groups[v.exit_goal_group]))
            )
            self._exit_group_fields[v.exit_goal_group] = field_g
        return field_g

    def _start_passing_through(self, v: Visitor) -> None:
        """素通り客の退出先を決めて歩き出させる(§8 の目的地選択は行わない)。

        入場した開口部と同じ開口部を目的地にすると、その場で退出扱いになり
        会場内をまったく歩かないので、必ず別の連結成分の開口部を選ぶ。
        開口部が 1 か所しかない会場では通り抜けようがないため、
        通常の来場者(CHOOSING)として扱う。
        """
        entry_group = self._exit_group_of.get(v.position)
        others = [
            index
            for index in range(len(self.exit_groups))
            if index != entry_group
        ]
        if not others:
            v.is_passerby = False
            self._set_state(v, CHOOSING)
            return
        idx = int(self._rng_of(v.visitor_id).integers(len(others)))
        v.exit_goal_group = others[idx]
        self._start_exiting(v)

    def _start_exiting(self, v: Visitor) -> None:
        self._set_state(v, EXITING)
        v.target_booth_id = None
        goal_set = self._exit_goal_set(v)
        if v.position in goal_set:
            self._finish(v)
            return
        path = astar(
            self.walkable,
            self.occupancy,
            self.capacity,
            v.position,
            goal_set,
            v.congestion_avoidance,
            self._exit_heuristic(v),
        )
        v.path = path if path is not None else []
        v.path_index = 0
        v.steps_since_plan = 0
        v.blocked_count = 0

    def _finish(self, v: Visitor) -> None:
        x, y = v.position
        self.occupancy[y, x] -= 1
        v.position = None
        self._set_state(v, FINISHED)

    # ------------------------------------------------------------ 5. 移動(§3.4/§10)
    def _cell_travel_cost(self, cell: tuple) -> float:
        """混雑率に応じた1セル分の所要時間倍率を返す。

        占有率 r=n/C に対して 1 + w*r^2 とする。通常セルは1セル分、
        混雑セルは1セルより大きい移動残高が必要になるため、実効速度が落ちる。
        満員セルへの進入可否は従来どおり容量競合処理で別途判定する。
        """
        x, y = cell
        capacity = max(float(self.capacity[y, x]), 1.0)
        occupancy_ratio = min(
            1.0,
            max(0.0, float(self.occupancy[y, x]) / capacity),
        )
        return 1.0 + self.cfg.congestion_travel_time_weight * occupancy_ratio**2

    def _movement_step(self, t: int) -> None:
        cfg = self.cfg
        movers = [
            v for v in self._visitors_from_mask(self._moving_mask)
            if v.position is not None
        ]

        # 移動時間残高の加算(§3.4)。混雑セルでは1セルの移動に1より大きい
        # 残高が必要になる。最も遅いセルを通過できる値までは蓄積を許す。
        max_cell_cost = 1.0 + cfg.congestion_travel_time_weight
        credit_cap = max(float(cfg.n_substeps), max_cell_cost)
        for v in movers:
            gain = v.speed_mps * cfg.dt_seconds / cfg.cell_size_meters
            v.move_credit = min(v.move_credit + gain, credit_cap)
            v.steps_since_plan += 1

        # 再計画条件(§9.3)
        for v in movers:
            need = (
                v.blocked_count >= cfg.blocked_replan_count
                or v.steps_since_plan > cfg.replan_interval_steps
                or v.path_index >= len(v.path)
            )
            if need:
                self._replan(v)

        blocked_dt = cfg.dt_seconds / cfg.n_substeps
        for _sub in range(cfg.n_substeps):
            # 1) 移動希望の一括収集
            proposals: dict[tuple, list[Visitor]] = {}
            proposal_costs: dict[int, float] = {}
            for v in movers:
                if (
                    v.state in (WALKING, EXITING)
                    and v.position is not None
                    and v.path_index < len(v.path)
                ):
                    nxt = v.path[v.path_index]
                    travel_cost = self._cell_travel_cost(nxt)
                    if v.move_credit >= travel_cost:
                        proposals.setdefault(nxt, []).append(v)
                        proposal_costs[v.visitor_id] = travel_cost
            if not proposals:
                continue

            # 2-5) 一括競合解決。空きはサブステップ開始時点の占有で判定
            #      (このサブステップで退出する者の空きは次サブステップから有効)
            occ_snapshot = self.occupancy.copy()
            granted: list[Visitor] = []
            for (nx, ny), vs in proposals.items():
                avail = int(self.capacity[ny, nx] - occ_snapshot[ny, nx])
                if avail >= len(vs):
                    ok, ng = vs, []
                else:
                    vs_sorted = sorted(
                        vs,
                        key=lambda a: (
                            -a.blocked_count,  # 優先 1: 連続ブロック回数 大
                            a.max_visit_seconds
                            - (t - a.actual_spawn_step) * cfg.dt_seconds,  # 優先 2: 残り滞在時間 小
                            self._tiebreak.random(),  # 優先 3: 系統3乱数
                        ),
                    )
                    k = max(avail, 0)
                    ok, ng = vs_sorted[:k], vs_sorted[k:]
                for v in ok:
                    granted.append(v)
                for v in ng:  # 6) 拒否
                    v.blocked_count += 1
                    v.blocked_seconds += blocked_dt

            # 7) 許可された移動を実行
            arrivals: list[Visitor] = []
            for v in granted:
                ox, oy = v.position
                nx, ny = v.path[v.path_index]
                self.occupancy[oy, ox] -= 1
                self.occupancy[ny, nx] += 1
                v.position = (nx, ny)
                v.path_index += 1
                v.move_credit -= proposal_costs[v.visitor_id]
                v.moved_cells += 1
                v.blocked_count = 0
                if v.path_index >= len(v.path):
                    arrivals.append(v)

            for v in arrivals:
                self._handle_arrival(v, t)

    def _replan(self, v: Visitor) -> None:
        """§9.3 の再探索。計画時点の占有スナップショットでコスト評価(§9.2)。"""
        if v.state == WALKING and v.target_booth_id is None:  # 防御(通常は発生しない)
            self._set_state(v, CHOOSING)
            v.path, v.path_index = [], 0
            return
        if v.state == EXITING:
            goal_set, hfield = self._exit_goal_set(v), self._exit_heuristic(v)
        else:
            b = v.target_booth_id
            if self.cfg.booths[b].is_table:
                goal_set = set(self.layout.table_access_cells[b])
            else:
                goal_set = {self.layout.receptions[b]}
            hfield = self.booth_fields[b]
        if v.position in goal_set:
            v.path, v.path_index = [], 0
            self._handle_arrival(v, self.t)
            return
        path = astar(
            self.walkable, self.occupancy, self.capacity, v.position, goal_set,
            v.congestion_avoidance, hfield,
        )
        if path is not None:
            v.path, v.path_index = path, 0
            v.steps_since_plan = 0
            v.blocked_count = 0
        elif v.state == WALKING:
            self._give_up_target(v, v.target_booth_id)
            self._set_state(v, CHOOSING)
            v.path, v.path_index = [], 0

    def _handle_arrival(self, v: Visitor, t: int) -> None:
        if v.state == EXITING:
            if v.position in self._exit_goal_set(v):
                self._finish(v)
            return
        if v.state == WALKING and v.target_booth_id is not None:
            target = v.target_booth_id
            if self.cfg.booths[target].is_table:
                arrived = v.position in set(self.layout.table_access_cells[target])
            else:
                arrived = v.position == self.layout.receptions[target]
            if arrived:
                self._arrive_at_reception(v, t)

    # ------------------------------------------------------------ 入場・行列参加(§11.1, §11.3)
    def _arrive_at_reception(self, v: Visitor, t: int) -> None:
        cfg = self.cfg
        b = v.target_booth_id
        spec = cfg.booths[b]
        rt = self.booths[b]
        if spec.is_table:
            self._arrive_at_table(v, t, b)
            return
        rx, ry = self.layout.receptions[b]
        separate_exit = self.layout.booth_exits[b] != self.layout.receptions[b]

        can_enter = (
            (separate_exit or not rt.exit_waiting)
            and len(rt.active_users) < spec.capacity
            and not rt.queue
        )
        if can_enter:
            self.occupancy[ry, rx] -= 1
            v.position = None
            self._set_state(v, EXPERIENCING)
            rt.active_users[v.visitor_id] = t + self.service_steps[b]
            return

        # 行列参加判定(§11.3): 満杯 or 残り時間不足なら諦め
        est_wait = len(rt.queue) * spec.mean_service_seconds / spec.capacity
        remaining = (cfg.n_steps - t) * cfg.dt_seconds
        queue_capacity = len(self.layout.queue_cells[b]) * cfg.queue_capacity_per_cell
        if len(rt.queue) >= queue_capacity or remaining < est_wait + spec.mean_service_seconds:
            self._give_up_target(v, b)
            self._set_state(v, CHOOSING)
            return

        slot = len(rt.queue)
        qx, qy = self.layout.queue_cells[b][slot // cfg.queue_capacity_per_cell]
        self.occupancy[ry, rx] -= 1
        self.occupancy[qy, qx] += 1  # 待機者は実セルを占有(§5.2)
        v.position = (qx, qy)
        self._set_state(v, QUEUING)
        v.current_queue_wait = 0.0
        rt.queue.append(v.visitor_id)

    def _arrive_at_table(self, v: Visitor, t: int, table_id: int) -> None:
        """空席なら着席させる。到着時に満席なら次ステップに再選択する。"""
        spec = self.cfg.booths[table_id]
        rt = self.booths[table_id]
        if len(rt.active_users) >= spec.capacity:
            v.target_booth_id = None
            self._set_state(v, CHOOSING)
            return
        access_x, access_y = v.position
        self.occupancy[access_y, access_x] -= 1
        v.position = None
        self._set_state(v, RESTING)
        rt.active_users[v.visitor_id] = t + self.table_rest_steps
        v.table_visit_count += 1
        if v.table_requested_after_food:
            v.food_table_visit_count += 1
        v.table_requested_after_food = False

    def _fatigue_step(self) -> None:
        """状態に応じて疲労度を更新し、0～1に制限する。"""
        cfg = self.cfg
        dt = cfg.dt_seconds
        active = self._visitors_from_mask(self._active_mask)
        for v in active:
            if v.state in (WALKING, EXITING):
                delta = cfg.fatigue_walk_per_second * dt
            elif v.state in (QUEUING, CHOOSING):
                delta = cfg.fatigue_wait_per_second * dt
            elif v.state == EXPERIENCING:
                delta = cfg.fatigue_experience_per_second * dt
            elif v.state == RESTING:
                delta = -cfg.fatigue_recovery_per_second * dt
                v.table_seconds += dt
            else:
                delta = 0.0
            v.fatigue = float(np.clip(v.fatigue + delta, 0.0, 1.0))
            v.peak_fatigue = max(v.peak_fatigue, v.fatigue)

    # ------------------------------------------------------------ 6. 集計(§12.2)
    def _metrics_step(self, t: int) -> None:
        cfg = self.cfg

        occ = self._occupancy_flat[self._walkable_flat_indices].astype(np.float64)
        cap = self._walkable_capacity
        weights = self._walkable_weights

        # ---------------------------------------------------------
        # 1. タイムステップ t における混雑度
        # ---------------------------------------------------------
        #excess = np.maximum(
        #    0.0,
        #    occ / cap - cfg.safe_occupancy_ratio,
        #)

        denom = self._walkable_congestion_denom  # 1人までは混雑なし、容量いっぱいで1.0
                                 
        congestion_contrib = np.clip((occ - 1.0) / denom, 0.0, 1.0)

        # 従来の非加重混雑度
        step_congestion_sum = float(np.sum(congestion_contrib))

        # ブース周辺セルだけを評価した混雑度
        step_weighted_congestion_sum = float(np.sum(weights * congestion_contrib))

        # 通行可能セル数で正規化した、その時点の混雑度
        if self.n_walkable > 0:
            step_congestion = step_congestion_sum / self.n_walkable
        else:
            step_congestion = 0.0

        # 時系列表示は，目的関数に近いブース周辺混雑度を使う。
        if self.congestion_weight_sum > 0:
            step_weighted_congestion = step_weighted_congestion_sum / self.congestion_weight_sum
        else:
            step_weighted_congestion = step_congestion

        # 最終目的関数計算用の累積値
        self.congestion_sum += step_congestion_sum
        self.weighted_congestion_sum += step_weighted_congestion_sum

        # ---------------------------------------------------------
        # 2. タイムステップ t までの移動負担
        # ---------------------------------------------------------
        if self.collect_time_series:
            spawned = self._visitors_from_mask(self._spawned_mask)

            if spawned:
                movement_burden = float(
                    np.mean([
                        v.moved_cells
                        * cfg.cell_size_meters
                        / v.speed_mps
                        for v in spawned
                    ])
                )
            else:
                movement_burden = 0.0

            # -----------------------------------------------------
            # 3. 可視化を要求された評価だけ時系列として保存
            # -----------------------------------------------------
            self.time_steps.append(t)
            self.congestion_history.append(step_weighted_congestion)
            self.movement_history.append(movement_burden)

        # ---------------------------------------------------------
        # 既存のKPI集計
        # ---------------------------------------------------------
        if occ.size:
            self.max_occ_ratio = max(
                self.max_occ_ratio,
                float(np.max(occ / cap)),
            )

        if self.collect_heatmap:
            self.heatmap += self.occupancy

            over = np.maximum(
                0.0,
                self.occupancy / np.maximum(self.capacity, 1)
                - cfg.safe_occupancy_ratio,
            )

            self.over_map += (
                (over > 0)
                * cfg.dt_seconds
                * self.walkable
            )
        
    def _capture_frame(self, t: int) -> None:
        snap = [
            (
                v.visitor_id,
                v.position[0],
                v.position[1],
                v.state,
                v.target_booth_id,
                v.blocked_count,
            )
            for v in self.visitors
            if v.position is not None
        ]
        booth_stats = tuple(
            (rt.booth_id, len(rt.active_users), len(rt.queue))
            for rt in self.booths
        )
        self.frames.append((t, snap, booth_stats))

    # ------------------------------------------------------------ 終了処理
    def _finalize(self, timed_out: bool) -> SimResult:
        from .metrics import compute_metrics

        # 正常に早期終了した場合は、残り時間を補完する。
        #
        # 来場者が全員退出した後なので、
        #   混雑度   = 0
        #   移動負担 = 最終値のまま
        #
        # とする。
        if (
            self.collect_time_series
            and not timed_out
            and len(self.time_steps) < self.cfg.n_steps
        ):
            start = len(self.time_steps)
            remaining = self.cfg.n_steps - start

            if self.movement_history:
                last_movement = self.movement_history[-1]
            else:
                last_movement = 0.0

            self.time_steps.extend(
                range(start, self.cfg.n_steps)
            )

            self.congestion_history.extend(
                [0.0] * remaining
            )

            self.movement_history.extend(
                [last_movement] * remaining
            )

        f_move, f_cong, kpis = compute_metrics(self)

        return SimResult(
            f_move=f_move,
            f_congestion=f_cong,
            kpis=kpis,
            timed_out=timed_out,
            heatmap=self.heatmap,
            over_threshold_map=self.over_map,
            frames=self.frames,

            # 追加
            time_steps=self.time_steps,
            congestion_history=self.congestion_history,
            movement_history=self.movement_history,
        )