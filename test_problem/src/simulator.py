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
    WALKING,
    EXIT,
    INF_DIST,
    BoothRuntime,
    Config,
    Visitor,
)
from .rng import decision_rng, softmax_choice, tiebreak_rng
from .routing import astar, bfs_distance_field, descend_path
from .venue import Layout, Venue
from .visitors import instantiate_visitors


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
        record_frames: bool = False,
        frame_interval: int = 1,
    ) -> None:
        self.cfg = cfg
        self.venue = venue
        self.layout = layout
        self.collect_heatmap = collect_heatmap
        self.record_frames = record_frames
        if frame_interval < 1:
            raise ValueError("frame_interval must be at least 1")
        self.frame_interval = frame_interval

        H, W = cfg.grid_height, cfg.grid_width
        self.walkable = layout.walkable
        self.capacity = layout.capacity.astype(np.int16)
        self.occupancy = np.zeros((H, W), dtype=np.int16)

        # §9.4 静的距離場(配置ごとに 1 回、6+1 回の BFS)
        self.booth_fields = [
            bfs_distance_field(self.walkable, [layout.receptions[b]])
            for b in range(len(cfg.booths))
        ]
        self.exit_field = bfs_distance_field(self.walkable, venue.exit_cells)
        self.exit_set = set(venue.exit_cells)

        # 来場者(テーブルから毎回新規生成)・ブース実行時状態
        self.visitors: list[Visitor] = instantiate_visitors(visitor_records)
        self.booths = [BoothRuntime(booth_id=b.booth_id) for b in cfg.booths]
        self.service_steps = [
            max(1, round(b.mean_service_seconds / cfg.dt_seconds)) for b in cfg.booths
        ]

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
        self.heatmap = np.zeros((H, W), dtype=np.int64) if collect_heatmap else None
        self.over_map = np.zeros((H, W), dtype=np.float64) if collect_heatmap else None
        self.frames: list = []
        self.time_steps: list[int] = [] # 時系列グラフ用
        self.congestion_history: list[float] = [] # 時系列グラフ用
        self.movement_history: list[float] = [] # 時系列グラフ用
        self.experiences_completed = 0

    def _visitor(self, vid: int):
        """visitor_id からエージェントを引く(テスト等で id が連番でなくても安全)。"""
        v = self.visitors[vid] if vid < len(self.visitors) else None
        if v is not None and v.visitor_id == vid:
            return v
        for v in self.visitors:
            if v.visitor_id == vid:
                return v
        raise KeyError(vid)

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

        # layout.positions がある場合はブース本体セルも代表点に含める。
        # 座標系は本実装内の他処理に合わせて (x, y) として扱う。
        for b, pos in enumerate(getattr(self.layout, "positions", [])):
            if b >= len(cfg.booths):
                break
            spec = cfg.booths[b]
            x0, y0 = int(round(pos[0])), int(round(pos[1]))
            for y in range(y0, y0 + spec.height):
                for x in range(x0, x0 + spec.width):
                    if 0 <= x < W and 0 <= y < H:
                        booth_related.add((x, y))

        if not booth_related:
            return weights

        # ブース周辺と見なす範囲。まずは半径3セルを推奨。
        target_radius = 3
        booth_xy = list(booth_related)

        for y in range(H):
            for x in range(W):
                if not self.walkable[y, x]:
                    continue
                d_min = min(abs(x - bx) + abs(y - by) for bx, by in booth_xy)
                if d_min <= target_radius:
                    weights[y, x] = 1.0

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
            self._metrics_step(t)

            if self.record_frames and t % self.frame_interval == 0:
                self._capture_frame(t)

            # §17.6 早期終了・タイムアウト
            if t % 50 == 0 and time.perf_counter() - start_wall > cfg.timeout_seconds:
                timed_out = True
                break
            if self._pending_idx >= len(self._pending) and not self._delayed and all(
                v.state in (FINISHED,) for v in self.visitors
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
            v.state = CHOOSING
            v.actual_spawn_step = t

    def _free_entrance_cell(self) -> tuple | None:
        for (x, y) in self.venue.entrance_cells:
            if self.occupancy[y, x] < self.capacity[y, x]:
                return (x, y)
        return None

    # ------------------------------------------------------------ 2. ブース更新(§11)
    def _booth_step(self, t: int) -> None:
        for rt in self.booths:
            spec = self.cfg.booths[rt.booth_id]
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
                v.visited_booths.add(rt.booth_id)
                v.state = CHOOSING  # max_visit 超過時は意思決定で EXITING へ(§6.2)

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
                v.state = EXPERIENCING
                rt.active_users[vid] = t + self.service_steps[rt.booth_id]
                self._shift_queue(rt)

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
                v.give_up_count += 1
                v.given_up_booths.add(rt.booth_id)
                v.state = CHOOSING  # その場(現在の行列セル)から再選択
                v.current_queue_wait = 0.0
            if leavers:
                self._shift_queue(rt)

    # ------------------------------------------------------------ 4. 意思決定(§8)
    def _decision_step(self, t: int) -> None:
        cfg = self.cfg
        for v in self.visitors:
            if v.state != CHOOSING or v.position is None:
                continue
            elapsed = (t - v.actual_spawn_step) * cfg.dt_seconds
            if elapsed > v.max_visit_seconds:
                self._start_exiting(v)
                continue
            self._choose_target(v, t)

    def _choose_target(self, v: Visitor, t: int) -> None:
        """効用 + softmax によるブース選択(§8)。候補なしなら EXITING。"""
        cfg = self.cfg
        x, y = v.position
        cand_ids: list[int] = []
        utils: list[float] = []
        for b, spec in enumerate(cfg.booths):
            if b in v.visited_booths:
                continue  # 体験済みは除外
            field_b = self.booth_fields[b]
            if field_b[y, x] >= INF_DIST:
                continue
            u = self._utility(v, b, field_b[y, x])
            if b in v.given_up_booths:
                u -= cfg.giveup_utility_penalty  # 諦めブースは減点して候補に残す
            cand_ids.append(b)
            utils.append(u)

        if not cand_ids or max(utils) < cfg.utility_min_threshold:
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
            v.state = WALKING
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
        v.state = WALKING

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
        cells = descend_path(self.booth_fields[b], pos)
        if not cells:
            return 0.0
        s = 0.0
        for (x, y) in cells:
            s += self.occupancy[y, x] / float(self.capacity[y, x])
        return min(1.0, s / len(cells))

    def _start_exiting(self, v: Visitor) -> None:
        v.state = EXITING
        v.target_booth_id = None
        if v.position in self.exit_set:
            self._finish(v)
            return
        path = astar(
            self.walkable,
            self.occupancy,
            self.capacity,
            v.position,
            self.exit_set,
            v.congestion_avoidance,
            self.exit_field,
        )
        v.path = path if path is not None else []
        v.path_index = 0
        v.steps_since_plan = 0
        v.blocked_count = 0

    def _finish(self, v: Visitor) -> None:
        x, y = v.position
        self.occupancy[y, x] -= 1
        v.position = None
        v.state = FINISHED

    # ------------------------------------------------------------ 5. 移動(§3.4/§10)
    def _movement_step(self, t: int) -> None:
        cfg = self.cfg
        movers = [v for v in self.visitors if v.state in (WALKING, EXITING) and v.position]

        # 移動距離残高の加算(§3.4)。累積上限は 1 ステップの最大移動量。
        credit_cap = float(cfg.n_substeps)
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
            for v in movers:
                if (
                    v.state in (WALKING, EXITING)
                    and v.position is not None
                    and v.move_credit >= 1.0
                    and v.path_index < len(v.path)
                ):
                    nxt = v.path[v.path_index]
                    proposals.setdefault(nxt, []).append(v)
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
                v.move_credit -= 1.0
                v.moved_cells += 1
                v.blocked_count = 0
                if v.path_index >= len(v.path):
                    arrivals.append(v)

            for v in arrivals:
                self._handle_arrival(v, t)

    def _replan(self, v: Visitor) -> None:
        """§9.3 の再探索。計画時点の占有スナップショットでコスト評価(§9.2)。"""
        if v.state == WALKING and v.target_booth_id is None:  # 防御(通常は発生しない)
            v.state = CHOOSING
            v.path, v.path_index = [], 0
            return
        if v.state == EXITING:
            goal_set, hfield = self.exit_set, self.exit_field
        else:
            b = v.target_booth_id
            goal_set, hfield = {self.layout.receptions[b]}, self.booth_fields[b]
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
            v.given_up_booths.add(v.target_booth_id)
            v.give_up_count += 1
            v.state = CHOOSING
            v.path, v.path_index = [], 0

    def _handle_arrival(self, v: Visitor, t: int) -> None:
        if v.state == EXITING:
            if v.position in self.exit_set:
                self._finish(v)
            return
        if v.state == WALKING and v.target_booth_id is not None:
            if v.position == self.layout.receptions[v.target_booth_id]:
                self._arrive_at_reception(v, t)

    # ------------------------------------------------------------ 入場・行列参加(§11.1, §11.3)
    def _arrive_at_reception(self, v: Visitor, t: int) -> None:
        cfg = self.cfg
        b = v.target_booth_id
        spec = cfg.booths[b]
        rt = self.booths[b]
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
            v.state = EXPERIENCING
            rt.active_users[v.visitor_id] = t + self.service_steps[b]
            return

        # 行列参加判定(§11.3): 満杯 or 残り時間不足なら諦め
        est_wait = len(rt.queue) * spec.mean_service_seconds / spec.capacity
        remaining = (cfg.n_steps - t) * cfg.dt_seconds
        queue_capacity = len(self.layout.queue_cells[b]) * cfg.queue_capacity_per_cell
        if len(rt.queue) >= queue_capacity or remaining < est_wait + spec.mean_service_seconds:
            v.give_up_count += 1
            v.given_up_booths.add(b)
            v.state = CHOOSING
            return

        slot = len(rt.queue)
        qx, qy = self.layout.queue_cells[b][slot // cfg.queue_capacity_per_cell]
        self.occupancy[ry, rx] -= 1
        self.occupancy[qy, qx] += 1  # 待機者は実セルを占有(§5.2)
        v.position = (qx, qy)
        v.state = QUEUING
        v.current_queue_wait = 0.0
        rt.queue.append(v.visitor_id)

    # ------------------------------------------------------------ 6. 集計(§12.2)
    def _metrics_step(self, t: int) -> None:
        cfg = self.cfg

        occ = self.occupancy[self.walkable].astype(np.float64)
        cap = self.capacity[self.walkable].astype(np.float64)
        weights = self.congestion_weights[self.walkable].astype(np.float64)

        # ---------------------------------------------------------
        # 1. タイムステップ t における混雑度
        # ---------------------------------------------------------
        #excess = np.maximum(
        #    0.0,
        #    occ / cap - cfg.safe_occupancy_ratio,
        #)

        denom = np.maximum(1.0, cap - 1.0)  # 1人までは混雑なし、容量いっぱいで1.0
                                 
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
        spawned = [
            v for v in self.visitors
            if v.state != NOT_ARRIVED
        ]

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

        # ---------------------------------------------------------
        # 3. 時系列として保存
        # ---------------------------------------------------------
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
        if not timed_out and len(self.time_steps) < self.cfg.n_steps:
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
