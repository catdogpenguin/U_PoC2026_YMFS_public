"""§23.2: セル容量・一括競合解決、§23.5: 目的関数の直交性。"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.metrics import free_walk_seconds
from src.models import WALKING, Visitor
from src.simulator import Simulation
from src.venue import apply_layout
from tests.conftest import CENTRAL_LAYOUT


def make_visitor(vid: int, pos, speed: float = 1.5) -> Visitor:
    v = Visitor(
        visitor_id=vid,
        spawn_time_step=0,
        position=pos,
        speed_mps=speed,
        congestion_avoidance=1.0,
        queue_tolerance_seconds=300.0,
        max_visit_seconds=3600.0,
        interests=np.zeros(5),
        distance_weight_factor=1.0,
        profile_name="test",
        state=WALKING,
    )
    v.actual_spawn_step = 0
    return v


def make_sim(cfg, ctx) -> Simulation:
    layout = apply_layout(cfg, ctx.venue, CENTRAL_LAYOUT)
    return Simulation(cfg, ctx.venue, layout, visitor_records=())


def place(sim: Simulation, v: Visitor) -> None:
    """検査用: 来場者をグリッドへ直接配置(占有整合を保つ)。"""
    x, y = v.position
    sim.occupancy[y, x] += 1
    sim.visitors.append(v)


def test_capacity_not_exceeded(cfg, ctx):
    """容量 3 のセルに 4 人目が入れない。許可人数が容量を超えない。"""
    sim = make_sim(cfg, ctx)
    target = (5, 5)  # EMPTY・容量 3
    starts = [(4, 5), (6, 5), (5, 4), (5, 6)]
    for i, s in enumerate(starts):
        v = make_visitor(i, s)
        v.path = [target]
        place(sim, v)
    sim._movement_step(t=0)
    tx, ty = target
    assert sim.occupancy[ty, tx] == 3  # 4 人目は入れない
    arrived = [v for v in sim.visitors if v.position == target]
    blocked = [v for v in sim.visitors if v.position != target]
    assert len(arrived) == 3 and len(blocked) == 1
    b = blocked[0]
    assert b.blocked_count >= 1 and b.blocked_seconds > 0
    assert b.moved_cells == 0


def test_blocked_can_move_next_substep(cfg, ctx):
    """このサブステップの退出による空きは、次サブステップから有効になり移動できる。"""
    # 20260629 の既定は 1 サブステップなので、この機構の単体検査時だけ 2 にする。
    substep_cfg = replace(cfg, max_speed_mps=2.0)
    layout = apply_layout(substep_cfg, ctx.venue, CENTRAL_LAYOUT)
    sim = Simulation(substep_cfg, ctx.venue, layout, visitor_records=())
    cell_a = (5, 5)
    # cell_a を満杯(3 人)にする。うち 1 人は今ステップで外へ出る
    stayers = [make_visitor(i, cell_a) for i in range(2)]
    for v in stayers:
        v.path = []  # 動かない
        place(sim, v)
    leaver = make_visitor(10, cell_a)
    leaver.path = [(6, 5)]
    place(sim, leaver)
    # 入りたい人
    joiner = make_visitor(20, (4, 5))
    joiner.path = [cell_a]
    place(sim, joiner)

    sim._movement_step(t=0)
    # leaver は退出、joiner は 2 サブステップ目以降に入れている
    assert leaver.position == (6, 5)
    assert joiner.position == cell_a
    assert joiner.blocked_seconds > 0  # 1 サブステップ目は拒否された
    ax, ay = cell_a
    assert sim.occupancy[ay, ax] == 3


def test_f_move_not_polluted_by_blocking(cfg, ctx):
    """§23.5: ブロック待機しても f_move(自由歩行換算)は増加しない。

    ブロックされ続けた来場者の moved_cells は 0 のままで、
    blocked_seconds のみが補助 KPI として計上される。
    """
    sim = make_sim(cfg, ctx)
    target = (5, 5)
    # target を満杯に固定
    for i in range(3):
        v = make_visitor(i, target)
        v.path = []
        place(sim, v)
    stuck = make_visitor(30, (4, 5))
    stuck.path = [target]
    place(sim, stuck)

    for t in range(3):
        # 再計画で経路が変わらないよう毎回リセット(検査目的)
        stuck.path, stuck.path_index = [target], 0
        stuck.blocked_count = 0
        sim._movement_step(t=t)

    assert stuck.moved_cells == 0
    assert free_walk_seconds(stuck, cfg) == 0.0  # f_move への寄与はゼロ
    assert stuck.blocked_seconds > 0  # 補助 KPI には計上される
