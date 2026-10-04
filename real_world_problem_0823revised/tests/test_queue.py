"""§23.3: ブース体験・待ち行列・諦めのテスト。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models import CHOOSING, EXPERIENCING, QUEUING, WALKING, load_config
from src.simulator import Simulation
from src.venue import apply_layout
from tests.conftest import CENTRAL_LAYOUT
from tests.test_movement import make_visitor


def make_sim(cfg, ctx) -> Simulation:
    layout = apply_layout(cfg, ctx.venue, CENTRAL_LAYOUT)
    return Simulation(cfg, ctx.venue, layout, visitor_records=())


def arrive(sim: Simulation, v, booth_id: int, t: int = 0) -> None:
    """検査用: 来場者を受付セルへ置いて到着処理を呼ぶ。"""
    rx, ry = sim.layout.receptions[booth_id]
    v.position = (rx, ry)
    v.target_booth_id = booth_id
    v.state = WALKING
    sim.occupancy[ry, rx] += 1
    sim.visitors.append(v)
    sim._arrive_at_reception(v, t)


def test_current_config_uses_two_people_per_queue_cell():
    """現行設定のグリッド容量と待機人数換算を1セル2人へ統一する。"""

    current = load_config(Path(__file__).resolve().parents[1] / "config")
    assert current.cap_queue == 2
    assert current.queue_capacity_per_cell == 2


def test_capacity_and_queue(cfg, ctx):
    """収容人数を超えて体験者が入らない。超過分は待ち行列の実セルを占有する。"""
    sim = make_sim(cfg, ctx)
    b = 0  # 収容 1
    spec = cfg.booths[b]
    visitors = [make_visitor(i, None) for i in range(spec.capacity + 2)]
    for v in visitors:
        arrive(sim, v, b)
    rt = sim.booths[b]
    assert len(rt.active_users) == spec.capacity
    assert len(rt.queue) == 2
    q0 = sim.layout.queue_cells[b][0]
    assert visitors[spec.capacity].position == q0  # 1 セル最大 5 人
    assert visitors[spec.capacity + 1].position == q0
    assert q0 != sim.layout.receptions[b]
    assert all(
        sim._visitor(vid).position in sim.layout.queue_cells[b]
        for vid in rt.queue
    )
    # 体験中はグリッド占有に含まれない(§11.4)
    for v in visitors[: spec.capacity]:
        assert v.position is None and v.state == EXPERIENCING
    assert int(sim.occupancy.sum()) == 2  # 同じ行列セル上の 2 人のみ


def test_queue_head_enters_after_service(cfg, ctx):
    """体験終了後にキュー先頭が入場し、行列が前詰めされる。"""
    sim = make_sim(cfg, ctx)
    b = 0
    spec = cfg.booths[b]
    visitors = [make_visitor(i, None) for i in range(spec.capacity + 2)]
    for v in visitors:
        arrive(sim, v, b, t=0)
    rt = sim.booths[b]
    t_done = sim.service_steps[b]
    sim._booth_step(t=t_done)  # 体験完了 → 再出現 → キュー入場
    # 完了者はブース上側の専用退出セルへ再出現する
    reappeared = [v for v in visitors[: spec.capacity] if v.position is not None]
    assert len(reappeared) == spec.capacity
    for v in reappeared:
        assert b in v.visited_booths and v.state == CHOOSING
    # 専用退出セルのため、完了者の退出と同じ更新でキュー先頭が入場できる
    assert not rt.exit_waiting
    assert len(rt.active_users) == spec.capacity
    assert len(rt.queue) == 1


def test_queue_tolerance_give_up(cfg, ctx):
    """待ち許容時間超過で諦めて CHOOSING へ戻る。"""
    sim = make_sim(cfg, ctx)
    b = 0
    spec = cfg.booths[b]
    fillers = [make_visitor(i, None) for i in range(spec.capacity)]
    for v in fillers:
        arrive(sim, v, b)
    waiter = make_visitor(99, None)
    waiter.queue_tolerance_seconds = 2.0
    arrive(sim, waiter, b)
    assert waiter.state == QUEUING
    sim._queue_step(t=1)
    assert waiter.state == QUEUING  # 1 秒経過(未超過)
    sim._queue_step(t=2)
    assert waiter.state == QUEUING  # 2 秒経過(= 許容値、未超過)
    sim._queue_step(t=3)
    assert waiter.state == CHOOSING  # 3 秒 > 2 秒で諦め
    assert waiter.give_up_count == 1 and b in waiter.given_up_booths
    assert len(sim.booths[b].queue) == 0


def test_queue_full_give_up(cfg, ctx):
    """L字待機列の全セル×セル容量に達していると諦める。"""
    sim = make_sim(cfg, ctx)
    b = 5  # 収容 2・行列 8
    spec = cfg.booths[b]
    n = spec.capacity + len(sim.layout.queue_cells[b]) * cfg.queue_capacity_per_cell
    for i in range(n):
        arrive(sim, make_visitor(i, None), b)
    late = make_visitor(999, None)
    arrive(sim, late, b)
    assert late.state == CHOOSING and late.give_up_count == 1
