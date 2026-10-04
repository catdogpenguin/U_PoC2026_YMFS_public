"""§23.1: 配置制約のテスト。"""
from __future__ import annotations

from src.constraints import compute_violation
from src.venue import apply_layout
from tests.conftest import CENTRAL_LAYOUT, PERIMETER_LAYOUT


def test_valid_layouts(cfg, ctx):
    """手作りの中央密集・周辺分散配置は有効(g == 0)であること。"""
    assert compute_violation(cfg, ctx.venue, CENTRAL_LAYOUT) == 0
    assert compute_violation(cfg, ctx.venue, PERIMETER_LAYOUT) == 0


def test_20260629_geometry(cfg, ctx):
    """下辺入口・上辺出口、ブース左下受付、左上退出、L字列を再現する。"""
    layout = apply_layout(cfg, ctx.venue, CENTRAL_LAYOUT)
    assert ctx.venue.entrance_cells == ((9, 15), (10, 15))
    assert ctx.venue.exit_cells == ((9, 0), (10, 0))
    bx, by = CENTRAL_LAYOUT[0]
    assert layout.receptions[0] == (bx, by + 2)
    assert layout.booth_exits[0] == (bx, by - 1)
    assert layout.queue_cells[0] == (
        (bx + 1, by + 2),
        (bx + 2, by + 2),
        (bx + 2, by + 1),
        (bx + 2, by),
    )
    assert layout.receptions[0] not in layout.queue_cells[0]


def test_overlap_detected(cfg, ctx):
    """ブース重複を検出できる。"""
    pos = list(CENTRAL_LAYOUT)
    pos[1] = pos[0]  # B1 を B0 と同位置に
    assert compute_violation(cfg, ctx.venue, tuple(pos)) > 0


def test_out_of_bounds_detected(cfg, ctx):
    """会場外配置を検出できる。"""
    pos = list(CENTRAL_LAYOUT)
    pos[0] = (cfg.grid_width - 1, 0)  # 右へはみ出す
    assert compute_violation(cfg, ctx.venue, tuple(pos)) > 0


def test_forbidden_zone_detected(cfg, ctx):
    """入口セルとの重なりを検出できる。"""
    pos = list(CENTRAL_LAYOUT)
    pos[0] = (9, 13)  # 受付/待機列が下辺入口に重なる
    assert compute_violation(cfg, ctx.venue, tuple(pos)) > 0


def test_queue_space_detected(cfg, ctx):
    """受付・L字待ち行列が確保できない配置(下端すれすれ)を検出できる。"""
    pos = list(CENTRAL_LAYOUT)
    spec = cfg.booths[0]
    pos[0] = (5, cfg.grid_height - spec.height)
    assert compute_violation(cfg, ctx.venue, tuple(pos)) > 0


def test_20260629_has_no_extra_corridor_width_constraint(cfg):
    """旧条件どおり、既定の最低通路幅は 1 セルとする。"""
    assert cfg.min_corridor_width == 1
