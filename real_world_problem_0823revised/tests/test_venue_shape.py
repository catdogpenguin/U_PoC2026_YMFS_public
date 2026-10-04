"""
実会場形状(config/default_config.yaml の venue_shape)の構築検査。

実行用コマンド：
python real_world_problem/tests/preview_venue_shape.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models import EMPTY, FORBIDDEN, WALL, WALKABLE_TYPES, load_config
from src.routing import INF_DIST, bfs_distance_field
from src.venue import build_base_venue

ROOT = Path(__file__).resolve().parents[1]


def _real_cfg():
    return load_config(ROOT / "config")


def test_real_venue_cell_types():
    cfg = _real_cfg()
    venue = build_base_venue(cfg)
    g = venue.base_grid
    # 場外(建物・図のハッチング部)は WALL
    assert g[20, 0] == WALL         # 西側道路の西(建物)
    assert g[0, 56] == WALL         # 上部バスターミナル道路の北東の外
    assert g[90, 50] == WALL        # 会場南東の外
    # 広場はブース配置可能(EMPTY)
    assert g[20, 25] == EMPTY       # 上部広場
    assert g[60, 25] == EMPTY       # 下部広場
    # 道路は歩行可・配置不可(FORBIDDEN)
    assert g[20, 13] == FORBIDDEN   # 西側道路
    assert g[40, 25] == FORBIDDEN   # 広場間の 8m 道路
    assert g[85, 25] == FORBIDDEN   # 南側道路
    # 樹木(3m×3m)は通行不可
    assert g[67, 34] == WALL


def test_real_venue_openings():
    cfg = _real_cfg()
    venue = build_base_venue(cfg)
    H, W = cfg.grid_height, cfg.grid_width
    # 入口・出口の区別はなく、境界開口部を共通の入退場セル集合とする
    assert venue.entrance_cells == venue.exit_cells
    assert len(venue.entrance_cells) > 0
    for (x, y) in venue.entrance_cells:
        assert x == 0 or x == W - 1 or y == 0 or y == H - 1
    xs = {x for (x, y) in venue.entrance_cells}
    ys = {y for (x, y) in venue.entrance_cells}
    assert 0 in xs and W - 1 in xs   # 8m 道路の西端・バスターミナル側の東端
    assert 0 in ys and H - 1 in ys   # 北側道路の北端・南側縦道路の南端


def test_real_venue_connectivity():
    cfg = _real_cfg()
    venue = build_base_venue(cfg)
    walkable = np.isin(venue.base_grid, WALKABLE_TYPES)
    dist = bfs_distance_field(walkable, venue.entrance_cells)
    # 全歩行可能セル(両広場を含む)へ開口部から到達できる
    assert dist[20, 25] < INF_DIST
    assert dist[60, 25] < INF_DIST
    assert np.all(dist[walkable] < INF_DIST)
