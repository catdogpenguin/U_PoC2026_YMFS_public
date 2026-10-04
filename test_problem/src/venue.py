"""venue.py — 会場グリッド・セル種別・容量・入口出口・レイアウト適用(§4, §5)。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import (
    BOOTH,
    EMPTY,
    ENTRANCE,
    EXIT,
    FORBIDDEN,
    QUEUE,
    WALKABLE_TYPES,
    Config,
)


@dataclass(frozen=True)
class Venue:
    """ベース会場(ブース未配置)。"""

    base_grid: np.ndarray       # int8 (H, W) セル種別
    base_capacity: np.ndarray   # int8 (H, W)
    entrance_cells: tuple       # ((x,y), ...)
    exit_cells: tuple


@dataclass(frozen=True)
class Layout:
    """有効配置をグリッドへ適用した結果(§5.1, §5.2)。"""

    grid: np.ndarray              # int8 (H, W) セル種別(BOOTH/QUEUE 反映済み)
    capacity: np.ndarray          # int8 (H, W)
    walkable: np.ndarray          # bool (H, W)
    positions: tuple              # ((booth_x, booth_y), ...) 左上基準(§3.5)
    receptions: tuple             # ブースごとの受付セル (x, y)(§5.1)
    booth_exits: tuple             # ブース体験後に再出現するセル (x, y)
    queue_cells: tuple            # ブースごとの待ち行列セル列 ((x,y),...)(§5.2)


def build_base_venue(cfg: Config) -> Venue:
    """§4.1〜§4.5 に従いベース会場を構築する。

    - 全セル EMPTY、既定容量
    - 入口・出口: Config で指定された辺の中央セル
    - 入口・出口の周囲 clearance セル以内はブース配置不可 FORBIDDEN(通行可)
    """
    H, W = cfg.grid_height, cfg.grid_width
    grid = np.full((H, W), EMPTY, dtype=np.int8)
    capacity = np.full((H, W), cfg.cap_default, dtype=np.int8)

    for cells, ctype, cap in (
        (cfg.entrance_cells, ENTRANCE, cfg.cap_entrance),
        (cfg.exit_cells, EXIT, cfg.cap_exit),
    ):
        # 周囲 clearance セルを FORBIDDEN 化(先に塗ってから本体を上書き)
        r = cfg.entrance_clearance
        for (x, y) in cells:
            y0, y1 = max(0, y - r), min(H, y + r + 1)
            x0, x1 = max(0, x - r), min(W, x + r + 1)
            zone = grid[y0:y1, x0:x1]
            zone[zone == EMPTY] = FORBIDDEN
        for (x, y) in cells:
            grid[y, x] = ctype
            capacity[y, x] = cap

    return Venue(
        base_grid=grid,
        base_capacity=capacity,
        entrance_cells=cfg.entrance_cells,
        exit_cells=cfg.exit_cells,
    )


def reception_cell(cfg: Config, booth_id: int, bx: int, by: int) -> tuple[int, int]:
    """設定されたブース受付セルを返す。"""
    spec = cfg.booths[booth_id]
    if cfg.reception_position == "bottom_center":
        return (bx + spec.width // 2, by + spec.height)
    if cfg.reception_position == "bottom_left":
        return (bx, by + spec.height)
    raise ValueError(f"unsupported reception_position: {cfg.reception_position!r}")


def booth_exit_cell(cfg: Config, booth_id: int, bx: int, by: int) -> tuple[int, int]:
    """設定されたブース体験後の退出セルを返す。"""
    if cfg.booth_exit_position == "reception":
        return reception_cell(cfg, booth_id, bx, by)
    if cfg.booth_exit_position == "top_left":
        return (bx, by - 1)
    raise ValueError(f"unsupported exit_position: {cfg.booth_exit_position!r}")


def queue_cells_of(cfg: Config, booth_id: int, bx: int, by: int) -> tuple[tuple[int, int], ...]:
    """受付側を先頭として、設定された形状の待ち行列セル列を返す。"""
    spec = cfg.booths[booth_id]
    ex, ey = reception_cell(cfg, booth_id, bx, by)
    if cfg.queue_pattern == "straight_down":
        return tuple((ex, ey + 1 + k) for k in range(spec.max_queue_length))
    if cfg.queue_pattern == "bottom_then_right":
        # ブース下辺を左→右、その後に右辺を下→上へ並ぶ L 字。
        # 左下の受付セルは到着・入場処理専用とし、待機列には含めない。
        cells = [(bx + dx, by + spec.height) for dx in range(spec.width)]
        cells.append((bx + spec.width, by + spec.height))
        cells.extend(
            (bx + spec.width, y)
            for y in range(by + spec.height - 1, by - 1, -1)
        )
        dedicated_cells = (cell for cell in dict.fromkeys(cells) if cell != (ex, ey))
        return tuple(dedicated_cells)[: spec.max_queue_length]
    raise ValueError(f"unsupported queue_pattern: {cfg.queue_pattern!r}")


def booth_cells_of(cfg: Config, booth_id: int, bx: int, by: int) -> list[tuple[int, int]]:
    """ブース占有セル(§3.5)。会場外セルも含めてそのまま返す(制約チェック用)。"""
    spec = cfg.booths[booth_id]
    return [(x, y) for y in range(by, by + spec.height) for x in range(bx, bx + spec.width)]


def position_bounds(cfg: Config, booth_id: int) -> tuple[int, int, int, int]:
    """ブース本体・受付・退出・待機列が場内に収まる左上座標の上下限。"""
    cells = booth_cells_of(cfg, booth_id, 0, 0)
    cells += [reception_cell(cfg, booth_id, 0, 0)]
    cells += [booth_exit_cell(cfg, booth_id, 0, 0)]
    cells += list(queue_cells_of(cfg, booth_id, 0, 0))
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return (
        -min(xs),
        cfg.grid_width - 1 - max(xs),
        -min(ys),
        cfg.grid_height - 1 - max(ys),
    )


def apply_layout(cfg: Config, venue: Venue, positions: tuple) -> Layout:
    """有効(制約充足済み)配置をグリッドへ適用する。

    前提: constraints.compute_violation() で g == 0 が確認済みであること。
    """
    grid = venue.base_grid.copy()
    capacity = venue.base_capacity.copy()
    receptions = []
    booth_exits = []
    queue_cells = []
    for bid, (bx, by) in enumerate(positions):
        spec = cfg.booths[bid]
        grid[by : by + spec.height, bx : bx + spec.width] = BOOTH
        rx, ry = reception_cell(cfg, bid, bx, by)
        receptions.append((rx, ry))
        booth_exits.append(booth_exit_cell(cfg, bid, bx, by))
        qcells = queue_cells_of(cfg, bid, bx, by)
        queue_cells.append(qcells)
        for (qx, qy) in qcells:
            grid[qy, qx] = QUEUE
            capacity[qy, qx] = cfg.cap_queue

    walkable = np.isin(grid, WALKABLE_TYPES)
    if not cfg.queue_walkable:
        walkable[grid == QUEUE] = False
    return Layout(
        grid=grid,
        capacity=capacity,
        walkable=walkable,
        positions=tuple(tuple(p) for p in positions),
        receptions=tuple(receptions),
        booth_exits=tuple(booth_exits),
        queue_cells=tuple(queue_cells),
    )
