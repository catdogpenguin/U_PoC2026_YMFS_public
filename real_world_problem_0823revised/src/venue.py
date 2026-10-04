"""venue.py — 会場グリッド・セル種別・容量・入口出口・レイアウト適用(§4, §5)。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .models import (
    BOOTH,
    EMPTY,
    ENTRANCE,
    EXIT,
    FORBIDDEN,
    QUEUE,
    WALL,
    WALKABLE_TYPES,
    Config,
)

# theta は北から時計回りの方向コードとする。
THETA_NORTH = 0
THETA_EAST = 1
THETA_SOUTH = 2
THETA_WEST = 3

# UI項目は追加せず、環境選択と経路割当で評価する最大通路幅を4セルに固定する。
#: 到達性を測る目標通路幅。1 にすると幅の概念が無効化される。
#: 2026-08-17: 4 → 1。「4 マス幅」制約を破棄したため。
#:   - passable_corridor_mask(raw, 1) は raw をそのまま返すので、
#:     _widest_path の 4→3→2→1 フォールバックは実質何もしなくなる。
#:   - compute_width_reachability_profile の four_cell_achievement_count は
#:     「幅 1 で到達できるブース数」になり、実行可能解では常に全基となるため、
#:     環境選択の第 1 キー(-width4_counts)が全個体で同値になり無効化される。
TARGET_CORRIDOR_WIDTH_CELLS = 1

#: ブース正面に常時確保する受付・整列・通行用空間の奥行き。
#: 正面空間同士は共用できるが、他ブースの本体・受付・退出口・待機列は
#: この領域へ侵入できない。
FRONT_CLEARANCE_DEPTH_CELLS = 4

VALID_THETAS = (
    THETA_NORTH,
    THETA_EAST,
    THETA_SOUTH,
    THETA_WEST,
)

THETA_LABELS = {
    THETA_NORTH: "north",
    THETA_EAST: "east",
    THETA_SOUTH: "south",
    THETA_WEST: "west",
}

THETA_ARROWS = {
    THETA_NORTH: "↑",
    THETA_EAST: "→",
    THETA_SOUTH: "↓",
    THETA_WEST: "←",
}


def _validate_theta(theta: int) -> int:
    theta = int(theta)

    if theta not in VALID_THETAS:
        raise ValueError(
            f"theta must be one of {VALID_THETAS}: "
            f"{theta}"
        )

    return theta


def oriented_booth_size(
    cfg: Config,
    booth_id: int,
    theta: int,
) -> tuple[int, int]:
    """向きを反映したブース本体の (幅, 高さ) を返す。"""
    spec = cfg.booths[booth_id]

    if spec.is_table:
        return spec.width, spec.height

    theta = _validate_theta(theta)

    if theta in (
        THETA_NORTH,
        THETA_SOUTH,
    ):
        return spec.width, spec.height

    return spec.height, spec.width


def front_clearance_cells(
    cfg: Config,
    booth_id: int,
    x_pos: int,
    y_pos: int,
    theta: int,
    depth: int = FRONT_CLEARANCE_DEPTH_CELLS,
) -> tuple[tuple[int, int], ...]:
    """ブース正面幅全体に確保する、奥行き ``depth`` の空間を返す。

    正面を持たないテーブル・椅子には適用しない。受付・退出口など当該ブース
    自身の設備は置けるが、他の配置物がこの領域へ侵入する配置は物理制約違反
    として扱う。
    """

    if cfg.booths[booth_id].is_table:
        return ()
    width, height = oriented_booth_size(cfg, booth_id, int(theta))
    if int(theta) == THETA_NORTH:
        cells = [
            (x_value, y_pos - offset)
            for offset in range(1, int(depth) + 1)
            for x_value in range(x_pos, x_pos + width)
        ]
    elif int(theta) == THETA_EAST:
        cells = [
            (x_pos + width - 1 + offset, y_value)
            for offset in range(1, int(depth) + 1)
            for y_value in range(y_pos, y_pos + height)
        ]
    elif int(theta) == THETA_SOUTH:
        cells = [
            (x_value, y_pos + height - 1 + offset)
            for offset in range(1, int(depth) + 1)
            for x_value in range(x_pos, x_pos + width)
        ]
    else:  # THETA_WEST
        cells = [
            (x_pos - offset, y_value)
            for offset in range(1, int(depth) + 1)
            for y_value in range(y_pos, y_pos + height)
        ]
    return tuple(cells)


def _rotate_relative_cell(
    spec,
    x: int,
    y: int,
    theta: int,
) -> tuple[int, int]:
    """南向きを基準にした相対セルを、指定方向へ回転する。

    入力の相対座標はブース本体の左上を (0, 0) とする。
    受付・退出口・待機列のような本体外セルにも同じ変換を適用できる。
    回転後も、回転したブース本体の左上が (0, 0) になる。
    """
    theta = _validate_theta(theta)

    w = int(spec.width)
    h = int(spec.height)

    if theta == THETA_SOUTH:
        return x, y

    if theta == THETA_EAST:
        return y, w - 1 - x

    if theta == THETA_NORTH:
        return w - 1 - x, h - 1 - y

    return h - 1 - y, x


@dataclass(frozen=True)
class Venue:
    """ベース会場(ブース未配置)。"""

    base_grid: np.ndarray       # int8 (H, W) セル種別
    base_capacity: np.ndarray   # int8 (H, W)
    entrance_cells: tuple       # ((x,y), ...)
    exit_cells: tuple
    area_masks: dict = None     # 広場名 -> bool (H, W)(所属広場制約用)


@dataclass(frozen=True)
class Layout:
    """有効配置をグリッドへ適用した結果(§5.1, §5.2)。"""

    grid: np.ndarray              # int8 (H, W) セル種別(BOOTH/QUEUE 反映済み)
    capacity: np.ndarray          # int8 (H, W)
    walkable: np.ndarray          # bool (H, W)
    positions: tuple              # ((booth_x, booth_y, theta), ...) 回転後左上基準
    receptions: tuple             # ブースごとの受付セル (x, y)(§5.1)
    booth_exits: tuple             # ブース体験後に再出現するセル (x, y)
    queue_cells: tuple            # ブースごとの待ち行列セル列 ((x,y),...)(§5.2)
    table_access_cells: tuple      # テーブル周囲4辺の利用可能セル列
    # ブース等だけを除いた元の通行可能マスク。walkable はここから
    # 最低通路幅を満たさないセルを除いた、実際に経路探索で使うマスク。
    raw_walkable: np.ndarray | None = None


def _points_in_polygon(px: np.ndarray, py: np.ndarray, polygon) -> np.ndarray:
    """偶奇則(レイキャスト)による点内包判定。

    セル中心(半整数)と整数頂点の組み合わせを想定し、境界上の点は扱わない。
    """
    inside = np.zeros(px.shape, dtype=bool)
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if y1 == y2:
            continue
        crosses = (y1 > py) != (y2 > py)
        x_at = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (px < x_at)
    return inside


def _rasterize_one(cfg: Config, polygon) -> np.ndarray:
    """セル中心 ((x+0.5), (y+0.5)) × cell_size_meters による 1 多角形のマスク。"""
    H, W = cfg.grid_height, cfg.grid_width
    ys, xs = np.mgrid[0:H, 0:W]
    px = (xs + 0.5) * cfg.cell_size_meters
    py = (ys + 0.5) * cfg.cell_size_meters
    return _points_in_polygon(px, py, polygon)


def _rasterize(cfg: Config, named_polygons) -> np.ndarray:
    """(name, polygon) の列を合併マスクへ変換する。"""
    H, W = cfg.grid_height, cfg.grid_width
    mask = np.zeros((H, W), dtype=bool)
    for _name, poly in named_polygons:
        mask |= _rasterize_one(cfg, poly)
    return mask


def venue_masks(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """(場内マスク, 道路マスク) を返す。venue_shape 未指定なら全面が配置可能区域。

    - 場内: 配置区域または道路に含まれ、障害物に含まれないセル
    - 道路: 場内のうち歩行のみ可能なセル(配置区域と重なる部分は配置区域を優先)
    """
    H, W = cfg.grid_height, cfg.grid_width
    if not cfg.venue_areas:
        return np.ones((H, W), dtype=bool), np.zeros((H, W), dtype=bool)
    area = _rasterize(cfg, cfg.venue_areas)
    walkway = _rasterize(cfg, cfg.venue_walkways)
    obstacle = _rasterize(cfg, cfg.venue_obstacles)
    inside = (area | walkway) & ~obstacle
    walkway_only = walkway & ~area & ~obstacle
    return inside, walkway_only


def boundary_openings(walkway_only: np.ndarray) -> tuple[tuple[int, int], ...]:
    """グリッド境界に接する道路セル=領域外への開口部を (x, y) で返す。

    来場者は領域外に通じる道路から到来・退出する想定(入口/出口の区別なし)。
    """
    H, W = walkway_only.shape
    boundary = np.zeros((H, W), dtype=bool)
    boundary[0, :] = boundary[H - 1, :] = True
    boundary[:, 0] = boundary[:, W - 1] = True
    return tuple((int(x), int(y)) for y, x in np.argwhere(walkway_only & boundary))


def build_base_venue(cfg: Config) -> Venue:
    """§4.1〜§4.5 に従いベース会場を構築する。

    - venue_shape の場外セルは WALL(通行・配置とも不可)
    - 配置区域(広場)は EMPTY、道路は FORBIDDEN(通行可・配置不可)。
      広場と道路の境界は自由に通り抜けられる。
    - 入退場セル: 明示指定があればそれを使用。venue_shape 使用時は境界開口部
      (領域外に通じる道路セル)を入口・出口共通の集合として自動導出する。
    - 入退場セルの周囲 clearance セル以内はブース配置不可 FORBIDDEN(通行可)
    """
    H, W = cfg.grid_height, cfg.grid_width
    grid = np.full((H, W), EMPTY, dtype=np.int8)
    capacity = np.full((H, W), cfg.cap_default, dtype=np.int8)

    inside, walkway_only = venue_masks(cfg)
    grid[~inside] = WALL
    grid[walkway_only] = FORBIDDEN

    if cfg.entrance_cells_override or cfg.exit_cells_override or not cfg.venue_areas:
        entrance_cells = cfg.entrance_cells
        exit_cells = cfg.exit_cells
    else:
        entrance_cells = exit_cells = boundary_openings(walkway_only)
        if not entrance_cells:
            raise ValueError(
                "venue_shape にグリッド境界へ達する道路(開口部)がありません。"
                "walkways を境界まで延ばすか、entrance_cells/exit_cells を明示してください"
            )
    for (x, y) in (*entrance_cells, *exit_cells):
        if not (0 <= x < W and 0 <= y < H) or not inside[y, x]:
            raise ValueError(f"入口/出口セル {(x, y)} が会場範囲外です")

    for cells, ctype, cap in (
        (entrance_cells, ENTRANCE, cfg.cap_entrance),
        (exit_cells, EXIT, cfg.cap_exit),
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

    # 広場名 -> マスク(BoothSpec.area の所属広場制約用。障害物セルは除く)
    obstacle = _rasterize(cfg, cfg.venue_obstacles)
    area_masks = {
        name: _rasterize_one(cfg, poly) & ~obstacle
        for name, poly in cfg.venue_areas
    }

    return Venue(
        base_grid=grid,
        base_capacity=capacity,
        entrance_cells=entrance_cells,
        exit_cells=exit_cells,
        area_masks=area_masks,
    )


def _south_reception_relative(
    cfg: Config,
    booth_id: int,
) -> tuple[int, int] | None:
    """南向きを基準にした受付セルの相対座標を返す。

    受付・待機列・退出は常にブース本体の南辺(theta 回転前の正面)に配置する。
    ブースの正面方向は theta で表現するため、この関数自体はアスペクト比に
    依存しない(theta と正面方向を一致させるため)。
    """
    spec = cfg.booths[booth_id]

    if spec.is_table:
        return None

    if cfg.reception_position == "bottom_center":
        return (
            spec.width // 2,
            spec.height,
        )

    if cfg.reception_position == "bottom_left":
        return (
            0,
            spec.height,
        )

    raise ValueError(
        "unsupported reception_position: "
        f"{cfg.reception_position!r}"
    )


def reception_cell(
    cfg: Config,
    booth_id: int,
    bx: int,
    by: int,
    theta: int,
) -> tuple[int, int] | None:
    """角度を反映したブース受付セルを返す。テーブルはNone。"""
    relative = _south_reception_relative(
        cfg,
        booth_id,
    )

    if relative is None:
        return None

    rx, ry = _rotate_relative_cell(
        cfg.booths[booth_id],
        relative[0],
        relative[1],
        theta,
    )

    return (
        bx + rx,
        by + ry,
    )


def booth_exit_cell(
    cfg: Config,
    booth_id: int,
    bx: int,
    by: int,
    theta: int,
) -> tuple[int, int] | None:
    """角度を反映した体験後退出/テーブル離席セルを返す。"""
    spec = cfg.booths[booth_id]

    if spec.is_table:
        return None

    if cfg.booth_exit_position == "reception":
        return reception_cell(
            cfg,
            booth_id,
            bx,
            by,
            theta,
        )

    if cfg.booth_exit_position == "top_left":
        relative = (0, -1)
    elif cfg.booth_exit_position == "bottom_right":
        # 受付(南辺左端)と同じ辺(南辺=正面)の右端。待ち行列は
        # queue_pattern="left_then_clockwise" と組み合わせて南辺を空けておく前提。
        relative = (spec.width - 1, spec.height)
    else:
        raise ValueError(
            "unsupported exit_position: "
            f"{cfg.booth_exit_position!r}"
        )

    ex, ey = _rotate_relative_cell(
        spec,
        relative[0],
        relative[1],
        theta,
    )

    return (
        bx + ex,
        by + ey,
    )


def queue_cells_of(
    cfg: Config,
    booth_id: int,
    bx: int,
    by: int,
    theta: int,
) -> tuple[tuple[int, int], ...]:
    """受付側を先頭として、角度を反映した待ち行列セル列を返す。"""
    spec = cfg.booths[booth_id]

    if spec.is_table:
        return ()

    reception_relative = _south_reception_relative(
        cfg,
        booth_id,
    )

    assert reception_relative is not None

    if cfg.queue_pattern == "straight_down":
        south_cells = [
            (
                reception_relative[0],
                reception_relative[1] + 1 + k,
            )
            for k in range(
                spec.max_queue_length
            )
        ]

    elif cfg.queue_pattern == "bottom_then_right":
        # 南向き時のL字形状を作成後、ブース角度に合わせて回転する。
        cells = [
            (dx, spec.height)
            for dx in range(spec.width)
        ]

        cells.append(
            (
                spec.width,
                spec.height,
            )
        )

        cells.extend(
            (
                spec.width,
                y,
            )
            for y in range(
                spec.height - 1,
                -1,
                -1,
            )
        )

        south_cells = [
            cell
            for cell in dict.fromkeys(cells)
            if cell != reception_relative
        ][:spec.max_queue_length]

    elif cfg.queue_pattern == "left_then_clockwise":
        # 受付セル(南辺=正面)の左隣を起点に、ブース外周を時計回り
        # (左辺→上辺→右辺)へ一周する。南辺は受付・退出セル専用に空けておき、
        # exit_position="bottom_right"(南辺右端)と重複しないようにする。
        cells = [
            (reception_relative[0] - 1, y)
            for y in range(spec.height, -2, -1)
        ]

        cells.extend(
            (x, -1)
            for x in range(reception_relative[0] - 1, spec.width + 1)
        )

        cells.extend(
            (spec.width, y)
            for y in range(-1, spec.height + 1)
        )

        south_cells = [
            cell
            for cell in dict.fromkeys(cells)
            if cell != reception_relative
            and not (cell[1] == spec.height and 0 <= cell[0] < spec.width)
        ][:spec.max_queue_length]

    else:
        raise ValueError(
            "unsupported queue_pattern: "
            f"{cfg.queue_pattern!r}"
        )

    rotated = [
        _rotate_relative_cell(
            spec,
            x,
            y,
            theta,
        )
        for x, y in south_cells
    ]

    return tuple(
        (
            bx + x,
            by + y,
        )
        for x, y in rotated
    )


def booth_cells_of(
    cfg: Config,
    booth_id: int,
    bx: int,
    by: int,
    theta: int,
) -> list[tuple[int, int]]:
    """角度を反映したブース本体の占有セルを返す。"""
    width, height = oriented_booth_size(
        cfg,
        booth_id,
        theta,
    )

    return [
        (x, y)
        for y in range(
            by,
            by + height,
        )
        for x in range(
            bx,
            bx + width,
        )
    ]


def table_access_cells_of(
    cfg: Config,
    booth_id: int,
    bx: int,
    by: int,
    theta: int = 0,
) -> tuple[tuple[int, int], ...]:
    """テーブルの北・東・南・西4辺に隣接する全セルを返す。

    専用の受付・出入口は設けず、各辺のどの隣接セルからも着席・離席できる。
    ``theta`` はAPI互換のため受け取るが、テーブルの形状・アクセスには使わない。
    """
    spec = cfg.booths[booth_id]
    if not spec.is_table:
        return ()
    width, height = spec.width, spec.height
    cells = [
        *((x, by - 1) for x in range(bx, bx + width)),
        *((bx + width, y) for y in range(by, by + height)),
        *((x, by + height) for x in range(bx, bx + width)),
        *((bx - 1, y) for y in range(by, by + height)),
    ]
    return tuple(dict.fromkeys(cells))


def position_bounds(
    cfg: Config,
    booth_id: int,
    theta: int,
) -> tuple[int, int, int, int]:
    """指定角度で本体・受付・退出・待機列が場内に収まる座標範囲。"""
    cells = booth_cells_of(
        cfg,
        booth_id,
        0,
        0,
        theta,
    )

    for aux in (
        reception_cell(
            cfg,
            booth_id,
            0,
            0,
            theta,
        ),
        booth_exit_cell(
            cfg,
            booth_id,
            0,
            0,
            theta,
        ),
    ):
        if aux is not None:
            cells.append(aux)

    cells += list(
        queue_cells_of(
            cfg,
            booth_id,
            0,
            0,
            theta,
        )
    )
    cells += list(table_access_cells_of(cfg, booth_id, 0, 0, theta))

    xs = [
        x
        for x, _ in cells
    ]

    ys = [
        y
        for _, y in cells
    ]

    return (
        -min(xs),
        cfg.grid_width - 1 - max(xs),
        -min(ys),
        cfg.grid_height - 1 - max(ys),
    )


def passable_corridor_mask(
    raw_walkable: np.ndarray,
    min_width: int,
) -> np.ndarray:
    """最低幅 ``min_width`` を満たすセルだけを通行可能として返す。

    あるセルを人が歩けるのは、そのセルが少なくとも1つの
    ``min_width × min_width`` 全通行可能ブロックに含まれる場合だけとする。
    したがって、広い空間同士を結ぶ隙間が3セル以下なら、A*/BFSからも
    除外され、エージェントが細い隙間をすり抜けることはない。
    """
    raw = np.asarray(raw_walkable, dtype=bool)
    if raw.ndim != 2:
        raise ValueError("raw_walkable must be a 2-D array")
    width = int(min_width)
    if width < 1:
        raise ValueError("min_width must be at least 1")
    if width == 1:
        return raw.copy()

    height, grid_width = raw.shape
    if height < width or grid_width < width:
        return np.zeros_like(raw)

    block_height = height - width + 1
    block_width = grid_width - width + 1
    full_blocks = np.ones((block_height, block_width), dtype=bool)
    for dy in range(width):
        for dx in range(width):
            full_blocks &= raw[
                dy : dy + block_height,
                dx : dx + block_width,
            ]

    passable = np.zeros_like(raw)
    for dy in range(width):
        for dx in range(width):
            passable[
                dy : dy + block_height,
                dx : dx + block_width,
            ] |= full_blocks
    return passable


def _reachable_cells(mask: np.ndarray, sources) -> np.ndarray:
    """4近傍でsourcesから到達できるセルをbool配列で返す。"""

    height, width = mask.shape
    reached = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x_pos, y_pos in sources:
        x_pos, y_pos = int(x_pos), int(y_pos)
        if (
            0 <= x_pos < width
            and 0 <= y_pos < height
            and mask[y_pos, x_pos]
            and not reached[y_pos, x_pos]
        ):
            reached[y_pos, x_pos] = True
            queue.append((x_pos, y_pos))
    while queue:
        x_pos, y_pos = queue.popleft()
        for next_x, next_y in (
            (x_pos + 1, y_pos),
            (x_pos - 1, y_pos),
            (x_pos, y_pos + 1),
            (x_pos, y_pos - 1),
        ):
            if (
                0 <= next_x < width
                and 0 <= next_y < height
                and mask[next_y, next_x]
                and not reached[next_y, next_x]
            ):
                reached[next_y, next_x] = True
                queue.append((next_x, next_y))
    return reached


def _bfs_tree(
    walkable: np.ndarray,
    sources,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """指定マスク上の距離と最短路復元用の親座標を作る。"""

    height, width = walkable.shape
    distance = np.full((height, width), -1, dtype=np.int32)
    parent_x = np.full((height, width), -1, dtype=np.int32)
    parent_y = np.full((height, width), -1, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()
    for x_pos, y_pos in sources:
        x_pos, y_pos = int(x_pos), int(y_pos)
        if (
            0 <= x_pos < width
            and 0 <= y_pos < height
            and walkable[y_pos, x_pos]
            and distance[y_pos, x_pos] < 0
        ):
            distance[y_pos, x_pos] = 0
            parent_x[y_pos, x_pos] = x_pos
            parent_y[y_pos, x_pos] = y_pos
            queue.append((x_pos, y_pos))
    while queue:
        x_pos, y_pos = queue.popleft()
        next_distance = int(distance[y_pos, x_pos]) + 1
        for next_x, next_y in (
            (x_pos + 1, y_pos),
            (x_pos - 1, y_pos),
            (x_pos, y_pos + 1),
            (x_pos, y_pos - 1),
        ):
            if (
                0 <= next_x < width
                and 0 <= next_y < height
                and walkable[next_y, next_x]
                and distance[next_y, next_x] < 0
            ):
                distance[next_y, next_x] = next_distance
                parent_x[next_y, next_x] = x_pos
                parent_y[next_y, next_x] = y_pos
                queue.append((next_x, next_y))
    return distance, parent_x, parent_y


def augment_walkable_with_hierarchical_widths(
    raw_walkable: np.ndarray,
    entrance_cells,
    exit_cells,
    receptions,
    booth_exits,
    table_access_cells,
) -> np.ndarray:
    """各対象へ4→3→2→1セルの順で最も広い到達経路を割り当てる。"""

    raw = np.asarray(raw_walkable, dtype=bool)
    widths = tuple(range(TARGET_CORRIDOR_WIDTH_CELLS, 0, -1))
    masks = {
        width: passable_corridor_mask(raw, width)
        for width in widths
    }
    entrance_trees = {
        width: _bfs_tree(masks[width], entrance_cells)
        for width in widths
    }
    exit_trees = {
        width: _bfs_tree(masks[width], exit_cells)
        for width in widths
    }
    walkable = masks[TARGET_CORRIDOR_WIDTH_CELLS].copy()

    def add_widest_path(target, trees) -> int:
        """targetへ到達できる最も広い経路をwalkableへ追加し、その幅を返す。"""

        target_x, target_y = int(target[0]), int(target[1])
        selected_width = 0
        selected_tree = None
        for width in widths:
            tree = trees[width]
            if tree[0][target_y, target_x] >= 0:
                selected_width = width
                selected_tree = tree
                break
        if selected_tree is None:
            return 0
        if selected_width == TARGET_CORRIDOR_WIDTH_CELLS:
            return selected_width
        _, parent_x, parent_y = selected_tree
        x_pos, y_pos = target_x, target_y
        while True:
            walkable[y_pos, x_pos] = True
            next_x = int(parent_x[y_pos, x_pos])
            next_y = int(parent_y[y_pos, x_pos])
            if next_x == x_pos and next_y == y_pos:
                break
            if next_x < 0 or next_y < 0:
                break
            x_pos, y_pos = next_x, next_y
        return selected_width

    # 入口から少なくとも1つの会場出口へ、可能な限り広い経路を確保する。
    best_exit = None
    best_exit_width = 0
    best_exit_distance = np.iinfo(np.int32).max
    for exit_cell in exit_cells:
        exit_x, exit_y = int(exit_cell[0]), int(exit_cell[1])
        for width in widths:
            distance = int(entrance_trees[width][0][exit_y, exit_x])
            if distance >= 0:
                if width > best_exit_width or (
                    width == best_exit_width and distance < best_exit_distance
                ):
                    best_exit = (exit_x, exit_y)
                    best_exit_width = width
                    best_exit_distance = distance
                break
    if best_exit is not None:
        add_widest_path(best_exit, entrance_trees)

    for cell in receptions:
        if cell is not None:
            add_widest_path(cell, entrance_trees)

    for accesses in table_access_cells:
        for cell in accesses:
            add_widest_path(cell, entrance_trees)

    for cell in booth_exits:
        if cell is not None:
            add_widest_path(cell, exit_trees)

    return walkable


def apply_layout(
    cfg: Config,
    venue: Venue,
    positions: tuple,
    include_narrow_fallbacks: bool = True,
) -> Layout:
    """有効な ``((x, y, theta), ...)`` 配置をグリッドへ適用する。"""
    grid = venue.base_grid.copy()
    capacity = venue.base_capacity.copy()

    receptions = []
    booth_exits = []
    queue_cells = []
    table_access_cells = []

    for bid, (
        bx,
        by,
        theta,
    ) in enumerate(positions):

        for x, y in booth_cells_of(
            cfg,
            bid,
            bx,
            by,
            theta,
        ):
            grid[y, x] = BOOTH

        receptions.append(
            reception_cell(
                cfg,
                bid,
                bx,
                by,
                theta,
            )
        )

        booth_exits.append(
            booth_exit_cell(
                cfg,
                bid,
                bx,
                by,
                theta,
            )
        )

        qcells = queue_cells_of(
            cfg,
            bid,
            bx,
            by,
            theta,
        )

        queue_cells.append(qcells)
        table_access_cells.append(
            table_access_cells_of(cfg, bid, bx, by, theta)
        )

        for qx, qy in qcells:
            grid[qy, qx] = QUEUE
            capacity[qy, qx] = cfg.cap_queue

    raw_walkable = np.isin(
        grid,
        WALKABLE_TYPES,
    )

    if not cfg.queue_walkable:
        raw_walkable[grid == QUEUE] = False

    four_cell_walkable = passable_corridor_mask(
        raw_walkable,
        TARGET_CORRIDOR_WIDTH_CELLS,
    )
    # シミュレーションは4セル幅を基本とし、対象ごとに3→2→1セル幅へ
    # 段階的に下げた最短経路を追加する。制約計算時は経路復元を省略できる。
    walkable = (
        augment_walkable_with_hierarchical_widths(
            raw_walkable,
            venue.entrance_cells,
            venue.exit_cells,
            receptions,
            booth_exits,
            table_access_cells,
        )
        if include_narrow_fallbacks
        else four_cell_walkable
    )

    return Layout(
        grid=grid,
        capacity=capacity,
        walkable=walkable,
        positions=tuple(
            tuple(p)
            for p in positions
        ),
        receptions=tuple(receptions),
        booth_exits=tuple(booth_exits),
        queue_cells=tuple(queue_cells),
        table_access_cells=tuple(table_access_cells),
        raw_walkable=raw_walkable,
    )
