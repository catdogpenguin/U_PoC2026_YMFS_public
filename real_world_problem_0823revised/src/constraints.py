"""constraints.py — 配置制約チェックと違反量 g(§14)。

違反量の定義(§14【v2】):
- 項目 1〜5(幾何): 違反セル数を合算(重み 1)
- 項目 6〜7(BFS 到達性): 違反件数 × 100

初期個体生成・遺伝的操作の必須物理制約には1セル幅到達性を使う。
より広い通路の到達性は診断値または受容可能性制約として別に評価し、
物理制約や目的関数より先に置くlexicographic選択には使わない。
有効配置は g = 0。無効配置はシミュレーションを実行しない。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import EMPTY, FORBIDDEN, WALKABLE_TYPES, Config
from .routing import bfs_distance_field, INF_DIST
from .venue import (
    Venue,
    apply_layout,
    booth_cells_of,
    booth_exit_cell,
    front_clearance_cells,
    queue_cells_of,
    reception_cell,
    table_access_cells_of,
    passable_corridor_mask,
    TARGET_CORRIDOR_WIDTH_CELLS,
    VALID_THETAS,
)

REACHABILITY_PENALTY = 100.0


@dataclass(frozen=True)
class PlacementGeometry:
    """1配置から一度だけ構築する、物理制約・整然さ制約共通の幾何情報。"""

    positions: tuple[tuple[int, int, int], ...]
    booth_cells: tuple[tuple[tuple[int, int], ...], ...]
    receptions: tuple[tuple[int, int] | None, ...]
    booth_exits: tuple[tuple[int, int] | None, ...]
    queue_cells: tuple[tuple[tuple[int, int], ...], ...]
    table_access_cells: tuple[tuple[tuple[int, int], ...], ...]
    front_clearance_cells: tuple[tuple[tuple[int, int], ...], ...]
    claimed_cells: tuple[frozenset[tuple[int, int]], ...]


def build_placement_geometry(
    cfg: Config,
    positions: tuple[tuple[int, int, int], ...],
) -> PlacementGeometry:
    """配置物セルを列挙する処理を、制約間で共有できる形で返す。"""

    normalized = tuple(
        (int(x_pos), int(y_pos), int(theta))
        for x_pos, y_pos, theta in positions
    )
    booth_cells = tuple(
        tuple(booth_cells_of(cfg, booth_id, *normalized[booth_id]))
        for booth_id in range(len(cfg.booths))
    )
    receptions = tuple(
        reception_cell(cfg, booth_id, *normalized[booth_id])
        for booth_id in range(len(cfg.booths))
    )
    booth_exits = tuple(
        booth_exit_cell(cfg, booth_id, *normalized[booth_id])
        for booth_id in range(len(cfg.booths))
    )
    queue_cells = tuple(
        tuple(queue_cells_of(cfg, booth_id, *normalized[booth_id]))
        for booth_id in range(len(cfg.booths))
    )
    table_access_cells = tuple(
        tuple(table_access_cells_of(cfg, booth_id, *normalized[booth_id]))
        for booth_id in range(len(cfg.booths))
    )
    front_clearances = tuple(
        tuple(front_clearance_cells(cfg, booth_id, *normalized[booth_id]))
        for booth_id in range(len(cfg.booths))
    )
    claimed_cells = []
    for booth_id in range(len(cfg.booths)):
        claimed = set(booth_cells[booth_id])
        if receptions[booth_id] is not None:
            claimed.add(receptions[booth_id])
        if booth_exits[booth_id] is not None:
            claimed.add(booth_exits[booth_id])
        claimed.update(queue_cells[booth_id])
        claimed.update(table_access_cells[booth_id])
        claimed_cells.append(frozenset(claimed))
    return PlacementGeometry(
        positions=normalized,
        booth_cells=booth_cells,
        receptions=receptions,
        booth_exits=booth_exits,
        queue_cells=queue_cells,
        table_access_cells=table_access_cells,
        front_clearance_cells=front_clearances,
        claimed_cells=tuple(claimed_cells),
    )


def design_variable_count(cfg: Config) -> int:
    """通常ブース3変数、テーブル2変数として総設計変数数を返す。"""
    return sum(2 if booth.is_table else 3 for booth in cfg.booths)


def booth_variable_indices(
    cfg: Config, booth_id: int
) -> tuple[int, int, int | None]:
    """指定配置物の x, y, theta の遺伝子位置を返す。テーブルのthetaはNone。"""
    offset = sum(2 if booth.is_table else 3 for booth in cfg.booths[:booth_id])
    return offset, offset + 1, None if cfg.booths[booth_id].is_table else offset + 2


def positions_to_design(cfg: Config, positions) -> np.ndarray:
    """内部 ``(x,y,theta)`` 配置を、テーブルthetaなしの設計ベクトルへ変換する。"""
    if len(positions) != len(cfg.booths):
        raise ValueError("positions length does not match cfg.booths")
    values: list[int] = []
    for booth, position in zip(cfg.booths, positions):
        x_pos, y_pos, theta = (int(value) for value in position)
        values.extend((x_pos, y_pos))
        if not booth.is_table:
            values.append(theta)
    return np.asarray(values, dtype=np.int64)


def to_integer_layout(
    x: np.ndarray, cfg: Config | None = None
) -> tuple[tuple[int, int, int], ...]:
    """整数設計ベクトルを ``((x1, y1, theta1), ...)`` に変換する。

    cfg指定時、テーブルは ``(x,y)`` の2変数だけを読み、内部thetaを0に固定する。
    通常ブースのthetaは ``0=北, 1=東, 2=南, 3=西`` とする。
    小数は暗黙に丸めず、含まれていれば ValueError を発生させる。
    """
    values = np.asarray(x)

    expected = design_variable_count(cfg) if cfg is not None else None
    if expected is not None and values.size != expected:
        raise ValueError(
            f"design vector length mismatch: expected {expected}, got {values.size}"
        )
    if cfg is None and values.size % 3 != 0:
        raise ValueError(
            "design vector length must be a multiple of 3: "
            f"{values.size}"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "design vector contains non-finite values"
        )

    rounded = np.rint(values)

    if not np.allclose(
        values,
        rounded,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "design vector must contain integers; "
            f"fractional values were found: {values}"
        )

    integers = rounded.astype(np.int64, copy=False).reshape(-1)
    if cfg is not None:
        triples = []
        cursor = 0
        for booth in cfg.booths:
            x_pos, y_pos = int(integers[cursor]), int(integers[cursor + 1])
            cursor += 2
            theta = 0 if booth.is_table else int(integers[cursor])
            cursor += 0 if booth.is_table else 1
            triples.append((x_pos, y_pos, theta))
    else:
        triples = [tuple(int(value) for value in row) for row in integers.reshape(-1, 3)]

    invalid_theta = [
        int(theta)
        for booth, (_, _, theta) in zip(
            cfg.booths if cfg is not None else (None,) * len(triples), triples
        )
        if booth is None or not booth.is_table
        if int(theta) not in VALID_THETAS
    ]

    if invalid_theta:
        raise ValueError(
            f"theta must be one of {VALID_THETAS}: "
            f"{invalid_theta}"
        )

    return tuple(
        (
            int(x_pos),
            int(y_pos),
            int(theta),
        )
        for x_pos, y_pos, theta in triples
    )


def _table_equivalence_key(booth) -> tuple:
    """最適化上で同一視できるテーブル仕様を返す。

    ``name``、``booth_id``、``fixed_position`` はシミュレーション中の挙動を
    変えないため除外する。一方、所属広場や寸法・容量などが異なるテーブルは
    同一視しない。
    """

    return (
        str(booth.area),
        int(booth.width),
        int(booth.height),
        str(booth.category),
        float(booth.attractiveness),
        int(booth.capacity),
        float(booth.mean_service_seconds),
        int(booth.max_queue_length),
        str(booth.kind),
        str(booth.theme),
    )


def equivalent_table_groups(cfg: Config) -> tuple[tuple[int, ...], ...]:
    """同一仕様かつ同一広場に属するテーブルのインデックス群を返す。"""

    grouped: dict[tuple, list[int]] = {}
    for booth_id, booth in enumerate(cfg.booths):
        if not booth.is_table:
            continue
        grouped.setdefault(_table_equivalence_key(booth), []).append(int(booth_id))
    return tuple(
        tuple(sorted(booth_ids))
        for booth_ids in grouped.values()
        if len(booth_ids) >= 2
    )


def tables_are_equivalent(cfg: Config, first: int, second: int) -> bool:
    """2配置物が、入れ替えても意味が変わらない同一テーブルかを返す。"""

    first_booth = cfg.booths[int(first)]
    second_booth = cfg.booths[int(second)]
    return bool(
        first_booth.is_table
        and second_booth.is_table
        and _table_equivalence_key(first_booth)
        == _table_equivalence_key(second_booth)
    )


def canonicalize_table_design(cfg: Config, x: np.ndarray) -> np.ndarray:
    """同一テーブルの座標を行優先順へ並べ、代表遺伝子へ正規化する。

    同一仕様・同一広場のテーブルは識別子を交換しても配置の意味が変わらない。
    各等価群について座標を ``(y, x)`` の辞書順に並べ、番号の小さいテーブルへ
    順に割り当てることで、同じ配置が必ず同じ整数設計ベクトルになるようにする。
    """

    positions = [list(position) for position in to_integer_layout(x, cfg)]
    for booth_ids in equivalent_table_groups(cfg):
        ordered_coordinates = sorted(
            (
                (int(positions[booth_id][0]), int(positions[booth_id][1]))
                for booth_id in booth_ids
            ),
            key=lambda position: (position[1], position[0]),
        )
        for booth_id, (x_pos, y_pos) in zip(booth_ids, ordered_coordinates):
            positions[booth_id] = [int(x_pos), int(y_pos), 0]
    return positions_to_design(cfg, positions)


def _in_bounds(cfg: Config, x: int, y: int) -> bool:
    return 0 <= x < cfg.grid_width and 0 <= y < cfg.grid_height


def compute_violation(
    cfg: Config,
    venue: Venue,
    positions: tuple,
    geometry: PlacementGeometry | None = None,
) -> float:
    """配置の制約違反量 g を返す(§14)。g == 0 なら有効配置。"""
    H, W = cfg.grid_height, cfg.grid_width
    g = 0.0
    n_booths = len(cfg.booths)

    # 各ブースの占有セル・受付・行列セルを収集。直接最適化では整然さ制約と共有する。
    geom = geometry or build_placement_geometry(cfg, positions)
    booth_cells = geom.booth_cells
    receptions = geom.receptions
    booth_exits = geom.booth_exits
    qcells = geom.queue_cells
    table_access_cells = geom.table_access_cells
    front_clearances = geom.front_clearance_cells

    # 使用セル集合(受付+行列): 相互重複チェック用
    claim_count = np.zeros((H, W), dtype=np.int16)  # 場内セルの多重使用数

    def claim(cells) -> int:
        """場外セル数を返しつつ、場内セルの使用回数を加算する。"""
        out = 0
        for (x, y) in cells:
            if _in_bounds(cfg, x, y):
                claim_count[y, x] += 1
            else:
                out += 1
        return out

    # --- 項目 2: 会場外はみ出し(ブース・受付・行列)
    for b in range(n_booths):
        # 20260629 方式では受付が待機列先頭と同じセル。ブース内の重複は
        # 合法なので、ブースごとに一意化してから他ブースとの競合を数える。
        # テーブルは専用出入口を持たず、周囲4辺のアクセスセルを確保する。
        g += claim(geom.claimed_cells[b])

    # --- 項目 1: 重なり(ブース同士・受付・行列の相互重複)
    g += float(np.sum(np.maximum(claim_count - 1, 0)))

    # --- 項目 3〜5: ベースグリッドとの整合
    base = venue.base_grid
    for b in range(n_booths):
        # ブース本体: EMPTY 以外(FORBIDDEN/WALL/入口/出口/周辺)への重なりは違反
        for (x, y) in booth_cells[b]:
            if _in_bounds(cfg, x, y) and base[y, x] != EMPTY:
                g += 1
        # 受付セル: EMPTY のみ許可(§5.1: 配置不可領域も不可)
        if receptions[b] is not None:
            rx, ry = receptions[b]
            if _in_bounds(cfg, rx, ry) and base[ry, rx] != EMPTY:
                g += 1
        # 行列セル: EMPTY または FORBIDDEN(通行可)を許可(§5.2)
        for (x, y) in qcells[b]:
            if _in_bounds(cfg, x, y) and base[y, x] not in (EMPTY, FORBIDDEN):
                g += 1
        # テーブルは周囲4辺の全隣接セルから利用できる。専用出入口ではないため
        # セル種別は変えず、通行可能であることだけを要求する。
        for (x, y) in table_access_cells[b]:
            if _in_bounds(cfg, x, y) and base[y, x] not in WALKABLE_TYPES:
                g += 1
        # 体験後退出セル: EMPTY のみ許可。受付共用時の重複検査は上で一意化済み。
        if booth_exits[b] is not None:
            ex, ey = booth_exits[b]
            if _in_bounds(cfg, ex, ey) and base[ey, ex] != EMPTY:
                g += 1

    # --- 所属広場制約: BoothSpec.area 指定時、本体セルはその広場内に収める
    if venue.area_masks:
        for b in range(n_booths):
            area = cfg.booths[b].area
            mask = venue.area_masks.get(area) if area else None
            if mask is None:
                continue
            for (x, y) in booth_cells[b]:
                if _in_bounds(cfg, x, y) and not mask[y, x]:
                    g += 1

    # --- ブース正面2セル空間: 場内の通行可能領域に置き、他配置物の侵入を禁止
    # 正面空間同士は共有通路として重なってよい。当該ブース自身の受付・退出口・
    # 待機列も許可し、他ブースが使用しているセルだけを違反として数える。
    walkable_types = set(int(value) for value in WALKABLE_TYPES)
    for booth_id, cells in enumerate(front_clearances):
        own_claimed = geom.claimed_cells[booth_id]
        for x_pos, y_pos in cells:
            if not _in_bounds(cfg, x_pos, y_pos):
                g += 1
                continue
            if int(base[y_pos, x_pos]) not in walkable_types:
                g += 1
            other_claims = int(claim_count[y_pos, x_pos]) - int(
                (x_pos, y_pos) in own_claimed
            )
            if other_claims > 0:
                g += other_claims

    if g > 0:
        return g  # 幾何違反があれば到達性チェックは行わない(高速化)

    # --- 有効な幾何配置についてグリッドを構築し、項目 6〜7 を検査
    layout = apply_layout(
        cfg,
        venue,
        positions,
        include_narrow_fallbacks=False,
    )

    # 配置制約と初期個体生成には、4セル幅処理前の従来マスクを使う。
    # layout.walkable はシミュレーション専用で、ここでは参照しない。
    walkable = (
        layout.raw_walkable
        if layout.raw_walkable is not None
        else layout.walkable
    )

    ent_field = bfs_distance_field(walkable, venue.entrance_cells)

    # 項目 6: 入口→出口の経路存在
    if all(ent_field[y, x] >= INF_DIST for (x, y) in venue.exit_cells):
        g += REACHABILITY_PENALTY

    # 項目 7: 入口→各ブース受付の到達性
    for b in range(n_booths):
        if receptions[b] is None:
            continue
        rx, ry = receptions[b]
        if ent_field[ry, rx] >= INF_DIST:
            g += REACHABILITY_PENALTY

    # テーブル周囲4辺の全アクセスセルへ到達できることを要求する。
    for b in range(n_booths):
        for tx, ty in table_access_cells[b]:
            if ent_field[ty, tx] >= INF_DIST:
                g += REACHABILITY_PENALTY

    # ブース退出セル→会場出口の到達性
    exit_field = bfs_distance_field(walkable, venue.exit_cells)
    for cell in booth_exits:
        if cell is None:
            continue
        ex, ey = cell
        if exit_field[ey, ex] >= INF_DIST:
            g += REACHABILITY_PENALTY

    return g


def compute_width_reachability_profile(
    cfg: Config,
    venue: Venue,
    positions: tuple,
    geometry: PlacementGeometry | None = None,
) -> dict[str, object]:
    """各ブースへ到達できる最大通路幅（4/3/2/1）を返す。

    通常ブースは入口側受付と退出側セルの狭い方、テーブルは周囲アクセス
    セルのうち最も広く到達できるセルを、そのブースの到達幅とする。
    """

    geom = geometry or build_placement_geometry(cfg, positions)
    layout = apply_layout(
        cfg,
        venue,
        positions,
        include_narrow_fallbacks=False,
    )
    widths = tuple(range(TARGET_CORRIDOR_WIDTH_CELLS, 0, -1))
    entrance_fields = {}
    exit_fields = {}
    for width in widths:
        mask = passable_corridor_mask(layout.raw_walkable, width)
        entrance_fields[width] = bfs_distance_field(mask, venue.entrance_cells)
        exit_fields[width] = bfs_distance_field(mask, venue.exit_cells)

    def widest(cell, fields) -> int:
        if cell is None:
            return 0
        x_pos, y_pos = cell
        for width in widths:
            if fields[width][y_pos, x_pos] < INF_DIST:
                return width
        return 0

    booth_widths: list[int] = []
    for booth_id, booth in enumerate(cfg.booths):
        if booth.is_table:
            access_widths = [
                widest(cell, entrance_fields)
                for cell in geom.table_access_cells[booth_id]
            ]
            booth_widths.append(max(access_widths, default=0))
        else:
            reception_width = widest(geom.receptions[booth_id], entrance_fields)
            exit_width = widest(geom.booth_exits[booth_id], exit_fields)
            booth_widths.append(min(reception_width, exit_width))

    achieved_counts = {
        width: int(sum(value >= width for value in booth_widths))
        for width in widths
    }
    exact_counts = {
        width: int(sum(value == width for value in booth_widths))
        for width in widths
    }
    return {
        "target_width": int(TARGET_CORRIDOR_WIDTH_CELLS),
        "required_booth_count": int(len(cfg.booths)),
        "booth_widths": tuple(int(value) for value in booth_widths),
        "achieved_counts": achieved_counts,
        "exact_counts": exact_counts,
        "four_cell_achievement_count": int(
            achieved_counts[TARGET_CORRIDOR_WIDTH_CELLS]
        ),
        "all_booths_one_cell_reachable": bool(
            all(value >= 1 for value in booth_widths)
        ),
    }


def compute_wide_reachability_violation(
    cfg: Config,
    venue: Venue,
    positions: tuple,
    geometry: PlacementGeometry | None = None,
) -> float:
    """後方互換用。4セル幅未達ブース数を返す（ハード制約にはしない）。"""

    profile = compute_width_reachability_profile(
        cfg,
        venue,
        positions,
        geometry=geometry,
    )
    return float(
        int(profile["required_booth_count"])
        - int(profile["four_cell_achievement_count"])
    )


def compute_violation_vector(
    cfg: Config,
    venue: Venue,
    x: np.ndarray,
) -> float:
    """整数の設計ベクトルに対する違反量を返すユーティリティ。"""
    return compute_violation(
        cfg,
        venue,
        to_integer_layout(x, cfg),
    )
