"""constraints.py — 配置制約チェックと違反量 g(§14)。

違反量の定義(§14【v2】):
- 項目 1〜5(幾何): 違反セル数を合算(重み 1)
- 項目 6〜7(BFS 到達性): 違反件数 × 100
- 項目 8(最低通路幅): 違反セル数 × 1(幾何量として扱う。実装注記)
有効配置は g = 0。無効配置はシミュレーションを実行しない。
"""
from __future__ import annotations

import numpy as np

from .models import EMPTY, FORBIDDEN, WALKABLE_TYPES, Config
from .routing import bfs_distance_field, INF_DIST
from .venue import (
    Venue,
    apply_layout,
    booth_cells_of,
    booth_exit_cell,
    queue_cells_of,
    reception_cell,
)

REACHABILITY_PENALTY = 100.0


def to_integer_layout(x: np.ndarray) -> tuple[tuple[int, int], ...]:
    """整数の設計ベクトルを ((x1, y1), ..., (xn, yn)) に変換する。

    小数を暗黙に丸めない。
    小数成分が含まれていれば、最適化演算子の整数化設定が
    適用されていないものとして ValueError を発生させる。
    """
    values = np.asarray(x)

    if values.size % 2 != 0:
        raise ValueError(
            f"design vector length must be even: {values.size}"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "design vector contains non-finite values"
        )

    rounded = np.rint(values)

    # 3.0 のような整数値の浮動小数点表現は許可するが、
    # 3.2 のような実際の小数値は許可しない。
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

    pairs = rounded.astype(
        np.int64,
        copy=False,
    ).reshape(-1, 2)

    return tuple(
        (int(x_pos), int(y_pos))
        for x_pos, y_pos in pairs
    )


def _in_bounds(cfg: Config, x: int, y: int) -> bool:
    return 0 <= x < cfg.grid_width and 0 <= y < cfg.grid_height


def compute_violation(cfg: Config, venue: Venue, positions: tuple) -> float:
    """配置の制約違反量 g を返す(§14)。g == 0 なら有効配置。"""
    H, W = cfg.grid_height, cfg.grid_width
    g = 0.0
    n_booths = len(cfg.booths)

    # 各ブースの占有セル・受付・行列セルを収集
    booth_cells = [booth_cells_of(cfg, b, *positions[b]) for b in range(n_booths)]
    receptions = [reception_cell(cfg, b, *positions[b]) for b in range(n_booths)]
    booth_exits = [booth_exit_cell(cfg, b, *positions[b]) for b in range(n_booths)]
    qcells = [queue_cells_of(cfg, b, *positions[b]) for b in range(n_booths)]

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
        claimed = set(booth_cells[b])
        claimed.add(receptions[b])
        claimed.add(booth_exits[b])
        claimed.update(qcells[b])
        g += claim(claimed)

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
        rx, ry = receptions[b]
        if _in_bounds(cfg, rx, ry) and base[ry, rx] != EMPTY:
            g += 1
        # 行列セル: EMPTY または FORBIDDEN(通行可)を許可(§5.2)
        for (x, y) in qcells[b]:
            if _in_bounds(cfg, x, y) and base[y, x] not in (EMPTY, FORBIDDEN):
                g += 1
        # 体験後退出セル: EMPTY のみ許可。受付共用時の重複検査は上で一意化済み。
        ex, ey = booth_exits[b]
        if _in_bounds(cfg, ex, ey) and base[ey, ex] != EMPTY:
            g += 1

    if g > 0:
        return g  # 幾何違反があれば到達性チェックは行わない(高速化)

    # --- 有効な幾何配置についてグリッドを構築し、項目 6〜8 を検査
    layout = apply_layout(cfg, venue, positions)
    walkable = layout.walkable

    ent_field = bfs_distance_field(walkable, venue.entrance_cells)

    # 項目 6: 入口→出口の経路存在
    if all(ent_field[y, x] >= INF_DIST for (x, y) in venue.exit_cells):
        g += REACHABILITY_PENALTY

    # 項目 7: 入口→各ブース受付の到達性
    for b in range(n_booths):
        rx, ry = receptions[b]
        if ent_field[ry, rx] >= INF_DIST:
            g += REACHABILITY_PENALTY

    # ブース退出セル→会場出口の到達性
    exit_field = bfs_distance_field(walkable, venue.exit_cells)
    for ex, ey in booth_exits:
        if exit_field[ey, ex] >= INF_DIST:
            g += REACHABILITY_PENALTY

    # 項目 8: 最低通路幅(既定 2 セル)
    # 判定: 各通行可能セルが、少なくとも 1 つの「幅 w × w の全通行可能ブロック」に
    # 属していれば合格。属さないセル数を違反セル数として加算する。
    w = cfg.min_corridor_width
    ok_block = np.zeros_like(walkable)
    # w×w ブロックの AND をシフト積で計算
    blk = walkable.copy()
    for _ in range(w - 1):
        blk = blk[:, :-1] & blk[:, 1:]
    for _ in range(w - 1):
        blk = blk[:-1, :] & blk[1:, :]
    # blk[j,i] == True なら (i..i+w-1, j..j+w-1) が全て通行可能
    for j in range(blk.shape[0]):
        for i in range(blk.shape[1]):
            if blk[j, i]:
                ok_block[j : j + w, i : i + w] = True
    narrow = walkable & ~ok_block
    g += float(np.sum(narrow))

    return g


def compute_violation_vector(
    cfg: Config,
    venue: Venue,
    x: np.ndarray,
) -> float:
    """整数の設計ベクトルに対する違反量を返すユーティリティ。"""
    return compute_violation(
        cfg,
        venue,
        to_integer_layout(x),
    )
