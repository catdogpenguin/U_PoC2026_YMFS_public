"""routing.py — BFS 静的距離場・混雑考慮 A*(§9)。

Config で指定された 2 次元グリッドを collections.deque と heapq で探索する。
移動は上下左右の 4 近傍(§3.5)。近傍展開順は固定し、決定性を保証する。
"""
from __future__ import annotations

import heapq
from collections import deque

import numpy as np

from .models import INF_DIST

#: 4 近傍の展開順(固定): 右, 左, 下, 上
NEIGHBORS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def bfs_distance_field(walkable: np.ndarray, sources) -> np.ndarray:
    """複数ソースからの BFS 距離場(§9.4 静的距離場)。

    Args:
        walkable: bool (H, W)
        sources: iterable of (x, y)
    Returns:
        int32 (H, W)。到達不能は INF_DIST。
    """
    H, W = walkable.shape
    dist = np.full((H, W), INF_DIST, dtype=np.int32)
    q: deque = deque()
    for (x, y) in sources:
        if 0 <= x < W and 0 <= y < H and walkable[y, x]:
            dist[y, x] = 0
            q.append((x, y))
    while q:
        x, y = q.popleft()
        d = dist[y, x] + 1
        for dx, dy in NEIGHBORS_4:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and walkable[ny, nx] and dist[ny, nx] > d:
                dist[ny, nx] = d
                q.append((nx, ny))
    return dist


def astar(
    walkable: np.ndarray,
    occupancy: np.ndarray,
    capacity: np.ndarray,
    start: tuple,
    goal_cells: set,
    lam: float,
    heuristic_field: np.ndarray,
) -> list | None:
    """混雑を考慮した A*(§9.2)。

    セル s へ進入するコスト: c = 1 + λ (n_s / C_s)^2。
    n_s は経路計画時点の占有スナップショット(計画後は再計画まで固定)。
    ヒューリスティックは静的距離場(コスト >= 1/セルなので許容的)。

    Returns:
        start を含まない次セルからの経路 list[(x,y)]。到達不能なら None。
        start がゴールなら空リスト。
    """
    if start in goal_cells:
        return []
    H, W = walkable.shape
    sx, sy = start
    ratio = occupancy.astype(np.float64) / capacity  # スナップショット
    step_cost = 1.0 + lam * ratio * ratio

    g_score = np.full((H, W), np.inf)
    g_score[sy, sx] = 0.0
    came: dict = {}
    counter = 0
    h0 = float(heuristic_field[sy, sx])
    if h0 >= float(INF_DIST):
        return None
    heap = [(h0, counter, sx, sy)]
    closed = np.zeros((H, W), dtype=bool)
    while heap:
        f, _, x, y = heapq.heappop(heap)
        if closed[y, x]:
            continue
        closed[y, x] = True
        if (x, y) in goal_cells:
            # 経路復元
            path = []
            cur = (x, y)
            while cur != (sx, sy):
                path.append(cur)
                cur = came[cur]
            path.reverse()
            return path
        g = g_score[y, x]
        for dx, dy in NEIGHBORS_4:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H) or not walkable[ny, nx] or closed[ny, nx]:
                continue
            h = float(heuristic_field[ny, nx])
            if h >= float(INF_DIST):
                continue
            ng = g + step_cost[ny, nx]
            if ng < g_score[ny, nx]:
                g_score[ny, nx] = ng
                came[(nx, ny)] = (x, y)
                counter += 1
                heapq.heappush(heap, (ng + h, counter, nx, ny))
    return None


def descend_path(dist_field: np.ndarray, start: tuple) -> list:
    """静的距離場の勾配降下による静的最短経路のセル列(start を含む)。

    §8 の C̃(静的最短経路上の占有率平均)の算出に使用。
    """
    H, W = dist_field.shape
    x, y = start
    if dist_field[y, x] >= INF_DIST:
        return []
    cells = [(x, y)]
    while dist_field[y, x] > 0:
        best = None
        bd = dist_field[y, x]
        for dx, dy in NEIGHBORS_4:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and dist_field[ny, nx] < bd:
                bd = dist_field[ny, nx]
                best = (nx, ny)
        if best is None:  # 局所停留(理論上発生しない)
            break
        x, y = best
        cells.append(best)
    return cells
