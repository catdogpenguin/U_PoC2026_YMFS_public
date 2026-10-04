"""routing.py — BFS 静的距離場・混雑考慮 A*(§9)。

Config で指定された 2 次元グリッドを collections.deque と heapq で探索する。
移動は上下左右の 4 近傍(§3.5)。近傍展開順は固定し、決定性を保証する。
"""
from __future__ import annotations

import heapq
from collections import deque

import numpy as np

from .models import INF_DIST

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - requirements.txt導入前の安全な後方互換
    NUMBA_AVAILABLE = False

    def njit(*_args, **_kwargs):
        def decorate(func):
            return func

        return decorate

#: 4 近傍の展開順(固定): 右, 左, 下, 上
NEIGHBORS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bfs_distance_field_python(walkable: np.ndarray, sources) -> np.ndarray:
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


@njit(cache=True)
def _bfs_distance_field_kernel(
    walkable: np.ndarray,
    sources_xy: np.ndarray,
    inf_dist: int,
) -> np.ndarray:
    """Python版と同じFIFO・近傍順を使うJIT BFS。"""
    H, W = walkable.shape
    dist = np.full((H, W), inf_dist, dtype=np.int32)
    qx = np.empty(H * W, dtype=np.int32)
    qy = np.empty(H * W, dtype=np.int32)
    head = 0
    tail = 0

    for i in range(sources_xy.shape[0]):
        x = int(sources_xy[i, 0])
        y = int(sources_xy[i, 1])
        if 0 <= x < W and 0 <= y < H and walkable[y, x] and dist[y, x] != 0:
            dist[y, x] = 0
            qx[tail] = x
            qy[tail] = y
            tail += 1

    dxs = (1, -1, 0, 0)
    dys = (0, 0, 1, -1)
    while head < tail:
        x = int(qx[head])
        y = int(qy[head])
        head += 1
        d = int(dist[y, x]) + 1
        for k in range(4):
            nx = x + dxs[k]
            ny = y + dys[k]
            if (
                0 <= nx < W
                and 0 <= ny < H
                and walkable[ny, nx]
                and dist[ny, nx] > d
            ):
                dist[ny, nx] = d
                qx[tail] = nx
                qy[tail] = ny
                tail += 1
    return dist


def bfs_distance_field(walkable: np.ndarray, sources) -> np.ndarray:
    """静的BFS距離場。JIT版もPython版と同じ整数距離を返す。"""
    source_tuple = tuple((int(x), int(y)) for x, y in sources)
    if not NUMBA_AVAILABLE:
        return _bfs_distance_field_python(walkable, source_tuple)
    source_array = np.asarray(source_tuple, dtype=np.int32).reshape(-1, 2)
    return _bfs_distance_field_kernel(
        np.ascontiguousarray(walkable, dtype=np.bool_),
        source_array,
        int(INF_DIST),
    )


def _astar_python(
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


@njit(cache=True)
def _astar_kernel(
    walkable: np.ndarray,
    occupancy: np.ndarray,
    capacity: np.ndarray,
    sx: int,
    sy: int,
    lam: float,
    heuristic_field: np.ndarray,
    inf_dist: int,
) -> tuple[np.ndarray, bool]:
    """Python版A*と同じ辞書順・浮動小数点式を平坦配列で実行する。"""
    H, W = walkable.shape
    start_idx = sy * W + sx
    h0 = float(heuristic_field[sy, sx])
    if h0 >= float(inf_dist):
        return np.empty(0, dtype=np.int32), False
    if heuristic_field[sy, sx] == 0:
        return np.empty(0, dtype=np.int32), True

    n_cells = H * W
    step_cost = np.empty(n_cells, dtype=np.float64)
    for idx in range(n_cells):
        y = idx // W
        x = idx - y * W
        ratio = float(occupancy[y, x]) / float(capacity[y, x])
        step_cost[idx] = 1.0 + lam * ratio * ratio

    g_score = np.full(n_cells, np.inf, dtype=np.float64)
    came = np.full(n_cells, -1, dtype=np.int32)
    closed = np.zeros(n_cells, dtype=np.uint8)
    g_score[start_idx] = 0.0

    # counterはPython版と同じ挿入順タイブレーク。各pushで一意なので、
    # tupleの第3要素が比較されることはない。
    heap = [(h0, 0, start_idx)]
    counter = 0
    goal_idx = -1
    dxs = (1, -1, 0, 0)
    dys = (0, 0, 1, -1)

    while heap:
        _f, _order, idx = heapq.heappop(heap)
        if closed[idx] != 0:
            continue
        closed[idx] = 1
        y = idx // W
        x = idx - y * W
        if heuristic_field[y, x] == 0:
            goal_idx = idx
            break

        g = g_score[idx]
        for k in range(4):
            nx = x + dxs[k]
            ny = y + dys[k]
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            nidx = ny * W + nx
            if not walkable[ny, nx] or closed[nidx] != 0:
                continue
            h = float(heuristic_field[ny, nx])
            if h >= float(inf_dist):
                continue
            ng = g + step_cost[nidx]
            if ng < g_score[nidx]:
                g_score[nidx] = ng
                came[nidx] = idx
                counter += 1
                heapq.heappush(heap, (ng + h, counter, nidx))

    if goal_idx < 0:
        return np.empty(0, dtype=np.int32), False

    length = 0
    cur = goal_idx
    while cur != start_idx:
        cur = int(came[cur])
        if cur < 0:
            return np.empty(0, dtype=np.int32), False
        length += 1

    path = np.empty(length, dtype=np.int32)
    cur = goal_idx
    for i in range(length - 1, -1, -1):
        path[i] = cur
        cur = int(came[cur])
    return path, True


def astar(
    walkable: np.ndarray,
    occupancy: np.ndarray,
    capacity: np.ndarray,
    start: tuple,
    goal_cells: set,
    lam: float,
    heuristic_field: np.ndarray,
) -> list | None:
    """混雑考慮A*。JIT版は元実装と同じ経路を返す。"""
    if not NUMBA_AVAILABLE:
        return _astar_python(
            walkable,
            occupancy,
            capacity,
            start,
            goal_cells,
            lam,
            heuristic_field,
        )
    if start in goal_cells:
        return []
    sx, sy = (int(start[0]), int(start[1]))
    flat_path, found = _astar_kernel(
        np.ascontiguousarray(walkable, dtype=np.bool_),
        np.ascontiguousarray(occupancy),
        np.ascontiguousarray(capacity),
        sx,
        sy,
        float(lam),
        np.ascontiguousarray(heuristic_field, dtype=np.int32),
        int(INF_DIST),
    )
    if not found:
        return None
    W = int(walkable.shape[1])
    return [(int(idx % W), int(idx // W)) for idx in flat_path]


def _descend_path_python(dist_field: np.ndarray, start: tuple) -> list:
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


@njit(cache=True)
def _descend_path_indices_kernel(
    dist_field: np.ndarray,
    sx: int,
    sy: int,
    inf_dist: int,
) -> np.ndarray:
    H, W = dist_field.shape
    if dist_field[sy, sx] >= inf_dist:
        return np.empty(0, dtype=np.int64)
    length = int(dist_field[sy, sx]) + 1
    cells = np.empty(length, dtype=np.int64)
    x = sx
    y = sy
    cells[0] = y * W + x
    dxs = (1, -1, 0, 0)
    dys = (0, 0, 1, -1)
    for i in range(1, length):
        best_x = x
        best_y = y
        best_d = int(dist_field[y, x])
        for k in range(4):
            nx = x + dxs[k]
            ny = y + dys[k]
            if 0 <= nx < W and 0 <= ny < H and dist_field[ny, nx] < best_d:
                best_d = int(dist_field[ny, nx])
                best_x = nx
                best_y = ny
        if best_x == x and best_y == y:
            return cells[:i]
        x = best_x
        y = best_y
        cells[i] = y * W + x
    return cells


def descend_path_indices(dist_field: np.ndarray, start: tuple) -> np.ndarray:
    """決定的最短経路を1次元セル番号で返す高速版。"""
    if not NUMBA_AVAILABLE:
        cells = _descend_path_python(dist_field, start)
        W = int(dist_field.shape[1])
        return np.fromiter(
            (int(y) * W + int(x) for x, y in cells),
            dtype=np.intp,
            count=len(cells),
        )
    return _descend_path_indices_kernel(
        np.ascontiguousarray(dist_field, dtype=np.int32),
        int(start[0]),
        int(start[1]),
        int(INF_DIST),
    ).astype(np.intp, copy=False)


def descend_path(dist_field: np.ndarray, start: tuple) -> list:
    """互換API。内部では同じ経路の平坦インデックス版を使用する。"""
    if not NUMBA_AVAILABLE:
        return _descend_path_python(dist_field, start)
    flat = descend_path_indices(dist_field, start)
    W = int(dist_field.shape[1])
    return [(int(idx % W), int(idx // W)) for idx in flat]


def warmup_routing_kernels() -> None:
    """Pool生成前にJITを完了し、forkワーカーへ機械語を継承させる。"""
    if not NUMBA_AVAILABLE:
        return
    walkable = np.ones((3, 3), dtype=np.bool_)
    occupancy = np.zeros((3, 3), dtype=np.int16)
    capacity = np.ones((3, 3), dtype=np.int16)
    field = bfs_distance_field(walkable, [(2, 2)])
    astar(walkable, occupancy, capacity, (0, 0), {(2, 2)}, 1.0, field)
    descend_path_indices(field, (0, 0))
