"""parallel.py — 1 配置評価の入口とプロセス並列のためのコンテキスト管理(§16)。

静的コンテキスト(会場・来場者テーブル・距離場以外の共有物)はワーカー
プロセスごとに 1 回だけ構築してモジュールグローバルへキャッシュする(§16.3)。
各タスクには 12 次元の配置ベクトル(または整数配置)だけを渡す。
乱数は §2.3.2 の 3 系統方式のため、並列・直列で結果が完全に一致する(§16.5)。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .constraints import compute_violation, to_integer_layout
from .models import Config
from .simulator import SimResult, Simulation
from .venue import Venue, apply_layout, build_base_venue
from .visitors import generate_visitor_table

#: 無効・タイムアウト配置に与える目的関数値(§17.6 の代替として定数を使用)
PENALTY_OBJECTIVE = 1.0e6


@dataclass(frozen=True)
class StaticContext:
    """全評価で共有する静的データ(配置に依存しない)。"""

    cfg: Config
    venue: Venue
    visitor_records: tuple


def build_context(cfg: Config) -> StaticContext:
    return StaticContext(
        cfg=cfg,
        venue=build_base_venue(cfg),
        visitor_records=generate_visitor_table(cfg),
    )


# --- ワーカープロセス内キャッシュ(プロセスごとに 1 回だけ構築)
_CTX: StaticContext | None = None


def init_worker(cfg: Config) -> None:
    """multiprocessing.Pool の initializer。"""
    global _CTX
    _CTX = build_context(cfg)


def get_context(cfg: Config) -> StaticContext:
    """キャッシュ済みコンテキストを返す(未初期化なら構築)。"""
    global _CTX
    if _CTX is None or _CTX.cfg is not cfg and _CTX.cfg != cfg:
        _CTX = build_context(cfg)
    return _CTX


def evaluate_layout(
    ctx: StaticContext,
    positions: tuple,
    collect_heatmap: bool = False,
    record_frames: bool = False,
    frame_interval: int = 1,
) -> SimResult:
    """有効配置を 1 回シミュレーションする(§2.3: 1 配置 1 シミュレーション)。"""
    layout = apply_layout(ctx.cfg, ctx.venue, positions)
    sim = Simulation(
        ctx.cfg,
        ctx.venue,
        layout,
        ctx.visitor_records,
        collect_heatmap=collect_heatmap,
        record_frames=record_frames,
        frame_interval=frame_interval,
    )
    return sim.run()


def evaluate_vector(
    x: np.ndarray,
    cfg: Config,
) -> tuple[float, float, float, dict]:
    """整数の配置ベクトルを評価する。

    g > 0 の配置はシミュレーションを実行せず、
    ペナルティ目的関数値を返す。
    """
    ctx = get_context(cfg)

    # 小数を暗黙に丸めず、整数であることを検証する。
    positions = to_integer_layout(x)

    g = compute_violation(
        cfg,
        ctx.venue,
        positions,
    )

    if g > 0:
        return (
            PENALTY_OBJECTIVE,
            PENALTY_OBJECTIVE,
            g,
            {},
        )

    res = evaluate_layout(
        ctx,
        positions,
    )

    if res.timed_out:
        return (
            PENALTY_OBJECTIVE,
            PENALTY_OBJECTIVE,
            0.0,
            res.kpis,
        )

    return (
        res.f_move,
        res.f_congestion,
        0.0,
        res.kpis,
    )


def default_workers() -> int:
    """§16.4: 並列数の既定値。"""
    return max(1, (os.cpu_count() or 2) - 1)
