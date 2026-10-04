"""§23.4: 再現性、§23.6: トレードオフ確認(フェーズ4 ゲート条件)。"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np

from src.parallel import evaluate_layout, evaluate_vector, init_worker
from src.visitors import generate_visitor_table
from tests.conftest import CENTRAL_LAYOUT, PERIMETER_LAYOUT


def _flat(positions) -> np.ndarray:
    return np.array([c for p in positions for c in p], dtype=float)


def test_same_layout_same_result(cfg, ctx):
    """同じ配置・設定・シードで、目的関数と KPI が完全に一致する。"""
    r1 = evaluate_layout(ctx, CENTRAL_LAYOUT)
    r2 = evaluate_layout(ctx, CENTRAL_LAYOUT)
    assert r1.f_move == r2.f_move
    assert r1.f_congestion == r2.f_congestion
    assert r1.kpis == r2.kpis


def test_visitor_table_layout_independent(cfg):
    """来場者テーブル(属性・入場時刻・興味)は配置に依存せず完全に一致する。

    テーブル生成は配置を引数に取らない設計だが、複数回生成しても
    同一であること(系統1乱数の決定性)を確認する。
    """
    t1 = generate_visitor_table(cfg)
    t2 = generate_visitor_table(cfg)
    assert t1 == t2
    assert len(t1) == cfg.n_visitors


def test_serial_equals_parallel(cfg):
    """並列実行と直列実行で結果が完全に一致する(§16.5)。"""
    xs = [_flat(CENTRAL_LAYOUT), _flat(PERIMETER_LAYOUT)]
    serial = [evaluate_vector(x, cfg) for x in xs]
    with Pool(2, initializer=init_worker, initargs=(cfg,)) as pool:
        parallel = pool.starmap(evaluate_vector, [(x, cfg) for x in xs])
    for s, p in zip(serial, parallel):
        assert s[0] == p[0] and s[1] == p[1] and s[2] == p[2]
        assert s[3] == p[3]


def test_distinct_layouts_produce_distinct_objectives(cfg, ctx):
    """異なる有効配置が移動負担・混雑度の双方で識別される。"""
    central = evaluate_layout(ctx, CENTRAL_LAYOUT)
    perimeter = evaluate_layout(ctx, PERIMETER_LAYOUT)
    assert central.f_move != perimeter.f_move
    assert central.f_congestion != perimeter.f_congestion
