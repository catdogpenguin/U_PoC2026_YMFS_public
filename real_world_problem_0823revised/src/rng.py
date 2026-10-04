"""rng.py — 乱数の 3 系統分離ヘルパ(§2.3.2【v2】)。

系統1: 来場者生成(属性・入場時刻・興味)。配置に一切依存しない。
系統2: 来場者ごとの意思決定(softmax 抽選)。visitor_id 単位で独立ストリーム。
系統3: 競合解決のタイブレーク。
系統(サブシード)3 は §15.3 の初期集団サンプリング用に予約。
系統4: ゴミステーション利用選好。既存の来場者属性乱数列を変えない。

グローバルな np.random は一切使用しない(§25)。
"""
from __future__ import annotations

import numpy as np


def visitor_gen_rng(seed: int) -> np.random.Generator:
    """系統1: 来場者テーブル生成用。全配置で完全に同一。"""
    return np.random.default_rng([seed, 0])


def decision_rng(seed: int, visitor_id: int) -> np.random.Generator:
    """系統2: 来場者 visitor_id 専用の意思決定ストリーム。

    他の来場者や配置の違いによる消費順序の影響を受けない。
    """
    return np.random.default_rng([seed, 1, visitor_id])


def tiebreak_rng(seed: int) -> np.random.Generator:
    """系統3: セル移動競合のタイブレーク用。"""
    return np.random.default_rng([seed, 2])


def sampling_rng(seed: int) -> np.random.Generator:
    """§15.3: 初期集団のリジェクションサンプリング用。"""
    return np.random.default_rng([seed, 3])


def waste_behavior_rng(seed: int) -> np.random.Generator:
    """系統4: 通常時・飲食後のゴミステーション利用選好生成用。"""
    return np.random.default_rng([seed, 4])


def softmax_choice(rng: np.random.Generator, candidate_ids: list, probabilities: np.ndarray):
    """1 回の選択で乱数をちょうど 1 個だけ消費する累積和方式の抽選。

    Generator.choice は内部消費数が実装依存のため使わない。これにより
    「来場者 i の k 回目の選択に使う乱数値は全配置で一致する」(§2.3.2)を厳密化する。
    """
    u = rng.random()
    cum = np.cumsum(probabilities)
    idx = int(np.searchsorted(cum, u * cum[-1], side="right"))
    idx = min(idx, len(candidate_ids) - 1)
    return candidate_ids[idx]