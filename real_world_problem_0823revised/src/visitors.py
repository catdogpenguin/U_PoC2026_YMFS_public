"""visitors.py — 来場者テーブルの決定的生成(§7【v2】)。

系統1乱数のみを使用し、配置に一切依存しない。シミュレーション開始前に
全来場者分を一括生成してテーブル化する。全配置で完全に同一(§2.3.2)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Config, Visitor, NOT_ARRIVED
from .rng import visitor_gen_rng, waste_behavior_rng


@dataclass(frozen=True)
class VisitorRecord:
    """来場者テーブルの 1 行(不変)。"""

    visitor_id: int
    spawn_time_step: int
    profile_name: str
    speed_mps: float
    congestion_avoidance: float
    queue_tolerance_seconds: float
    max_visit_seconds: float
    distance_weight_factor: float
    interests: tuple  # カテゴリ別興味 [0,1]
    table_after_food_propensity: float = 1.0  # 配置に依存しない着席選好 U(0,1)
    general_waste_propensity: float = 1.0  # 通常時のゴミステーション利用選好
    food_waste_propensity: float = 1.0  # 飲食後のゴミステーション利用選好
    is_passerby: bool = False  # ブースへ行かず会場を通り抜けるだけの通行人


def generate_visitor_table(cfg: Config) -> tuple:
    """§7 の決定的手順で来場者テーブルを生成する。

    1. 区間重み w_k = rate_k × (end_k − start_k)
    2. 区間人数 n_k = round(N × w_k / Σw)(端数は最終区間で調整)
    3. 区間内は等間隔基準時刻 ± (間隔/2) の一様ジッター(系統1乱数)
    4. (入場時刻, プロファイル, 興味) を一括生成し固定
    """
    rng = visitor_gen_rng(cfg.evaluation_seed)
    # 既存の到着・プロファイル・興味・テーブル選好を変えないよう、
    # ゴミステーション用の乱数は独立した系統4から生成する。
    waste_rng = waste_behavior_rng(cfg.evaluation_seed)
    n = cfg.n_visitors

    # --- 入場時刻
    weights = [rate * (end - start) for (start, end, rate) in cfg.arrival_profile]
    total_w = sum(weights)
    counts = [int(round(n * w / total_w)) for w in weights]
    counts[-1] += n - sum(counts)  # 端数調整で合計を N に一致
    arrival_seconds: list[float] = []
    for (start, end, _), nk in zip(cfg.arrival_profile, counts):
        if nk <= 0:
            continue
        spacing = (end - start) / nk
        for j in range(nk):
            base = start + (j + 0.5) * spacing
            jitter = rng.uniform(-spacing / 2.0, spacing / 2.0)
            arrival_seconds.append(min(max(base + jitter, start), end - 1e-9))
    arrival_seconds.sort()

    # --- プロファイル・興味
    ratios = np.array([p.ratio for p in cfg.profiles], dtype=float)
    cum = np.cumsum(ratios / ratios.sum())
    n_cat = len(cfg.categories)

    records = []
    for vid, t_sec in enumerate(arrival_seconds):
        u = rng.random()
        p_idx = int(np.searchsorted(cum, u, side="right"))
        p_idx = min(p_idx, len(cfg.profiles) - 1)
        prof = cfg.profiles[p_idx]
        if prof.interest_mode == "focused":  # 目的志向: 特定カテゴリ興味 高
            fav = int(rng.integers(n_cat))
            interests = rng.uniform(0.0, 0.4, size=n_cat)
            interests[fav] = rng.uniform(0.7, 1.0)
        elif prof.interest_mode == "dispersed":  # 回遊型: 興味分散
            interests = rng.uniform(0.4, 0.8, size=n_cat)
        else:
            interests = rng.uniform(0.0, 1.0, size=n_cat)
        records.append(
            VisitorRecord(
                visitor_id=vid,
                spawn_time_step=int(t_sec / cfg.dt_seconds),
                profile_name=prof.name,
                speed_mps=prof.speed_mps,
                congestion_avoidance=prof.congestion_avoidance,
                queue_tolerance_seconds=prof.queue_tolerance_seconds,
                max_visit_seconds=prof.max_visit_seconds,
                distance_weight_factor=prof.distance_weight_factor,
                interests=tuple(float(v) for v in interests),
                table_after_food_propensity=float(rng.random()),
                general_waste_propensity=float(waste_rng.random()),
                food_waste_propensity=float(waste_rng.random()),
                is_passerby=bool(prof.passerby),
            )
        )
    return tuple(records)


def instantiate_visitors(records: tuple) -> list:
    """テーブルからシミュレーション用の Visitor 実体を生成する(毎評価で新規)。"""
    return [
        Visitor(
            visitor_id=r.visitor_id,
            spawn_time_step=r.spawn_time_step,
            position=None,
            speed_mps=r.speed_mps,
            congestion_avoidance=r.congestion_avoidance,
            queue_tolerance_seconds=r.queue_tolerance_seconds,
            max_visit_seconds=r.max_visit_seconds,
            interests=np.array(r.interests, dtype=float),
            distance_weight_factor=r.distance_weight_factor,
            profile_name=r.profile_name,
            table_after_food_propensity=getattr(r, "table_after_food_propensity", 1.0),
            general_waste_propensity=getattr(r, "general_waste_propensity", 1.0),
            food_waste_propensity=getattr(r, "food_waste_propensity", 1.0),
            is_passerby=bool(getattr(r, "is_passerby", False)),
            state=NOT_ARRIVED,
        )
        for r in records
    ]