"""metrics.py — 目的関数(§12)と補助 KPI(§12.3)・幸福度(§13)。

f_move は自由歩行換算時間(§12.1【v2】)。ブロック待機時間・待ち行列時間・
入場遅延時間は f_move に含めない(§25)。
"""
from __future__ import annotations

import numpy as np

from .models import FINISHED, NOT_ARRIVED, Config


def free_walk_seconds(v, cfg: Config) -> float:
    """来場者 1 人の自由歩行換算時間(秒)。"""
    return v.moved_cells * cfg.cell_size_meters / v.speed_mps


def happiness(v, cfg: Config) -> float:
    """§13 の幸福度(補助 KPI)。"""
    u_sum = 0.0
    for b in v.visited_booths:
        spec = cfg.booths[b]
        interest = float(v.interests[cfg.category_index(spec.category)])
        u_sum += cfg.w_interest * interest + cfg.w_attract * spec.attractiveness
    walk_min = free_walk_seconds(v, cfg) / 60.0
    queue_min = v.queue_seconds / 60.0
    return (
        u_sum
        - cfg.hap_alpha * walk_min
        - cfg.hap_beta * queue_min
        - cfg.hap_gamma * v.give_up_count
        + cfg.hap_rest_bonus * v.table_visit_count
        - cfg.hap_fatigue_penalty * v.fatigue
    )


def compute_metrics(sim) -> tuple[float, float, dict]:
    """シミュレーション終了時に目的関数 2 つと補助 KPI 群を計算する。"""
    cfg: Config = sim.cfg
    spawned = [v for v in sim.visitors if v.state != NOT_ARRIVED]
    n_spawned = len(spawned)
    # 素通り客(ブースへ行かない通行人)。混雑・移動負担には通常どおり
    # 含めるが、体験系 KPI は「ブース目当てで来た人」だけの値も併記する。
    passersby = [v for v in spawned if getattr(v, "is_passerby", False)]
    booth_goers = [v for v in spawned if not getattr(v, "is_passerby", False)]

    # --- 目的関数 1: 移動負担(自由歩行換算時間, §12.1)
    if n_spawned:
        walk_secs = np.array(
           [free_walk_seconds(v, cfg) for v in spawned],
            dtype=float,
        )
        n_exp_for_move = np.array(
            [len(v.visited_booths) for v in spawned],
            dtype=float,
        )

    # 0除算防止。体験0人は denominator=1 として扱う
        denom_exp = np.maximum(1.0, n_exp_for_move)

        walk_per_experience = walk_secs / denom_exp

    # 目的関数
        f_move = float(walk_per_experience.mean())
    else:
       walk_secs = np.array([])
       f_move = 0.0
# --- 目的関数 1: 1体験あたり移動負担 + 少量の総歩行時間
    if n_spawned:
       walk_secs = np.array(
           [free_walk_seconds(v, cfg) for v in spawned],
           dtype=float,
       )

       n_exp_for_move = np.array(
           [len(v.visited_booths) for v in spawned],
           dtype=float,
        )

    # 0除算防止。体験0人は denominator=1 として扱う
       denom_exp = np.maximum(1.0, n_exp_for_move)

    # 1体験あたりの移動負担
       walk_per_experience = walk_secs / denom_exp

       mean_free_walk_seconds = float(walk_secs.mean())
       mean_move_seconds_per_experience = float(walk_per_experience.mean())
       p90_move_seconds_per_experience = float(
        np.percentile(walk_per_experience, 90)
    )
    # 総歩行時間を少しだけ反映する重み
       total_walk_weight = 0.15
       tail_weight = 0.20
    # 目的関数
       f_move_core = float(
           (1.0- tail_weight)*mean_move_seconds_per_experience
          + tail_weight * p90_move_seconds_per_experience
       )
       f_move = float(
        f_move_core
        + total_walk_weight * mean_free_walk_seconds
    )
    else:
       walk_secs = np.array([])
       mean_free_walk_seconds = 0.0
       mean_move_seconds_per_experience = 0.0
       p90_move_seconds_per_experience = 0.0
       f_move = 0.0

    # --- 補助 KPI 用の配列
    q = np.array([v.queue_seconds for v in spawned]) if n_spawned else np.array([0.0])
    blocked = np.array([v.blocked_seconds for v in spawned]) if n_spawned else np.array([0.0])
    delays = np.array([v.entry_delay_seconds for v in spawned]) if n_spawned else np.array([0.0])
    n_exp = np.array([len(v.visited_booths) for v in spawned]) if n_spawned else np.array([0])
    giveups = int(sum(v.give_up_count for v in spawned))
    total_exp = int(n_exp.sum())
    hap = np.array([happiness(v, cfg) for v in spawned]) if n_spawned else np.array([0.0])
    finished = sum(1 for v in spawned if v.state == FINISHED)
    fatigue = np.array([v.fatigue for v in spawned]) if n_spawned else np.array([0.0])
    peak_fatigue = (
        np.array([v.peak_fatigue for v in spawned]) if n_spawned else np.array([0.0])
    )
    table_seconds = (
        np.array([v.table_seconds for v in spawned]) if n_spawned else np.array([0.0])
    )
    table_visits = int(sum(v.table_visit_count for v in spawned))
    food_table_requests = int(sum(v.table_requests_after_food for v in spawned))
    food_table_visits = int(sum(v.food_table_visit_count for v in spawned))
    waste_station_requests = int(
        sum(v.waste_station_request_count for v in spawned)
    )
    waste_station_visits = int(
        sum(v.waste_station_visit_count for v in spawned)
    )
    waste_station_giveups = int(
        sum(v.waste_station_give_up_count for v in spawned)
    )
    food_waste_opportunities = int(
        sum(v.food_waste_opportunity_count for v in spawned)
    )
    food_waste_requests = int(
        sum(v.food_waste_request_count for v in spawned)
    )
    food_waste_visits = int(
        sum(v.food_waste_visit_count for v in spawned)
    )
    table_capacity = int(sum(b.capacity for b in cfg.booths if b.is_table))
    table_capacity_seconds = max(
        1.0, table_capacity * max(1, sim.executed_steps) * cfg.dt_seconds
    )

    # --- 目的関数 2: ブース周辺セルのみを対象にした混雑度
    denom_raw = max(1.0, float(cfg.n_steps * sim.n_walkable))
    f_congestion_raw = sim.congestion_sum / denom_raw

    weight_sum = float(getattr(sim, "congestion_weight_sum", sim.n_walkable))
    denom_weighted = max(1.0, float(cfg.n_steps)) #* weight_sum
    f_congestion_booth_near = getattr(sim, "weighted_congestion_sum", sim.congestion_sum) / denom_weighted

    mean_blocked_seconds = float(blocked.mean())
    blocked_weight = 0.00005
    f_congestion = f_congestion_booth_near#*1000

    kpis = {
        "n_spawned": n_spawned,
        "mean_queue_seconds": float(q.mean()),
        "max_queue_seconds": float(q.max()),
        "mean_free_walk_seconds": f_move,
        "total_moved_cells": int(sum(v.moved_cells for v in spawned)),
        "mean_blocked_seconds": mean_blocked_seconds,
        "mean_entry_delay_seconds": float(delays.mean()),
        "max_cell_occupancy_ratio": float(sim.max_occ_ratio),
        "congestion_sum": float(sim.congestion_sum),
        "weighted_congestion_sum": float(getattr(sim, "weighted_congestion_sum", sim.congestion_sum)),
        "congestion_weight_sum": float(getattr(sim, "congestion_weight_sum", sim.n_walkable)),

        # 追加しておくと比較しやすい
        "f_congestion_raw": float(f_congestion_raw),
        "f_congestion_booth_near": float(f_congestion_booth_near),
        "f_congestion_weighted": float(f_congestion_booth_near),  # 互換用: 実体はブース周辺混雑度
        "blocked_congestion_weight": float(blocked_weight),
        "f_congestion_with_blocked": float(f_congestion),  # blocked項は足していない

        "mean_booths_experienced": float(n_exp.mean()),
        "experience_success_rate": (
            total_exp / (total_exp + giveups) if (total_exp + giveups) else 1.0
        ),
        "give_up_total": giveups,
        "exit_completion_rate": finished / n_spawned if n_spawned else 0.0,
        "mean_happiness": float(hap.mean()),
        "happiness_p10": float(np.percentile(hap, 10)) if n_spawned else 0.0,
        "mean_final_fatigue": float(fatigue.mean()),
        "mean_peak_fatigue": float(peak_fatigue.mean()),
        "table_visit_total": table_visits,
        "table_user_rate": (
            sum(v.table_visit_count > 0 for v in spawned) / n_spawned
            if n_spawned else 0.0
        ),
        "mean_table_seconds": float(table_seconds.mean()),
        "table_seat_utilization": float(table_seconds.sum() / table_capacity_seconds),
        "food_table_request_total": food_table_requests,
        "food_table_visit_total": food_table_visits,
        "food_table_followthrough_rate": (
            food_table_visits / food_table_requests if food_table_requests else 1.0
        ),
        # --- ゴミステーション利用
        "waste_station_request_total": waste_station_requests,
        "waste_station_visit_total": waste_station_visits,
        "waste_station_give_up_total": waste_station_giveups,
        "waste_station_user_rate": (
            sum(v.waste_station_visit_count > 0 for v in spawned) / n_spawned
            if n_spawned else 0.0
        ),
        "waste_station_followthrough_rate": (
            waste_station_visits / waste_station_requests
            if waste_station_requests else 1.0
        ),
        "food_waste_opportunity_total": food_waste_opportunities,
        "food_waste_request_total": food_waste_requests,
        "food_waste_visit_total": food_waste_visits,
        "food_waste_request_rate": (
            food_waste_requests / food_waste_opportunities
            if food_waste_opportunities else 0.0
        ),
        "food_waste_followthrough_rate": (
            food_waste_visits / food_waste_requests
            if food_waste_requests else 1.0
        ),

        # --- 素通り客(広場を通り抜けるだけの通行人)
        "passerby_total": len(passersby),
        "passerby_rate": len(passersby) / n_spawned if n_spawned else 0.0,
        "mean_passerby_walk_seconds": (
            float(np.mean([free_walk_seconds(v, cfg) for v in passersby]))
            if passersby else 0.0
        ),
        # 上の体験系 KPI は素通り客も母数に含むため、ブース目当ての来場者だけの
        # 値も併記する(素通り客がいなければ同じ値になる)。
        "mean_booths_experienced_booth_goers": (
            float(np.mean([len(v.visited_booths) for v in booth_goers]))
            if booth_goers else 0.0
        ),
        "mean_happiness_booth_goers": (
            float(np.mean([happiness(v, cfg) for v in booth_goers]))
            if booth_goers else 0.0
        ),
    }
    return f_move, f_congestion, kpis