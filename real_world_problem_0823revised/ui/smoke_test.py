"""smoke_test.py — UI を開かずに手順 1〜3 をまとめて確認するスクリプト。

    cd real_world_problem
    python -m ui.smoke_test                 # 矩形会場・縮小設定で一通り
    python -m ui.smoke_test --venue real    # 実会場ポリゴンで設定生成だけ確認
    python -m ui.smoke_test --skip-anim     # GIF 生成を飛ばす(速い)

確認する内容:
  1. UI の入力値 → config YAML → load_config() が通るか
  2. 完全ランダム初期集団から直接配置 NSGA-II が回り、スナップショットが出るか
  3. レイアウト図 / 歩行可能通路 / ヒートマップ / アニメーションが生成できるか

pytest には載せていない。実行に十数秒〜数十秒かかるため、
手動の動作確認用として切り離してある。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constraints import design_variable_count  # noqa: E402
from src.models import load_config  # noqa: E402
from ui import render, run_store  # noqa: E402


def step(n: int, title: str) -> None:
    print(f"\n{'=' * 70}\n[手順 {n}] {title}\n{'=' * 70}")


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK  " if ok else "NG  "
    print(f"  {mark}{label}" + (f"  ({detail})" if detail else ""))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--venue", choices=["rect", "real"], default="rect")
    ap.add_argument("--run-id", default="SMOKE")
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--gen", type=int, default=6)
    ap.add_argument("--interval", type=int, default=2)
    ap.add_argument("--visitors", type=int, default=60)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-anim", action="store_true")
    args = ap.parse_args()

    ok = True
    rd = run_store.run_dir(args.run_id)

    # ------------------------------------------------------------ 手順 1
    step(1, "入力値 → config YAML → load_config()")

    preset_main, preset_booths, _ = run_store.load_preset_yaml(args.venue)
    params = run_store.RunParams(
        venue_mode=args.venue,
        grid_width=int(preset_main["grid"]["width"]),
        grid_height=int(preset_main["grid"]["height"]),
        booths=preset_booths,
        n_visitors=args.visitors,
        duration_seconds=args.duration,
        timeout_seconds=60.0,
        population_size=args.pop,
        n_generations=args.gen,
        workers=args.workers,
        snapshot_interval=args.interval,
        # 縮小問題では手法の配線確認を目的とし、整然性制約は緩める。
        max_organization_nonmembership=10.0,
        max_motif_complexity=10.0,
        max_alignment_error=10.0,
        max_gap_variation=10.0,
        max_orientation_error=10.0,
        max_isolation=10.0,
        max_front_clearance_violation=10.0,
        save_artifacts=False,
    )
    params.validate()

    rd.mkdir(parents=True, exist_ok=True)
    run_store._write_json_atomic(rd / "params.json", params.to_dict())
    run_store.build_config_dir(params, rd / "config")

    cfg = load_config(rd / "config")
    print(f"  会場        : {cfg.grid_width} x {cfg.grid_height} セル")
    print(f"  ブース数    : {len(cfg.booths)}")
    print(f"  エージェント: {cfg.n_visitors} 人 / {cfg.duration_seconds:.0f} 秒")
    print(f"  最適化      : 個体数 {cfg.population_size} / 世代数 {cfg.n_generations}")
    print(f"  到着profile : {cfg.arrival_profile}")

    ok &= check(
        "入力値が Config に届いている",
        cfg.n_visitors == args.visitors
        and cfg.population_size == args.pop
        and cfg.n_generations == args.gen,
    )
    ok &= check(
        "到着プロファイルがシミュレート時間に一致",
        abs(max(r[1] for r in cfg.arrival_profile) - args.duration) < 1e-6,
        f"終端 {max(r[1] for r in cfg.arrival_profile):.1f} 秒",
    )
    ok &= check(
        "実際の通行可能幅が4セル以上",
        cfg.min_corridor_width >= 4,
        f"{cfg.min_corridor_width} セル",
    )
    ok &= check(
        "ゴミステーション利用確率が通常時 < 飲食後 < 100%",
        0.0 <= cfg.waste_station_general_probability
        < cfg.waste_station_after_food_probability
        < 1.0,
        f"通常 {cfg.waste_station_general_probability:.0%} / "
        f"飲食後 {cfg.waste_station_after_food_probability:.0%}",
    )
    if args.venue == "rect":
        ok &= check("矩形モードで会場ポリゴンが空", cfg.venue_areas == ())
    else:
        ok &= check(
            "実会場モードで広場が定義されている",
            len(cfg.venue_areas) > 0,
            ", ".join(a[0] for a in cfg.venue_areas),
        )

    # ------------------------------------------------------------ 手順 2
    step(2, "ワーカー単体で直接配置 NSGA-II を回す")

    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "ui.optimize_worker", str(rd)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
    ok &= check("ワーカーが正常終了", proc.returncode == 0, f"{elapsed:.1f} 秒")

    status = run_store.read_status(rd)
    ok &= check("status.json が done", status.get("state") == "done", str(status.get("state")))
    ok &= check(
        "新しい直接配置方式で実行されている",
        str(status.get("method", "")).startswith("single_level_direct_"),
        str(status.get("method", "未記録")),
    )

    gens = run_store.list_snapshot_generations(rd)
    print(f"  スナップショット: {gens}")
    ok &= check("第 1 世代が保存されている", 1 in gens)
    ok &= check("最終世代が保存されている", args.gen in gens)
    ok &= check(
        f"{args.interval} 世代間隔で保存されている",
        all(g % args.interval == 0 for g in gens if g not in (1, args.gen)),
    )

    if not gens:
        print("\nスナップショットが無いため中断します。")
        return 1

    # --- 比較基準(booths.yaml の実会場配置)
    fixed = run_store.read_fixed_layout(rd)
    if args.venue == "real":
        ok &= check(
            "実会場配置の比較基準が評価されている",
            fixed.get("available", False),
            fixed.get("reason", ""),
        )
        if fixed.get("available"):
            ok &= check(
                "実会場配置が配置制約を満たす",
                fixed.get("feasible", False),
                f"g = {fixed.get('violation', '?')}",
            )
            print(
                f"  実会場配置    : f_move {fixed['f_move']:.4f} / "
                f"f_congestion {fixed['f_congestion']:.4g}"
            )
    else:
        ok &= check(
            "矩形モードでは比較基準が無効",
            not fixed.get("available", False),
            "fixed_position は実会場座標のため落としている",
        )

    snap = run_store.read_snapshot(rd, gens[-1])
    sols = snap["solutions"]
    print(f"  最終世代の実行可能解: {len(sols)} 件")
    ok &= check("解に設計ベクトルが入っている", bool(sols) and "x" in sols[0])
    ok &= check(
        "設計変数の次元が現行エンコーディングと一致",
        bool(sols) and len(sols[0]["x"]) == design_variable_count(cfg),
        f"{len(sols[0]['x']) if sols else 0} / "
        f"期待 {design_variable_count(cfg)} 次元",
    )
    ok &= check(
        "非支配解(rank 0)が存在する",
        any(s["rank"] == 0 for s in sols),
    )

    if fixed.get("available") and sols:
        n_dom = sum(
            1
            for s in sols
            if s["f_move"] < fixed["f_move"]
            and s["f_congestion"] < fixed["f_congestion"]
        )
        print(f"  実会場を両目的で上回る解: {n_dom} / {len(sols)}")

    # ------------------------------------------------------------ 手順 3
    step(3, "選択解の可視化")

    x = tuple(int(v) for v in sols[0]["x"])
    out = rd / "ui_cache"
    ctx = render.viz_context(cfg)

    t0 = time.perf_counter()
    layout_png = render.render_layout(cfg, ctx, x, out / "smoke_layout.png")
    ok &= check(
        "レイアウト図", layout_png.exists(), f"{time.perf_counter() - t0:.1f} 秒"
    )

    t0 = time.perf_counter()
    walkable_png = render.render_walkable_corridors(
        cfg, ctx, x, out / "smoke_walkable.png"
    )
    ok &= check(
        "歩行可能通路図",
        walkable_png.exists(),
        f"{time.perf_counter() - t0:.1f} 秒",
    )

    t0 = time.perf_counter()
    heat_png, kpis = render.render_heatmap(cfg, ctx, x, out / "smoke_heat.png")
    ok &= check("ヒートマップ", heat_png.exists(), f"{time.perf_counter() - t0:.1f} 秒")
    ok &= check("可視化がタイムアウトしていない", not kpis["timed_out"])
    ok &= check(
        "再評価の目的関数値がスナップショットと一致",
        abs(kpis["f_move"] - sols[0]["f_move"]) < 1e-6
        and abs(kpis["f_congestion"] - sols[0]["f_congestion"]) < 1e-6,
        f"f_move {kpis['f_move']:.4f} vs {sols[0]['f_move']:.4f}",
    )

    if args.skip_anim:
        print("  --  アニメーションはスキップ")
    else:
        t0 = time.perf_counter()
        gif = render.render_agent_animation(
            cfg, ctx, x, out / "smoke_anim.gif", frame_interval=5
        )
        ok &= check(
            "アニメーション", gif.exists(), f"{time.perf_counter() - t0:.1f} 秒"
        )

    # ------------------------------------------------------------ まとめ
    print(f"\n{'=' * 70}")
    print("結果:", "すべて OK" if ok else "NG あり(上を確認)")
    print(f"生成物: {out}")
    print(
        "  smoke_layout.png / smoke_walkable.png / smoke_heat.png"
        + ("" if args.skip_anim else " / smoke_anim.gif")
    )
    print("これらを目視で確認してください(README 手順 3 の観点)。")
    print(f"後片付け: {rd} を削除")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":  # §16.3 Windows 対応ガード
    raise SystemExit(main())
