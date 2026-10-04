"""固定配置パターン(booths.yaml の fixed_position)の妥当性検査。

実行用コマンド:
python -m pytest real_world_problem/tests/test_fixed_layout.py -q
"""
from __future__ import annotations

from pathlib import Path

from src.constraints import compute_violation
from src.models import load_config
from src.venue import build_base_venue

ROOT = Path(__file__).resolve().parents[1]


def _real_cfg():
    return load_config(ROOT / "config")


def test_inventory():
    """ブース・テーブルの個数と大きさが要求どおり定義されている。"""
    cfg = _real_cfg()
    upper = [s for s in cfg.booths if s.area == "upper_plaza"]
    lower = [s for s in cfg.booths if s.area == "lower_plaza"]
    # 上部広場: ブース 13(3×2 が 12、7×3 が 1)+ テーブル 3(2×2)
    # 幅=長辺(正面/南辺)、高さ=奥行き。正面と長辺を一致させるため (width, height) は
    # 実寸(2×3, 3×7)の縦横を入れ替えて定義している。
    u_booth = [s for s in upper if not s.is_table]
    assert len(u_booth) == 13
    assert sum((s.width, s.height) == (3, 2) for s in u_booth) == 12
    assert sum((s.width, s.height) == (7, 3) for s in u_booth) == 1
    assert sum(s.is_table and (s.width, s.height) == (2, 2) for s in upper) == 3
    # 下部広場: ブース 15(3×2)+ テーブル 15(2×2)
    l_booth = [s for s in lower if not s.is_table]
    assert len(l_booth) == 15
    assert all((s.width, s.height) == (3, 2) for s in l_booth)
    assert sum(s.is_table and (s.width, s.height) == (2, 2) for s in lower) == 15
    # 下部広場ブースの属性内訳
    assert sum(s.kind == "kitchen_car" and s.theme == "normal" for s in l_booth) == 2
    assert sum(s.kind == "kitchen_car" and s.theme == "music" for s in l_booth) == 2
    assert sum(s.kind == "booth" and s.theme == "music" for s in l_booth) == 2
    assert sum(s.kind == "booth" and s.theme == "normal" for s in l_booth) == 9


def test_fixed_layout_is_valid():
    """固定配置パターン((x, y, theta))が全配置制約を満たす(g == 0)。"""
    cfg = _real_cfg()
    venue = build_base_venue(cfg)
    positions = tuple(s.fixed_position for s in cfg.booths)
    assert all(p is not None and len(p) == 3 for p in positions)
    assert compute_violation(cfg, venue, positions) == 0
