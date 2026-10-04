"""テスト共通フィクスチャ。中央密集・周辺分散の手作り配置(§23.6 用)。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import load_config  # noqa: E402
from src.parallel import build_context  # noqa: E402

#: 20×16 会場・L字待機列で制約を満たす固定配置
CENTRAL_LAYOUT = ((5, 2), (3, 6), (8, 10), (8, 5), (14, 8), (2, 11))
#: 再現性・比較用の別の有効配置
PERIMETER_LAYOUT = ((4, 2), (12, 11), (7, 3), (16, 6), (3, 8), (10, 6))


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "config")


@pytest.fixture(scope="session")
def ctx(cfg):
    return build_context(cfg)
