"""Dependency factories for shared, infrastructure-level services."""

from __future__ import annotations

from functools import lru_cache

from configs.base import get_settings
from core.nec.engine import NecEngine


@lru_cache
def get_engine() -> NecEngine:
    return NecEngine(get_settings())
