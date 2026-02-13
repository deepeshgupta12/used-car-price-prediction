from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass
class ModelCache:
    """
    Simple in-process cache for loaded joblib artifacts.
    Keyed by full resolved path to avoid collisions.
    """

    _cache: dict[str, Any]

    def __init__(self) -> None:
        self._cache = {}

    def get(self, path: Path) -> Any:
        key = str(path.resolve())
        if key in self._cache:
            return self._cache[key]
        obj = joblib.load(path)
        self._cache[key] = obj
        return obj
