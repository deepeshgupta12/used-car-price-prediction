from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BasicCheckResult:
    name: str
    ok: bool
    details: str


def check_non_empty(df: pd.DataFrame, name: str) -> BasicCheckResult:
    ok = len(df) > 0 and df.shape[1] > 0
    details = f"rows={len(df)}, cols={df.shape[1]}"
    return BasicCheckResult(name=f"{name}:non_empty", ok=ok, details=details)


def check_columns_present(df: pd.DataFrame, name: str, required: list[str]) -> BasicCheckResult:
    missing = [c for c in required if c not in df.columns]
    ok = len(missing) == 0
    details = "missing=" + (",".join(missing) if missing else "none")
    return BasicCheckResult(name=f"{name}:required_cols", ok=ok, details=details)


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "nulls_top10": df.isna().sum().sort_values(ascending=False).head(10).to_dict(),
    }
