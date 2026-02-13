from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pandera.errors import SchemaErrors

from ucpp.core.logging import info, warn
from ucpp.validate.schemas import schema_in, schema_template, schema_us

Market = Literal["IN", "US", "TEMPLATE"]


@dataclass(frozen=True)
class ValidationOutcome:
    market: Market
    ok: bool
    errors: str
    rows_in: int
    rows_out: int


def validate_df(df: pd.DataFrame, market: Market) -> tuple[pd.DataFrame, ValidationOutcome]:
    if market == "IN":
        schema = schema_in()
    elif market == "US":
        schema = schema_us()
    elif market == "TEMPLATE":
        schema = schema_template()
    else:
        raise ValueError(f"Unknown market={market}")

    rows_in = len(df)

    try:
        validated = schema.validate(df, lazy=True)
        outcome = ValidationOutcome(
            market=market,
            ok=True,
            errors="",
            rows_in=rows_in,
            rows_out=len(validated),
        )
        info(f"{market} schema validation OK (rows={rows_in})")
        return validated, outcome
    except SchemaErrors as e:
        failure_cases = e.failure_cases
        msg = failure_cases.head(50).to_string(index=False)
        outcome = ValidationOutcome(
            market=market,
            ok=False,
            errors=msg,
            rows_in=rows_in,
            rows_out=rows_in,
        )
        warn(f"{market} schema validation FAILED. Showing first 50 failure cases.")
        warn(msg)
        return df, outcome
