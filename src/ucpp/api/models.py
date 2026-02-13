from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Market = Literal["IN", "US"]


class PredictRequest(BaseModel):
    """
    Single-row inference request.
    payload is the raw row dict; we validate it using ucpp.predict.schema.validate_payload.
    """

    model_config = ConfigDict(extra="forbid")

    market: Market = Field(..., description="IN or US")
    model: str | None = Field(
        default=None, description="Optional model override (defaults to best-per-market)"
    )
    payload: dict[str, Any] = Field(..., description="Raw input fields for the market")


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: Market
    model: str
    prediction: float


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: Market = Field(..., description="IN or US")
    model: str | None = Field(
        default=None, description="Optional model override (defaults to best-per-market)"
    )
    rows: list[dict[str, Any]] = Field(..., description="List of raw payload rows")
    strict: bool = Field(
        default=True,
        description="If true: fail on first invalid row. If false: return errors[] for bad rows.",
    )


class BatchPredItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_index: int
    market: Market
    model: str
    prediction: float


class BatchErrorItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_index: int
    type: Literal["validation_error", "inference_error"]
    detail: Any


class BatchPredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: Market
    model: str
    preds: list[BatchPredItem]
    errors: list[BatchErrorItem] = Field(default_factory=list)
