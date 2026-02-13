from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from ucpp.api.cache import ModelCache
from ucpp.api.models import (
    BatchErrorItem,
    BatchPredictRequest,
    BatchPredictResponse,
    BatchPredItem,
    PredictRequest,
    PredictResponse,
)
from ucpp.features.preprocess import transform_in, transform_us
from ucpp.predict.predict import (
    DEFAULT_MODEL_BY_MARKET,
    _invert_target,
    _model_path,
    _predict_estimator,
)
from ucpp.predict.schema import validate_payload

app = FastAPI(title="Used Car Price Prediction API", version="v3")

_model_cache = ModelCache()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _choose_model(market_u: str, model: str | None) -> str:
    chosen = model or DEFAULT_MODEL_BY_MARKET.get(market_u)
    if chosen is None:
        raise HTTPException(status_code=400, detail="market must be IN or US")
    return chosen


def _feature_matrix(market_u: str, clean_payload: dict[str, Any]) -> Any:
    df = pd.DataFrame([clean_payload])
    if market_u == "IN":
        return transform_in(df)
    if market_u == "US":
        return transform_us(df)
    raise HTTPException(status_code=400, detail="market must be IN or US")


def _predict(
    *,
    market_u: str,
    model: str,
    payload: dict[str, Any],
    artifacts_dir: Path,
) -> float:
    model_file = _model_path(artifacts_dir, market_u, model)
    if not model_file.exists():
        raise HTTPException(status_code=400, detail=f"model not found: {model_file}")

    # schema validation / cleaning
    try:
        clean = validate_payload(market_u, payload)
    except ValidationError as ve:
        # FastAPI will usually convert ValidationError to 422 if raised inside request model parsing,
        # but here validation happens post-parse, so we translate it.
        raise HTTPException(status_code=422, detail=ve.errors()) from ve

    x = _feature_matrix(market_u, clean)
    estimator = _model_cache.get(model_file)
    yhat = _predict_estimator(estimator, x)
    return _invert_target(model, yhat)


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    market_u = req.market.upper()
    chosen = _choose_model(market_u, req.model)

    pred = _predict(
        market_u=market_u,
        model=chosen,
        payload=req.payload,
        artifacts_dir=Path("artifacts"),
    )

    return PredictResponse(market=req.market, model=chosen, prediction=pred)


@app.post("/v1/batch", response_model=BatchPredictResponse)
def batch_predict(req: BatchPredictRequest) -> BatchPredictResponse:
    market_u = req.market.upper()
    chosen = _choose_model(market_u, req.model)

    preds: list[BatchPredItem] = []
    errors: list[BatchErrorItem] = []

    for idx, raw in enumerate(req.rows):
        try:
            pred = _predict(
                market_u=market_u,
                model=chosen,
                payload=raw,
                artifacts_dir=Path("artifacts"),
            )
            preds.append(
                BatchPredItem(row_index=idx, market=req.market, model=chosen, prediction=pred)
            )
        except HTTPException as he:
            # translate back into batch error items
            # 422 => validation_error, others => inference_error
            err_type = "validation_error" if he.status_code == 422 else "inference_error"
            err = BatchErrorItem(row_index=idx, type=err_type, detail=he.detail)
            if req.strict:
                raise
            errors.append(err)
        except Exception as e:
            err = BatchErrorItem(row_index=idx, type="inference_error", detail=str(e))
            if req.strict:
                raise HTTPException(status_code=500, detail=str(e)) from e
            errors.append(err)

    return BatchPredictResponse(market=req.market, model=chosen, preds=preds, errors=errors)


if __name__ == "__main__":
    # Optional: allow running as `python -m ucpp.api.app`
    import uvicorn

    uvicorn.run("ucpp.api.app:app", host="127.0.0.1", port=8000, reload=True)
