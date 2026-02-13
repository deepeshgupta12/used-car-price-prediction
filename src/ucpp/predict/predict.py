from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

import joblib
import typer

from ucpp.features.preprocess import transform_in, transform_us
from ucpp.models.train_common import LgbmBundle, encode_for_lgbm_with_cols

# Default model choice (based on your v1 metrics summary)
DEFAULT_MODEL_BY_MARKET: dict[str, str] = {
    "IN": "lightgbm_log",
    "US": "lightgbm_log",
}


@dataclass(frozen=True)
class PredictResult:
    market: str
    model: str
    prediction: float


def _read_input_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise typer.BadParameter("input JSON must be an object")
    return obj


def _model_path(artifacts_dir: Path, market: str, model: str) -> Path:
    # artifacts/v1/in/lightgbm_log.joblib
    return artifacts_dir / "v1" / market.lower() / f"{model}.joblib"


def _is_log_model(model: str) -> bool:
    return model.endswith("_log")


def _invert_target(model: str, yhat: float) -> float:
    # log models were trained on log1p(target)
    if _is_log_model(model):
        return float(math.expm1(yhat))
    return float(yhat)


def _predict_estimator(estimator: Any, x: Any) -> float:
    """
    Supports:
      - CatBoostRegressor (has .predict)
      - LGBMRegressor (has .predict)
      - LgbmBundle(model=..., cat_cols=[...]) persisted via joblib
    """
    if isinstance(estimator, LgbmBundle):
        enc = encode_for_lgbm_with_cols(x, estimator.cat_cols)
        return float(estimator.model.predict(enc.x_num)[0])

    if not hasattr(estimator, "predict"):
        raise typer.BadParameter("Loaded artifact does not support predict()")

    return float(estimator.predict(x)[0])


def predict_one(
    *,
    market: str,
    model: str,
    payload: dict[str, Any],
    artifacts_dir: Path,
) -> PredictResult:
    market_u = market.upper()
    if market_u not in {"IN", "US"}:
        raise typer.BadParameter("market must be IN or US")

    model_file = _model_path(artifacts_dir, market_u, model)
    if not model_file.exists():
        raise typer.BadParameter(f"model not found: {model_file}")

    import pandas as pd  # local import to keep top clean

    df = pd.DataFrame([payload])

    if market_u == "IN":
        x = transform_in(df)
    else:
        x = transform_us(df)

    estimator = joblib.load(model_file)
    yhat = _predict_estimator(estimator, x)
    pred = _invert_target(model, yhat)

    return PredictResult(market=market_u, model=model, prediction=pred)


def main(
    market: Annotated[str, typer.Option(help="IN or US")],
    input_json: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Path to JSON with raw fields")
    ],
    model: Annotated[
        str | None, typer.Option(help="Override model name (default: best per market)")
    ] = None,
    artifacts_dir: Annotated[
        Path, typer.Option(help="Artifacts root (default: artifacts/)")
    ] = Path("artifacts"),
) -> None:
    """
    Predict price for a single input row.

    Usage:
      python -m ucpp.predict.predict --market IN --input-json data/samples/in_one.json
    """
    market_u = market.upper()
    chosen = model or DEFAULT_MODEL_BY_MARKET.get(market_u)
    if chosen is None:
        raise typer.BadParameter("market must be IN or US")

    payload = _read_input_json(input_json)
    res = predict_one(market=market_u, model=chosen, payload=payload, artifacts_dir=artifacts_dir)
    typer.echo(json.dumps(asdict(res), indent=2))


if __name__ == "__main__":
    typer.run(main)
