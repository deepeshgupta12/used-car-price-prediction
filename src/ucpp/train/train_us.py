from __future__ import annotations

import json
from pathlib import Path

import typer

from ucpp.core.constants import RAW_DIR
from ucpp.core.logging import info
from ucpp.features.preprocess import preprocess_us, train_valid_split
from ucpp.ingest.loaders import load_used_cars_csv
from ucpp.models.train_common import save_joblib, train_catboost, train_lightgbm


def _metrics_dict(mae: float, rmse: float, smape: float) -> dict[str, float]:
    return {"mae": float(mae), "rmse": float(rmse), "smape": float(smape)}


def main(
    used_cars_csv: Path = RAW_DIR / "used_cars.csv",
    out_dir: Path = Path("artifacts/v1/us"),
) -> None:
    """
    Train baseline models for US market and write artifacts + metrics.json.
    Run:
      python -m ucpp.train.train_us
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_used_cars_csv(used_cars_csv)
    x, y = preprocess_us(df)
    split = train_valid_split(x, y)

    info("Training US CatBoost (raw target)")
    cb_raw, cb_raw_res = train_catboost(
        split.x_train, split.y_train, split.x_valid, split.y_valid, use_log_target=False
    )
    save_joblib(cb_raw, str(out_dir / "catboost.joblib"))

    info("Training US CatBoost (log target)")
    cb_log, cb_log_res = train_catboost(
        split.x_train, split.y_train, split.x_valid, split.y_valid, use_log_target=True
    )
    save_joblib(cb_log, str(out_dir / "catboost_log.joblib"))

    info("Training US LightGBM (raw target)")
    lgb_raw, lgb_raw_res = train_lightgbm(
        split.x_train, split.y_train, split.x_valid, split.y_valid, use_log_target=False
    )
    save_joblib(lgb_raw, str(out_dir / "lightgbm.joblib"))

    info("Training US LightGBM (log target)")
    lgb_log, lgb_log_res = train_lightgbm(
        split.x_train, split.y_train, split.x_valid, split.y_valid, use_log_target=True
    )
    save_joblib(lgb_log, str(out_dir / "lightgbm_log.joblib"))

    metrics = {
        "catboost": _metrics_dict(cb_raw_res.mae, cb_raw_res.rmse, cb_raw_res.smape),
        "catboost_log": _metrics_dict(cb_log_res.mae, cb_log_res.rmse, cb_log_res.smape),
        "lightgbm": _metrics_dict(lgb_raw_res.mae, lgb_raw_res.rmse, lgb_raw_res.smape),
        "lightgbm_log": _metrics_dict(lgb_log_res.mae, lgb_log_res.rmse, lgb_log_res.smape),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    info(f"Saved US artifacts to: {out_dir}")
    info(f"US metrics: {metrics}")


if __name__ == "__main__":
    typer.run(main)
