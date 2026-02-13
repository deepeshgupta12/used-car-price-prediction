from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from pandas.api.types import is_categorical_dtype, is_object_dtype, is_string_dtype

from ucpp.models.metrics import mae, rmse, smape


@dataclass(frozen=True)
class TrainResult:
    model_name: str
    metrics: dict[str, float]


_CAT_MISSING = "__MISSING__"


def _cat_features_indices(df: pd.DataFrame) -> list[int]:
    """
    CatBoost categorical/text columns can be:
    - object
    - pandas string dtype
    - category
    """
    idx: list[int] = []
    for i, col in enumerate(df.columns):
        s = df[col]
        if is_object_dtype(s) or is_string_dtype(s) or is_categorical_dtype(s):
            idx.append(i)
    return idx


def _sanitize_for_catboost(df: pd.DataFrame, cat_idx: list[int]) -> pd.DataFrame:
    """
    CatBoost does NOT allow NaN in categorical features.
    Convert categorical columns to string and fill NaN with sentinel.
    Leave numeric columns untouched.
    """
    out = df.copy()
    for i in cat_idx:
        col = out.columns[i]
        # Ensure missing are filled, then cast to string
        out[col] = out[col].fillna(_CAT_MISSING).astype(str)
    return out


def train_catboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    seed: int = 42,
    use_log_target: bool = False,
) -> tuple[CatBoostRegressor, TrainResult]:
    cat_idx = _cat_features_indices(x_train)

    xtr = _sanitize_for_catboost(x_train, cat_idx)
    xva = _sanitize_for_catboost(x_valid, cat_idx)

    y_tr = y_train.to_numpy()
    y_va = y_valid.to_numpy()

    if use_log_target:
        y_tr = np.log1p(y_tr)
        y_va = np.log1p(y_va)

    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        random_seed=seed,
        eval_metric="RMSE",
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(
        xtr,
        y_tr,
        cat_features=cat_idx,
        eval_set=(xva, y_va),
        use_best_model=True,
    )

    pred = model.predict(xva)

    if use_log_target:
        pred = np.expm1(pred)
        y_eval = y_valid.to_numpy()
    else:
        y_eval = y_valid.to_numpy()

    res = TrainResult(
        model_name="catboost_log" if use_log_target else "catboost",
        metrics={
            "mae": mae(y_eval, pred),
            "rmse": rmse(y_eval, pred),
            "smape": smape(y_eval, pred),
        },
    )
    return model, res


def train_lightgbm(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    seed: int = 42,
    use_log_target: bool = False,
) -> tuple[LGBMRegressor, TrainResult]:
    def _encode_object_cols(
        train_df: pd.DataFrame, valid_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        tr = train_df.copy()
        va = valid_df.copy()
        for col in tr.columns:
            if (
                is_object_dtype(tr[col])
                or is_string_dtype(tr[col])
                or is_categorical_dtype(tr[col])
            ):
                combined = pd.concat([tr[col], va[col]], axis=0).astype("category")
                tr[col] = combined.iloc[: len(tr)].cat.codes
                va[col] = combined.iloc[len(tr) :].cat.codes
        return tr, va

    xtr, xva = _encode_object_cols(x_train, x_valid)

    y_tr = y_train.to_numpy()
    y_va = y_valid.to_numpy()

    if use_log_target:
        y_tr = np.log1p(y_tr)
        y_va = np.log1p(y_va)

    model = LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(
        xtr,
        y_tr,
        eval_set=[(xva, y_va)],
        eval_metric="rmse",
        callbacks=[],
    )

    pred = model.predict(xva)

    if use_log_target:
        pred = np.expm1(pred)
        y_eval = y_valid.to_numpy()
    else:
        y_eval = y_valid.to_numpy()

    res = TrainResult(
        model_name="lightgbm_log" if use_log_target else "lightgbm",
        metrics={
            "mae": mae(y_eval, pred),
            "rmse": rmse(y_eval, pred),
            "smape": smape(y_eval, pred),
        },
    )
    return model, res


def save_joblib(obj: Any, path: str) -> None:
    joblib.dump(obj, path)
