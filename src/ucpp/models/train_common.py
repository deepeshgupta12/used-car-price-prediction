from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from ucpp.models.metrics import compute_regression_metrics


@dataclass(frozen=True)
class TrainResult:
    mae: float
    rmse: float
    smape: float


def save_joblib(obj: Any, path: str) -> None:
    joblib.dump(obj, path)


def _as_numpy(a: Any) -> np.ndarray:
    if isinstance(a, pd.Series | pd.DataFrame):
        return a.to_numpy()
    return np.asarray(a)


def _is_categorical_series(s: pd.Series) -> bool:
    # Handles pandas 'object' and pandas 'string/str' dtype
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def _sanitize_catboost_categoricals(
    df: pd.DataFrame, *, cat_cols: list[str] | None = None
) -> tuple[pd.DataFrame, list[int]]:
    """
    CatBoost requires categorical values to be string or int.
    Missing values in categorical columns must be converted to a placeholder string.

    We treat both:
      - object dtype
      - pandas string dtype (printed as 'str' / 'string[python]')
    """
    x = df.copy()

    if cat_cols is None:
        cat_cols = [c for c in x.columns if _is_categorical_series(x[c])]

    cat_idx: list[int] = [x.columns.get_loc(c) for c in cat_cols]

    for c in cat_cols:
        s = x[c]
        s2 = s.astype(object).where(s.notna(), other="__MISSING__").astype(str)
        x[c] = s2

    return x, cat_idx


@dataclass(frozen=True)
class EncodedFrame:
    x_num: pd.DataFrame
    cat_cols: list[str]


def encode_for_lgbm(x: pd.DataFrame) -> EncodedFrame:
    """
    LightGBM sklearn wrapper can't handle raw string columns.
    We encode categoricals into integer codes (stable within a run).

    - Identify categorical columns (object or string dtype)
    - Fill missing with '__MISSING__'
    - Convert to pandas Categorical
    - Use .cat.codes
    """
    out = x.copy()
    cat_cols = [c for c in out.columns if _is_categorical_series(out[c])]

    for c in cat_cols:
        s = out[c].astype(object).where(out[c].notna(), other="__MISSING__").astype(str)
        out[c] = pd.Categorical(s).codes.astype("int32")

    return EncodedFrame(x_num=out, cat_cols=cat_cols)


def encode_for_lgbm_with_cols(x: pd.DataFrame, cat_cols: list[str]) -> EncodedFrame:
    """
    Encode only the provided categorical columns list.
    This ensures train/valid/inference encode the same set of columns.
    """
    out = x.copy()
    for c in cat_cols:
        if c not in out.columns:
            # If a column is missing entirely, create it as missing placeholder
            out[c] = "__MISSING__"
        s = out[c].astype(object).where(out[c].notna(), other="__MISSING__").astype(str)
        out[c] = pd.Categorical(s).codes.astype("int32")
    return EncodedFrame(x_num=out, cat_cols=cat_cols)


def _maybe_log1p(y: np.ndarray, use_log_target: bool) -> np.ndarray:
    if not use_log_target:
        return y
    return np.log1p(y)


def _maybe_expm1(y: np.ndarray, use_log_target: bool) -> np.ndarray:
    if not use_log_target:
        return y
    return np.expm1(y)


def train_catboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    use_log_target: bool,
) -> tuple[CatBoostRegressor, TrainResult]:
    # Determine categorical columns from train, reuse for valid
    train_cat_cols = [c for c in x_train.columns if _is_categorical_series(x_train[c])]
    x_tr, cat_idx = _sanitize_catboost_categoricals(x_train, cat_cols=train_cat_cols)
    x_va, _ = _sanitize_catboost_categoricals(x_valid, cat_cols=train_cat_cols)

    y_tr = _as_numpy(y_train).astype(float)
    y_va = _as_numpy(y_valid).astype(float)

    y_tr_t = _maybe_log1p(y_tr, use_log_target)
    y_va_t = _maybe_log1p(y_va, use_log_target)

    model = CatBoostRegressor(
        iterations=2000,
        depth=8,
        learning_rate=0.05,
        loss_function="MAE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(
        x_tr,
        y_tr_t,
        eval_set=(x_va, y_va_t),
        cat_features=cat_idx,
        use_best_model=True,
    )

    pred_t = _as_numpy(model.predict(x_va)).astype(float)
    pred = _maybe_expm1(pred_t, use_log_target)

    res = compute_regression_metrics(y_va, pred)
    return model, TrainResult(mae=res["mae"], rmse=res["rmse"], smape=res["smape"])


@dataclass(frozen=True)
class LgbmBundle:
    model: LGBMRegressor
    cat_cols: list[str]


def train_lightgbm(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    use_log_target: bool,
) -> tuple[LgbmBundle, TrainResult]:
    enc_tr = encode_for_lgbm(x_train)
    enc_va = encode_for_lgbm_with_cols(x_valid, enc_tr.cat_cols)

    y_tr = _as_numpy(y_train).astype(float)
    y_va = _as_numpy(y_valid).astype(float)

    y_tr_t = _maybe_log1p(y_tr, use_log_target)
    y_va_t = _maybe_log1p(y_va, use_log_target)

    model = LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        enc_tr.x_num,
        y_tr_t,
        eval_set=[(enc_va.x_num, y_va_t)],
        eval_metric="l1",
    )

    pred_t = _as_numpy(model.predict(enc_va.x_num)).astype(float)
    pred = _maybe_expm1(pred_t, use_log_target)

    res = compute_regression_metrics(y_va, pred)
    return LgbmBundle(model=model, cat_cols=enc_tr.cat_cols), TrainResult(
        mae=res["mae"], rmse=res["rmse"], smape=res["smape"]
    )
