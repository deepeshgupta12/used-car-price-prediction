from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _as_numpy(a: Any) -> np.ndarray:
    if isinstance(a, pd.Series | pd.DataFrame):
        return a.to_numpy()
    return np.asarray(a)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_pred - y_true
    return float(math.sqrt(float(np.mean(err**2))))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def compute_regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """
    Standard regression metrics used across training + reporting.
    Returns: {"mae": ..., "rmse": ..., "smape": ...}
    """
    yt = _as_numpy(y_true).astype(float).reshape(-1)
    yp = _as_numpy(y_pred).astype(float).reshape(-1)

    return {
        "mae": _mae(yt, yp),
        "rmse": _rmse(yt, yp),
        "smape": _smape(yt, yp),
    }
