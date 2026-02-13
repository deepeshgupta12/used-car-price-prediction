from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split

from ucpp.features.parsers import (
    parse_in_engine_cc,
    parse_in_mileage_kmpl,
    parse_in_new_price_lakh,
    parse_in_power_bhp,
    parse_us_milage_to_int,
    parse_us_price_to_float,
)

Market = Literal["IN", "US"]


@dataclass(frozen=True)
class DatasetSplit:
    x_train: pd.DataFrame
    x_valid: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series


def _clean_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure object columns are consistently string-like with missing values as None.
    This matches the behavior used during training.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str).replace({"nan": None, "None": None, "": None})
    return out


def transform_in(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build IN feature matrix X from raw input rows.
    Does NOT require 'Price' column.
    """
    work = df.copy()

    if "Price" in work.columns:
        work = work.drop(columns=["Price"])

    work["Mileage_num"] = work["Mileage"].apply(parse_in_mileage_kmpl)
    work["Engine_cc"] = work["Engine"].apply(parse_in_engine_cc)
    work["Power_bhp"] = work["Power"].apply(parse_in_power_bhp)
    work["NewPrice_lakh"] = work["New_Price"].apply(parse_in_new_price_lakh)

    work["Year"] = pd.to_numeric(work["Year"], errors="coerce")
    work["Kilometers_Driven"] = pd.to_numeric(work["Kilometers_Driven"], errors="coerce")
    work["Seats"] = pd.to_numeric(work["Seats"], errors="coerce")
    work["No. of Doors"] = pd.to_numeric(work["No. of Doors"], errors="coerce")

    work = work.drop(columns=["Mileage", "Engine", "Power", "New_Price"])

    work = _clean_object_cols(work)
    return work


def preprocess_in(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    India: target = Price (float)
    Baseline numeric + categorical features.
    """
    work = df.copy()
    y = work["Price"].astype(float)
    x = transform_in(work)
    return x, y


def transform_us(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build US feature matrix X from raw input rows.
    Does NOT require 'price' column.
    """
    work = df.copy()

    if "price" in work.columns:
        work = work.drop(columns=["price"])

    work["milage_miles"] = work["milage"].apply(parse_us_milage_to_int)
    work = work.drop(columns=["milage"])

    work["model_year"] = pd.to_numeric(work["model_year"], errors="coerce")

    work = _clean_object_cols(work)
    return work


def preprocess_us(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    US: target = price (string -> float USD)
    """
    work = df.copy()
    y = work["price"].apply(parse_us_price_to_float).astype(float)
    x = transform_us(work)
    return x, y


def train_valid_split(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """
    Basic split. Drops rows where y is missing.
    """
    mask = y.notna()
    x2 = x.loc[mask].reset_index(drop=True)
    y2 = y.loc[mask].reset_index(drop=True)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x2, y2, test_size=test_size, random_state=random_state
    )
    return DatasetSplit(x_train=x_train, x_valid=x_valid, y_train=y_train, y_valid=y_valid)
