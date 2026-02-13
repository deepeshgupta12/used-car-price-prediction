from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

Market = Literal["IN", "US", "TEMPLATE"]


@dataclass(frozen=True)
class DatasetPaths:
    cars_csv: Path
    used_cars_csv: Path
    test_final_xlsx: Path


def load_cars_csv(path: Path) -> pd.DataFrame:
    """
    India dataset loader: Cars.csv
    """
    df = pd.read_csv(path)
    return df


def load_used_cars_csv(path: Path) -> pd.DataFrame:
    """
    US dataset loader: used_cars.csv
    """
    df = pd.read_csv(path)
    return df


def load_test_final_xlsx(path: Path) -> pd.DataFrame:
    """
    Template/inference dataset loader: test_final.xlsx
    """
    df = pd.read_excel(path)
    return df
