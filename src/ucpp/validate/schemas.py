from __future__ import annotations

import pandera.pandas as pa
from pandera import Check


def schema_in() -> pa.DataFrameSchema:
    """
    India dataset (Cars.csv) schema contract.
    """
    return pa.DataFrameSchema(
        {
            "Name": pa.Column(pa.String, nullable=True),
            "Location": pa.Column(pa.String, nullable=True),
            "Year": pa.Column("Int64", nullable=True, checks=Check.between(1980, 2035)),
            "Kilometers_Driven": pa.Column("Int64", nullable=True, checks=Check.ge(0)),
            "Fuel_Type": pa.Column(pa.String, nullable=True),
            "Transmission": pa.Column(pa.String, nullable=True),
            "Owner_Type": pa.Column(pa.String, nullable=True),
            "Mileage": pa.Column(pa.String, nullable=True),
            "Engine": pa.Column(pa.String, nullable=True),
            "Power": pa.Column(pa.String, nullable=True),
            "Colour": pa.Column(pa.String, nullable=True),
            "Seats": pa.Column(pa.Float, nullable=True, checks=Check.between(1, 12)),
            "No. of Doors": pa.Column(pa.Float, nullable=True, checks=Check.between(2, 10)),
            "New_Price": pa.Column(pa.String, nullable=True),
            "Price": pa.Column(pa.Float, nullable=True, checks=Check.ge(0)),
        },
        strict=True,
        coerce=True,
    )


def schema_us() -> pa.DataFrameSchema:
    """
    US dataset (used_cars.csv) schema contract.
    """
    return pa.DataFrameSchema(
        {
            "brand": pa.Column(pa.String, nullable=False),
            "model": pa.Column(pa.String, nullable=False),
            "model_year": pa.Column(pa.Int, nullable=False, checks=Check.between(1950, 2035)),
            "milage": pa.Column(pa.String, nullable=False),
            "fuel_type": pa.Column(pa.String, nullable=True),
            "engine": pa.Column(pa.String, nullable=False),
            "transmission": pa.Column(pa.String, nullable=False),
            "ext_col": pa.Column(pa.String, nullable=False),
            "int_col": pa.Column(pa.String, nullable=False),
            "accident": pa.Column(pa.String, nullable=True),
            "clean_title": pa.Column(pa.String, nullable=True),
            "price": pa.Column(pa.String, nullable=False),
        },
        strict=True,
        coerce=True,
    )


def _flag01_unknown_or_selfname(colname: str) -> Check:
    """
    TEMPLATE flags contain values like:
      0/1 (ints), '0'/'1' (strings), 'unknown',
      and sometimes the column name itself (e.g. 'GDI' in column GDI).

    Allow: 0/1/unknown/self-name or missing.
    """
    allowed = {"0", "1", "unknown", colname.strip().lower()}
    return Check(
        lambda s: s.isna() | s.astype(str).str.strip().str.lower().isin(allowed),
        name="flag_in_{0,1,unknown,self}",
    )


def schema_template() -> pa.DataFrameSchema:
    """
    TEMPLATE dataset (test_final.xlsx) schema contract.
    """
    return pa.DataFrameSchema(
        {
            "id": pa.Column(pa.Int, nullable=False),
            "brand": pa.Column(pa.String, nullable=False),
            "model": pa.Column(pa.String, nullable=False),
            "model_year": pa.Column(pa.Int, nullable=False, checks=Check.between(1950, 2035)),
            "milage": pa.Column(pa.String, nullable=False),
            "fuel_type": pa.Column(pa.String, nullable=True),
            "transmission": pa.Column(pa.String, nullable=False),
            "ext_col": pa.Column(pa.String, nullable=False),
            "int_col": pa.Column(pa.String, nullable=False),
            "accident": pa.Column(pa.String, nullable=True),
            "clean_title": pa.Column(pa.String, nullable=True),
            "power": pa.Column(pa.Float, nullable=True, checks=Check.ge(0)),
            "dispersion": pa.Column(pa.Float, nullable=True, checks=Check.ge(0)),
            "battery": pa.Column(pa.Float, nullable=True, checks=Check.ge(0)),
            "engine": pa.Column(pa.String, nullable=True),
            "GDI": pa.Column(pa.String, nullable=True, checks=_flag01_unknown_or_selfname("GDI")),
            "DOHC": pa.Column(pa.String, nullable=True, checks=_flag01_unknown_or_selfname("DOHC")),
            "Turbo": pa.Column(
                pa.String, nullable=True, checks=_flag01_unknown_or_selfname("Turbo")
            ),
            "MPFI": pa.Column(pa.String, nullable=True, checks=_flag01_unknown_or_selfname("MPFI")),
            "PDI": pa.Column(pa.String, nullable=True, checks=_flag01_unknown_or_selfname("PDI")),
            "OHV": pa.Column(pa.String, nullable=True, checks=_flag01_unknown_or_selfname("OHV")),
            "SOHC": pa.Column(pa.String, nullable=True, checks=_flag01_unknown_or_selfname("SOHC")),
        },
        strict=True,
        coerce=True,
    )
