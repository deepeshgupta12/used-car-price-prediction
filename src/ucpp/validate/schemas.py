from __future__ import annotations

import pandera.pandas as pa
from pandera import Check


def schema_in() -> pa.DataFrameSchema:
    """
    India dataset (Cars.csv) schema contract.

    Observed columns:
    ['Name','Location','Year','Kilometers_Driven','Fuel_Type','Transmission','Owner_Type',
     'Mileage','Engine','Power','Colour','Seats','No. of Doors','New_Price','Price']

    Note: Year and Kilometers_Driven contain NaNs in this dataset.
    We therefore use pandas nullable integer dtype "Int64" to allow missing values.
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

    Observed columns:
    ['brand','model','model_year','milage','fuel_type','engine','transmission',
     'ext_col','int_col','accident','clean_title','price']

    Note: model_year includes at least one value 1974 in current dataset.
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


def schema_template() -> pa.DataFrameSchema:
    """
    TEMPLATE dataset (test_final.xlsx) schema contract.

    Observed columns:
    ['id','brand','model','model_year','milage','fuel_type','transmission','ext_col','int_col',
     'accident','clean_title','power','dispersion','battery','engine',
     'GDI','DOHC','Turbo','MPFI','PDI','OHV','SOHC']
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
            "GDI": pa.Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
            "DOHC": pa.Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
            "Turbo": pa.Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
            "MPFI": pa.Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
            "PDI": pa.Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
            "OHV": pa.Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
            "SOHC": pa.Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
        },
        strict=True,
        coerce=True,
    )
