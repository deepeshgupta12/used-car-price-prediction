from __future__ import annotations

import pandera as pa
from pandera import Check, Column, DataFrameSchema


def schema_in() -> DataFrameSchema:
    """
    India dataset (Cars.csv) schema contract.
    Columns observed:
    ['Name','Location','Year','Kilometers_Driven','Fuel_Type','Transmission','Owner_Type',
     'Mileage','Engine','Power','Colour','Seats','No. of Doors','New_Price','Price']
    """
    return DataFrameSchema(
        {
            "Name": Column(pa.String, nullable=True),
            "Location": Column(pa.String, nullable=True),
            "Year": Column(pa.Int, nullable=True, checks=Check.between(1980, 2030)),
            "Kilometers_Driven": Column(pa.Int, nullable=True, checks=Check.ge(0)),
            "Fuel_Type": Column(pa.String, nullable=True),
            "Transmission": Column(pa.String, nullable=True),
            "Owner_Type": Column(pa.String, nullable=True),
            "Mileage": Column(pa.String, nullable=True),
            "Engine": Column(pa.String, nullable=True),
            "Power": Column(pa.String, nullable=True),
            "Colour": Column(pa.String, nullable=True),
            "Seats": Column(pa.Float, nullable=True, checks=Check.between(1, 12)),
            "No. of Doors": Column(pa.Float, nullable=True, checks=Check.between(2, 10)),
            "New_Price": Column(pa.String, nullable=True),
            "Price": Column(pa.Float, nullable=True, checks=Check.ge(0)),
        },
        strict=True,
        coerce=True,
    )


def schema_us() -> DataFrameSchema:
    """
    US dataset (used_cars.csv) schema contract.
    Columns observed:
    ['brand','model','model_year','milage','fuel_type','engine','transmission',
     'ext_col','int_col','accident','clean_title','price']
    """
    return DataFrameSchema(
        {
            "brand": Column(pa.String, nullable=False),
            "model": Column(pa.String, nullable=False),
            "model_year": Column(pa.Int, nullable=False, checks=Check.between(1980, 2030)),
            "milage": Column(pa.String, nullable=False),
            "fuel_type": Column(pa.String, nullable=True),
            "engine": Column(pa.String, nullable=False),
            "transmission": Column(pa.String, nullable=False),
            "ext_col": Column(pa.String, nullable=False),
            "int_col": Column(pa.String, nullable=False),
            "accident": Column(pa.String, nullable=True),
            "clean_title": Column(pa.String, nullable=True),
            "price": Column(pa.String, nullable=False),
        },
        strict=True,
        coerce=True,
    )


def schema_template() -> DataFrameSchema:
    """
    TEMPLATE dataset (test_final.xlsx) schema contract.
    Columns observed:
    ['id','brand','model','model_year','milage','fuel_type','transmission','ext_col','int_col',
     'accident','clean_title','power','dispersion','battery','engine',
     'GDI','DOHC','Turbo','MPFI','PDI','OHV','SOHC']
    """
    return DataFrameSchema(
        {
            "id": Column(pa.Int, nullable=False),
            "brand": Column(pa.String, nullable=False),
            "model": Column(pa.String, nullable=False),
            "model_year": Column(pa.Int, nullable=False, checks=Check.between(1980, 2030)),
            "milage": Column(pa.String, nullable=False),
            "fuel_type": Column(pa.String, nullable=True),
            "transmission": Column(pa.String, nullable=False),
            "ext_col": Column(pa.String, nullable=False),
            "int_col": Column(pa.String, nullable=False),
            "accident": Column(pa.String, nullable=True),
            "clean_title": Column(pa.String, nullable=True),
            "power": Column(pa.Float, nullable=True, checks=Check.ge(0)),
            "dispersion": Column(pa.Float, nullable=True, checks=Check.ge(0)),
            "battery": Column(pa.Float, nullable=True, checks=Check.ge(0)),
            "engine": Column(pa.String, nullable=True),
            "GDI": Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
            "DOHC": Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
            "Turbo": Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
            "MPFI": Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
            "PDI": Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
            "OHV": Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
            "SOHC": Column(pa.Int, nullable=False, checks=Check.isin([0, 1])),
        },
        strict=True,
        coerce=True,
    )
