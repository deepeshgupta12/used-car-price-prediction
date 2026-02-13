from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Market = Literal["IN", "US"]


def _strip_or_none(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    return s


class _BasePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InPayload(_BasePayload):
    """
    Matches raw fields expected by transform_in() in src/ucpp/features/preprocess.py.
    We also allow optional Price for convenience (transform_in drops it anyway).
    """

    Name: str | None = Field(default=None)
    Location: str | None = Field(default=None)
    Year: int | float | None = Field(default=None)
    Kilometers_Driven: int | float | None = Field(default=None)
    Fuel_Type: str | None = Field(default=None)
    Transmission: str | None = Field(default=None)
    Owner_Type: str | None = Field(default=None)
    Colour: str | None = Field(default=None)
    Seats: int | float | None = Field(default=None)
    No__of__Doors: int | float | None = Field(default=None, alias="No. of Doors")

    Mileage: str | None = Field(default=None)
    Engine: str | None = Field(default=None)
    Power: str | None = Field(default=None)
    New_Price: str | None = Field(default=None)

    Price: float | None = Field(default=None)

    def cleaned_dict(self) -> dict[str, Any]:
        d = self.model_dump(by_alias=True)
        # normalize string-like fields
        for k in ["Name", "Location", "Fuel_Type", "Transmission", "Owner_Type", "Colour"]:
            d[k] = _strip_or_none(d.get(k))
        for k in ["Mileage", "Engine", "Power", "New_Price"]:
            d[k] = _strip_or_none(d.get(k))
        return d


class UsPayload(_BasePayload):
    """
    Matches raw fields expected by transform_us() in src/ucpp/features/preprocess.py.
    We also allow optional price for convenience (transform_us drops it anyway).
    """

    brand: str | None = Field(default=None)
    model: str | None = Field(default=None)
    model_year: int | float | None = Field(default=None)
    fuel_type: str | None = Field(default=None)
    engine: str | None = Field(default=None)
    transmission: str | None = Field(default=None)
    ext_col: str | None = Field(default=None)
    int_col: str | None = Field(default=None)
    accident: str | None = Field(default=None)
    clean_title: str | None = Field(default=None)
    milage: str | None = Field(default=None)

    price: str | float | None = Field(default=None)

    def cleaned_dict(self) -> dict[str, Any]:
        d = self.model_dump()
        for k in [
            "brand",
            "model",
            "fuel_type",
            "engine",
            "transmission",
            "ext_col",
            "int_col",
            "accident",
            "clean_title",
            "milage",
        ]:
            d[k] = _strip_or_none(d.get(k))
        return d


def validate_payload(market: str, payload: dict[str, Any]) -> dict[str, Any]:
    m = market.upper()
    if m == "IN":
        obj = InPayload.model_validate(payload)
        return obj.cleaned_dict()
    if m == "US":
        obj = UsPayload.model_validate(payload)
        return obj.cleaned_dict()
    raise ValueError("market must be IN or US")
