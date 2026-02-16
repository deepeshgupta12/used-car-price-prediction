from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

Market = Literal["IN", "US"]


def _strip_or_none(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    return s


def _require_any_non_null(
    *,
    market: str,
    cleaned: dict[str, Any],
    required_any_of: list[str],
) -> None:
    """
    Guardrail for batch (and single) inference:
    - We still allow partial rows (many fields optional).
    - But we reject completely empty / unusable rows (e.g. {} or all-null after cleaning).

    Raises pydantic.ValidationError so existing callers (API + batch CLI) treat it as schema invalid.
    """
    if any(cleaned.get(k) is not None for k in required_any_of):
        return

    msg = f"empty payload: provide at least one of {required_any_of} for market={market}"

    # Create a pydantic.ValidationError so existing handlers catch it as schema validation failure.
    raise ValidationError.from_exception_data(
        "Payload",
        [
            {
                "type": "value_error",
                "loc": ("payload",),
                "msg": msg,
                "input": cleaned,
            }
        ],
    )


class _BasePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InPayload(_BasePayload):
    """
    Matches raw fields expected by transform_in().

    We keep the original raw field names so JSON/CSV rows can be used as-is.
    Note: "No. of Doors" is represented via alias.
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
        # Use aliases so downstream sees "No. of Doors"
        d = self.model_dump(by_alias=True)

        for k in ["Name", "Location", "Fuel_Type", "Transmission", "Owner_Type", "Colour"]:
            d[k] = _strip_or_none(d.get(k))

        for k in ["Mileage", "Engine", "Power", "New_Price"]:
            d[k] = _strip_or_none(d.get(k))

        # "No. of Doors" stays numeric; no string normalization needed.
        return d


class UsPayload(_BasePayload):
    """
    Matches raw fields expected by transform_us().
    All string-ish fields are optional to support real-world CSV missingness.
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

        # price can be str/float/None; leave it (transform_us drops it anyway)
        return d


def validate_payload(market: str, payload: dict[str, Any]) -> dict[str, Any]:
    m = market.upper()

    if m == "IN":
        obj = InPayload.model_validate(payload)
        clean = obj.cleaned_dict()

        # Minimal viability: reject fully empty rows ({} or all-null after cleaning)
        _require_any_non_null(
            market="IN",
            cleaned=clean,
            required_any_of=[
                "Name",
                "Location",
                "Year",
                "Kilometers_Driven",
                "Fuel_Type",
                "Transmission",
            ],
        )
        return clean

    if m == "US":
        obj = UsPayload.model_validate(payload)
        clean = obj.cleaned_dict()

        _require_any_non_null(
            market="US",
            cleaned=clean,
            required_any_of=["brand", "model", "model_year", "milage"],
        )
        return clean

    raise ValueError("market must be IN or US")
