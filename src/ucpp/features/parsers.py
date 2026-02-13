from __future__ import annotations

import re

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def _first_number(text: str) -> float | None:
    m = _NUM_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


# -------------------------
# US parsing
# -------------------------
def parse_us_price_to_float(price: str) -> float | None:
    """
    used_cars.csv price examples can contain symbols/commas.
    Returns USD numeric.
    """
    if price is None:
        return None
    s = str(price).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    s = s.replace(",", "")
    v = _first_number(s)
    return float(v) if v is not None else None


def parse_us_milage_to_int(milage: str) -> int | None:
    """
    used_cars.csv column is named 'milage' (typo).
    Examples: '45,123 mi', '45123', etc.
    Returns miles as int.
    """
    if milage is None:
        return None
    s = str(milage).strip().lower()
    if s == "" or s in {"nan", "none"}:
        return None
    s = s.replace(",", "")
    v = _first_number(s)
    return int(round(v)) if v is not None else None


# -------------------------
# IN parsing
# -------------------------
def parse_in_mileage_kmpl(mileage: str) -> float | None:
    """
    Cars.csv Mileage examples: '18.9 kmpl', '14 km/kg', etc.
    Returns numeric mileage (unit-agnostic for baseline).
    """
    if mileage is None:
        return None
    s = str(mileage).strip().lower()
    if s == "" or s in {"nan", "none"}:
        return None
    v = _first_number(s)
    return float(v) if v is not None else None


def parse_in_engine_cc(engine: str) -> float | None:
    """
    Cars.csv Engine examples: '1197 CC'
    Returns engine displacement in CC.
    """
    if engine is None:
        return None
    s = str(engine).strip().lower()
    if s == "" or s in {"nan", "none"}:
        return None
    v = _first_number(s)
    return float(v) if v is not None else None


def parse_in_power_bhp(power: str) -> float | None:
    """
    Cars.csv Power examples: '82 bhp'
    Returns numeric bhp.
    """
    if power is None:
        return None
    s = str(power).strip().lower()
    if s == "" or s in {"nan", "none"}:
        return None
    v = _first_number(s)
    return float(v) if v is not None else None


def parse_in_new_price_lakh(new_price: str) -> float | None:
    """
    Cars.csv New_Price examples often like '10.5 Lakh' or '1.2 Crore'
    Convert to Lakhs for a consistent numeric scale.
    """
    if new_price is None:
        return None
    s = str(new_price).strip().lower()
    if s == "" or s in {"nan", "none"}:
        return None

    v = _first_number(s)
    if v is None:
        return None

    if "crore" in s:
        return float(v) * 100.0  # 1 Crore = 100 Lakhs
    if "lakh" in s or "lakhs" in s:
        return float(v)
    return float(v)


def parse_template_flag(x: object, flag_name: str) -> int | None:
    """
    TEMPLATE flags can be 0/1, '0'/'1', 'unknown', or the flag-name itself (e.g., 'GDI').
    Baseline rule:
      - 1 if value indicates presence (1 or flag-name)
      - 0 if value is 0
      - None if unknown/missing
    """
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"", "nan", "none"}:
        return None
    if s == "unknown":
        return None
    if s == "0":
        return 0
    if s == "1":
        return 1
    if s == flag_name.strip().lower():
        return 1
    return None
