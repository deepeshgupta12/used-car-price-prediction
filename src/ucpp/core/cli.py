from __future__ import annotations

from pathlib import Path

import typer

from ucpp.core.constants import RAW_DIR
from ucpp.core.logging import info, warn
from ucpp.ingest.loaders import load_cars_csv, load_test_final_xlsx, load_used_cars_csv
from ucpp.validate.basic_checks import check_columns_present, check_non_empty, summarize

app = typer.Typer(help="UCPP CLI")


@app.command("verify-data")
def verify_data(
    cars_csv: Path = RAW_DIR / "Cars.csv",
    used_cars_csv: Path = RAW_DIR / "used_cars.csv",
    test_final_xlsx: Path = RAW_DIR / "test_final.xlsx",
) -> None:
    """
    Loads datasets and runs basic sanity checks.
    """
    info(f"Reading: {cars_csv}")
    df_in = load_cars_csv(cars_csv)
    info(f"IN dataset summary: {summarize(df_in)}")

    info(f"Reading: {used_cars_csv}")
    df_us = load_used_cars_csv(used_cars_csv)
    info(f"US dataset summary: {summarize(df_us)}")

    info(f"Reading: {test_final_xlsx}")
    df_tpl = load_test_final_xlsx(test_final_xlsx)
    info(f"TEMPLATE dataset summary: {summarize(df_tpl)}")

    results = []
    results.append(check_non_empty(df_in, "cars_in"))
    results.append(check_non_empty(df_us, "used_us"))
    results.append(check_non_empty(df_tpl, "template"))

    # Minimal expected columns (we’ll tighten after we inspect column names in V0)
    results.append(check_columns_present(df_in, "cars_in", ["Year", "Price"]))
    results.append(check_columns_present(df_us, "used_us", ["model_year", "price"]))

    for r in results:
        if r.ok:
            info(f"{r.name} OK ({r.details})")
        else:
            warn(f"{r.name} FAIL ({r.details})")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
