from __future__ import annotations

from pathlib import Path

import typer
from typer import Context

from ucpp import __version__
from ucpp.core.constants import RAW_DIR
from ucpp.core.logging import info, warn
from ucpp.ingest.loaders import load_cars_csv, load_test_final_xlsx, load_used_cars_csv
from ucpp.validate.basic_checks import check_columns_present, check_non_empty, summarize
from ucpp.validate.validate import validate_df

app = typer.Typer(help="UCPP CLI", no_args_is_help=False)


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


@app.command("validate-data")
def validate_data(
    cars_csv: Path = RAW_DIR / "Cars.csv",
    used_cars_csv: Path = RAW_DIR / "used_cars.csv",
    test_final_xlsx: Path = RAW_DIR / "test_final.xlsx",
) -> None:
    """
    Validates datasets against strict schema contracts (Pandera).
    """
    info(f"Validating IN: {cars_csv}")
    df_in = load_cars_csv(cars_csv)
    _, out_in = validate_df(df_in, "IN")
    if not out_in.ok:
        raise typer.Exit(code=1)

    info(f"Validating US: {used_cars_csv}")
    df_us = load_used_cars_csv(used_cars_csv)
    _, out_us = validate_df(df_us, "US")
    if not out_us.ok:
        raise typer.Exit(code=1)

    info(f"Validating TEMPLATE: {test_final_xlsx}")
    df_tpl = load_test_final_xlsx(test_final_xlsx)
    _, out_tpl = validate_df(df_tpl, "TEMPLATE")
    if not out_tpl.ok:
        raise typer.Exit(code=1)

    info("All schema validations passed.")


@app.command("version")
def version() -> None:
    """Print package version."""
    info(f"ucpp version: {__version__}")


@app.callback(invoke_without_command=True)
def default(
    ctx: Context,
    cars_csv: Path = RAW_DIR / "Cars.csv",
    used_cars_csv: Path = RAW_DIR / "used_cars.csv",
    test_final_xlsx: Path = RAW_DIR / "test_final.xlsx",
) -> None:
    """
    Default behavior: if no subcommand is provided, run `verify-data`.
    Also supports passing dataset paths directly without specifying the command.
    """
    if ctx.invoked_subcommand is None:
        verify_data(cars_csv=cars_csv, used_cars_csv=used_cars_csv, test_final_xlsx=test_final_xlsx)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
