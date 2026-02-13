from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from pydantic import ValidationError

from ucpp.core.logging import info, warn
from ucpp.predict.predict import DEFAULT_MODEL_BY_MARKET, PredictResult, predict_one
from ucpp.predict.schema import validate_payload

app = typer.Typer(add_completion=False)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise typer.BadParameter(f"JSONL line {i} must be an object")
        rows.append(obj)
    return rows


def _read_input(path: Path) -> list[dict[str, Any]]:
    ext = path.suffix.lower()
    if ext == ".jsonl":
        return _read_jsonl(path)
    if ext == ".csv":
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
        return df.to_dict(orient="records")
    raise typer.BadParameter("input must be one of: .jsonl, .csv, .xlsx")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@app.command()
def run(
    market: str = typer.Option(..., help="IN or US"),
    input_path: Path = typer.Option(  # noqa: B008
        ..., exists=True, dir_okay=False, help="Input .jsonl/.csv/.xlsx"
    ),
    out_dir: Path = typer.Option(Path("outputs/v2"), help="Output directory"),  # noqa: B008
    model: str | None = typer.Option(None, help="Override model (default: best per market)"),
    artifacts_dir: Path = typer.Option(Path("artifacts"), help="Artifacts root"),  # noqa: B008
    strict: bool = typer.Option(
        True,
        help="If true: fail on first invalid row. If false: skip invalid rows and write errors.",
    ),
) -> None:
    """
    Batch inference:
      - Reads input (.jsonl/.csv/.xlsx)
      - Validates schema per row
      - Runs prediction
      - Writes:
          out_dir/preds.jsonl
          out_dir/preds.csv
          out_dir/errors.jsonl  (only when strict=false and errors exist)
    """
    market_u = market.upper()
    rows = _read_input(input_path)
    info(f"Loaded {len(rows)} rows from {input_path}")

    preds: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for idx, raw in enumerate(rows):
        try:
            clean = validate_payload(market_u, raw)
            chosen = model or DEFAULT_MODEL_BY_MARKET.get(market_u)
            if chosen is None:
                raise typer.BadParameter("market must be IN or US")

            res: PredictResult = predict_one(
                market=market_u,
                model=chosen,
                payload=clean,
                artifacts_dir=artifacts_dir,
            )
            preds.append(
                {
                    "row_index": idx,
                    "market": res.market,
                    "model": res.model,
                    "prediction": res.prediction,
                }
            )
        except ValidationError as ve:
            err = {"row_index": idx, "type": "validation_error", "detail": ve.errors()}
            if strict:
                raise typer.BadParameter(f"Row {idx} failed schema validation: {ve}") from ve
            errors.append(err)
        except Exception as e:
            err = {"row_index": idx, "type": "inference_error", "detail": str(e)}
            if strict:
                raise
            errors.append(err)

    out_dir.mkdir(parents=True, exist_ok=True)

    pred_jsonl = out_dir / "preds.jsonl"
    pred_csv = out_dir / "preds.csv"
    _write_jsonl(pred_jsonl, preds)
    pd.DataFrame(preds).to_csv(pred_csv, index=False)

    info(f"Wrote predictions: {pred_jsonl}")
    info(f"Wrote predictions: {pred_csv}")

    if errors:
        warn(f"{len(errors)} rows failed. Writing errors.jsonl")
        err_path = out_dir / "errors.jsonl"
        _write_jsonl(err_path, errors)
        info(f"Wrote errors: {err_path}")


if __name__ == "__main__":
    app()
