from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import typer

app = typer.Typer(help="Generate charts from saved metrics.json files.")

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = ROOT / "artifacts" / "v1"
REPORTS_DIR = ROOT / "reports" / "v1"


@dataclass(frozen=True)
class MarketMetrics:
    market: str
    metrics: dict[str, dict[str, float]]  # model -> metric_name -> value


def _load_metrics(market: str) -> MarketMetrics:
    m = market.lower()
    path = ARTIFACTS_DIR / m / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"metrics.json not found: {path}")

    data: dict[str, Any] = json.loads(path.read_text())
    # normalize: ensure float values
    cleaned: dict[str, dict[str, float]] = {}
    for model_name, model_metrics in data.items():
        cleaned[model_name] = {k: float(v) for k, v in model_metrics.items()}
    return MarketMetrics(market=market.upper(), metrics=cleaned)


def _best_model(mm: MarketMetrics, metric: str) -> tuple[str, float]:
    best_name = ""
    best_val = float("inf")
    for model_name, d in mm.metrics.items():
        v = float(d[metric])
        if v < best_val:
            best_val = v
            best_name = model_name
    return best_name, best_val


def _plot_bar(mm: MarketMetrics, metric: str, out_path: Path) -> None:
    models = list(mm.metrics.keys())
    values = [mm.metrics[m][metric] for m in models]

    plt.figure()
    plt.bar(models, values)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(metric.upper())
    plt.title(f"{mm.market} - {metric.upper()} by model")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _write_summary(in_mm: MarketMetrics | None, us_mm: MarketMetrics | None) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "metrics_summary.md"

    lines: list[str] = []
    lines.append("# V1 Baselines - Metrics Summary")
    lines.append("")

    def add_market(mm: MarketMetrics) -> None:
        lines.append(f"## {mm.market}")
        lines.append("")
        lines.append("| Model | MAE | RMSE | SMAPE |")
        lines.append("|---|---:|---:|---:|")
        for model_name, d in mm.metrics.items():
            lines.append(
                f"| `{model_name}` | {d['mae']:.6f} | {d['rmse']:.6f} | {d['smape']:.6f} |"
            )
        lines.append("")
        for metric in ["mae", "rmse", "smape"]:
            name, val = _best_model(mm, metric)
            lines.append(f"- Best {metric.upper()}: `{name}` = {val:.6f}")
        lines.append("")

    if in_mm is not None:
        add_market(in_mm)
    if us_mm is not None:
        add_market(us_mm)

    out.write_text("\n".join(lines))


@app.command()
def run(
    market: str = typer.Option("ALL", help="IN, US, or ALL", show_default=True),
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    m = market.upper().strip()
    in_mm: MarketMetrics | None = None
    us_mm: MarketMetrics | None = None

    if m in {"IN", "ALL"}:
        in_mm = _load_metrics("IN")
        _plot_bar(in_mm, "mae", REPORTS_DIR / "in_mae.png")
        _plot_bar(in_mm, "rmse", REPORTS_DIR / "in_rmse.png")
        _plot_bar(in_mm, "smape", REPORTS_DIR / "in_smape.png")

    if m in {"US", "ALL"}:
        us_mm = _load_metrics("US")
        _plot_bar(us_mm, "mae", REPORTS_DIR / "us_mae.png")
        _plot_bar(us_mm, "rmse", REPORTS_DIR / "us_rmse.png")
        _plot_bar(us_mm, "smape", REPORTS_DIR / "us_smape.png")

    _write_summary(in_mm, us_mm)

    typer.echo(f"Wrote charts + summary to: {REPORTS_DIR}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
