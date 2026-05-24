import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from structured_eval.evaluate import compare_summaries


console = Console()


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two eval summary.json files.")
    parser.add_argument("baseline", type=Path, help="Baseline summary.json path.")
    parser.add_argument("candidate", type=Path, help="Candidate summary.json path.")
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show only the largest absolute deltas.",
    )
    return parser.parse_args()


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def run() -> None:
    args = parse_args()
    baseline = load_summary(args.baseline)
    candidate = load_summary(args.candidate)
    rows = compare_summaries(baseline, candidate)
    rows.sort(key=lambda row: abs(row["delta"]), reverse=True)
    if args.top is not None:
        rows = rows[: args.top]

    console.print(f"[bold]Baseline:[/bold] {args.baseline}")
    console.print(f"[bold]Candidate:[/bold] {args.candidate}")
    console.print(
        "[bold]Overall delta:[/bold] "
        f"{candidate.get('overall_mean_score', 0) - baseline.get('overall_mean_score', 0):+.3f}"
    )

    table = Table(title="Run Comparison")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Delta", justify="right")

    for row in rows:
        table.add_row(
            row["metric"],
            format_value(row["baseline"]),
            format_value(row["candidate"]),
            f"{row['delta']:+.3f}",
        )
    console.print(table)


if __name__ == "__main__":
    run()
