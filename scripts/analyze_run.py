import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from structured_eval.evaluate import example_mismatches, load_jsonl
from structured_eval.schema import JobPostingLabel


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize field-level mismatches from an eval predictions.jsonl file.")
    parser.add_argument("predictions", type=Path, help="Path to an eval predictions.jsonl file.")
    parser.add_argument(
        "--examples-per-field",
        type=int,
        default=3,
        help="Maximum example mismatches to show per field.",
    )
    return parser.parse_args()


def collect_mismatches(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        expected = JobPostingLabel.model_validate(record["expected"])
        actual = JobPostingLabel.model_validate(record["actual"])
        for mismatch in example_mismatches(expected, actual):
            grouped[mismatch["field"]].append({"id": record["id"], **mismatch})
    return dict(grouped)


def render_mismatch_summary(grouped: dict[str, list[dict[str, Any]]], examples_per_field: int) -> None:
    table = Table(title="Mismatch Summary")
    table.add_column("Field")
    table.add_column("Mismatches", justify="right")
    table.add_column("Average Score", justify="right")

    rows = []
    for field_name, mismatches in grouped.items():
        average_score = sum(item["score"] for item in mismatches) / len(mismatches)
        rows.append((field_name, len(mismatches), average_score))
    for field_name, count, average_score in sorted(rows, key=lambda row: (-row[1], row[0])):
        table.add_row(field_name, str(count), f"{average_score:.3f}")
    console.print(table)

    for field_name, mismatches in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        console.print(f"\n[bold]{field_name}[/bold]")
        for mismatch in mismatches[:examples_per_field]:
            console.print(
                f"- {mismatch['id']} ({mismatch['metric']}, score {mismatch['score']:.3f}) "
                f"expected={mismatch['expected']!r} actual={mismatch['actual']!r}"
            )


def run() -> None:
    args = parse_args()
    grouped = collect_mismatches(load_jsonl(args.predictions))
    if not grouped:
        console.print("[green]No mismatches found.[/green]")
        return
    render_mismatch_summary(grouped, examples_per_field=args.examples_per_field)


if __name__ == "__main__":
    run()
