import sys
from pathlib import Path

from rich.console import Console

from structured_eval.dataset_checks import validate_dataset_integrity
from structured_eval.evaluate import load_jsonl


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "job_postings.jsonl"
LABELED_PATH = ROOT / "data" / "labeled" / "labeled_jobs.jsonl"
SPLITS_PATH = ROOT / "data" / "splits" / "job_splits.jsonl"
console = Console()


def run() -> None:
    raw_records = load_jsonl(RAW_PATH)
    labeled_records = load_jsonl(LABELED_PATH)
    split_records = load_jsonl(SPLITS_PATH)

    failures = validate_dataset_integrity(raw_records, labeled_records, split_records)
    if failures:
        console.print("[red]Dataset integrity checks failed:[/red]")
        for failure in failures:
            console.print(f"  - {failure}")
        sys.exit(1)

    console.print(
        "[green]Dataset integrity checks passed.[/green] "
        f"{len(raw_records)} raw, {len(labeled_records)} labeled, {len(split_records)} split assignments."
    )


if __name__ == "__main__":
    run()
