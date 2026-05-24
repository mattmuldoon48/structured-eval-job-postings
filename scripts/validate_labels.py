from pathlib import Path
import sys

from rich.console import Console
from structured_eval.validate_labels import validate_label_file


LABELED_PATH = Path(__file__).resolve().parents[1] / "data" / "labeled" / "labeled_jobs.jsonl"
console = Console()


def run() -> None:
    if not LABELED_PATH.exists():
        console.print(f"[yellow]No labeled data found at {LABELED_PATH}.[/yellow]")
        return

    errors = validate_label_file(LABELED_PATH)
    if not errors:
        console.print(f"[green]All labeled records are valid. ({LABELED_PATH})[/green]")
        return

    console.print(f"[red]Found {len(errors)} invalid record(s):[/red]")
    for index, message in errors:
        console.print(f"  - line {index}: {message}")
    sys.exit(1)


if __name__ == "__main__":
    run()
