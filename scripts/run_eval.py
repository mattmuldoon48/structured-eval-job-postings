import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from structured_eval.evaluate import EvalAccumulator, load_jsonl
from structured_eval.label_assist import generate_draft_label
from structured_eval.llm_client import LLMClient
from structured_eval.schema import JobPostingLabel


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "job_postings.jsonl"
LABELED_PATH = ROOT / "data" / "labeled" / "labeled_jobs.jsonl"
REPORTS_DIR = ROOT / "reports" / "runs"
console = Console()


def build_joined_records() -> list[dict[str, Any]]:
    raw_by_id = {record["id"]: record for record in load_jsonl(RAW_PATH)}
    labels = load_jsonl(LABELED_PATH)
    joined = []
    for label in labels:
        job_id = label["id"]
        if job_id not in raw_by_id:
            raise ValueError(f"Missing raw posting for labeled id {job_id}")
        joined.append({"id": job_id, "text": raw_by_id[job_id]["text"], "expected": label})
    return joined


def render_summary(summary: dict[str, Any]) -> None:
    console.print(f"[bold]Examples:[/bold] {summary['examples']}")
    console.print(f"[bold]Overall mean score:[/bold] {summary['overall_mean_score']:.3f}")

    table = Table(title="Field Metrics")
    table.add_column("Field")
    table.add_column("Metric")
    table.add_column("Score", justify="right")

    for field_name, score in summary["exact_accuracy"].items():
        table.add_row(field_name, "exact_accuracy", f"{score:.3f}")
    for field_name, score in summary["list_f1"].items():
        table.add_row(field_name, "list_f1", f"{score:.3f}")
    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structured extraction evals for labeled job postings.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of examples to evaluate.")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    joined = build_joined_records()
    if args.limit is not None:
        joined = joined[: args.limit]
    client = LLMClient.from_env()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = REPORTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    accumulator = EvalAccumulator()
    predictions_path = run_dir / "predictions.jsonl"
    errors_path = run_dir / "errors.jsonl"
    failures = 0

    with predictions_path.open("w", encoding="utf-8") as stream:
        for index, record in enumerate(joined, start=1):
            job_id = record["id"]
            console.print(f"[cyan]Evaluating {job_id}[/cyan] ({index}/{len(joined)})")
            expected = JobPostingLabel.model_validate(record["expected"])
            try:
                actual = generate_draft_label(record["text"], client=client)
            except Exception as exc:
                failures += 1
                with errors_path.open("a", encoding="utf-8") as error_stream:
                    error_stream.write(
                        json.dumps(
                            {
                                "id": job_id,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                console.print(f"[red]Stopping after {job_id}: {type(exc).__name__}: {exc}[/red]")
                break
            accumulator.add(expected, actual)
            stream.write(
                json.dumps(
                    {
                        "id": job_id,
                        "expected": expected.model_dump(mode="json"),
                        "actual": actual.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = accumulator.summary()
    summary["run_id"] = run_id
    summary["model"] = client.model
    summary["requested_examples"] = len(joined)
    summary["failed_examples"] = failures
    summary["predictions_path"] = str(predictions_path)
    if errors_path.exists():
        summary["errors_path"] = str(errors_path)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    render_summary(summary)
    console.print(f"[green]Wrote report:[/green] {summary_path}")


if __name__ == "__main__":
    run()
