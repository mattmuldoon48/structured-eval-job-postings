import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from structured_eval.evaluate import EvalAccumulator, example_mismatches, load_jsonl
from structured_eval.label_assist import generate_draft_label
from structured_eval.llm_client import LLMClient
from structured_eval.schema import JobPostingLabel


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "job_postings.jsonl"
LABELED_PATH = ROOT / "data" / "labeled" / "labeled_jobs.jsonl"
PROMPTS_DIR = ROOT / "prompts"
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
    for field_name, score in summary["normalized_text_score"].items():
        table.add_row(field_name, "normalized_text_score", f"{score:.3f}")
    for field_name, score in summary["exact_list_f1"].items():
        table.add_row(field_name, "exact_list_f1", f"{score:.3f}")
    for field_name, score in summary["soft_list_f1"].items():
        table.add_row(field_name, "soft_list_f1", f"{score:.3f}")
    console.print(table)


def render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Structured Extraction Eval Report",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Model: `{summary['model']}`",
        f"- Prompt: `{summary['prompt']}`",
        f"- Examples scored: `{summary['examples']}`",
        f"- Requested examples: `{summary['requested_examples']}`",
        f"- Failed examples: `{summary['failed_examples']}`",
        f"- Overall mean score: `{summary['overall_mean_score']:.3f}`",
        "",
        "## Field Metrics",
        "",
        "| Field | Metric | Score |",
        "| --- | --- | ---: |",
    ]

    for field_name, score in summary["exact_accuracy"].items():
        lines.append(f"| `{field_name}` | exact accuracy | {score:.3f} |")
    for field_name, score in summary["normalized_text_score"].items():
        lines.append(f"| `{field_name}` | normalized text score | {score:.3f} |")
    for field_name, score in summary["exact_list_f1"].items():
        lines.append(f"| `{field_name}` | exact list F1 | {score:.3f} |")
    for field_name, score in summary["soft_list_f1"].items():
        lines.append(f"| `{field_name}` | soft list F1 | {score:.3f} |")

    if summary.get("sample_mismatches"):
        lines.extend(["", "## Sample Mismatches", ""])
        for item in summary["sample_mismatches"]:
            lines.extend(
                [
                    f"### `{item['id']}` - `{item['field']}`",
                    "",
                    f"- Metric: `{item['metric']}`",
                    f"- Score: `{item['score']:.3f}`",
                    f"- Expected: `{json.dumps(item['expected'], ensure_ascii=False)}`",
                    f"- Actual: `{json.dumps(item['actual'], ensure_ascii=False)}`",
                    "",
                ]
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Exact accuracy is intentionally strict and useful for enums, booleans, and numeric fields.",
            "- Normalized text score uses token overlap for company, title, and location fields.",
            "- Soft list F1 scores skill lists by best token-overlap matches instead of requiring identical strings.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structured extraction evals for labeled job postings.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of examples to evaluate.")
    parser.add_argument(
        "--prompt",
        default="extract_v2.txt",
        help="Prompt file under prompts/ or an explicit path.",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    joined = build_joined_records()
    if args.limit is not None:
        joined = joined[: args.limit]
    prompt_path = Path(args.prompt)
    if not prompt_path.is_absolute():
        prompt_path = PROMPTS_DIR / prompt_path
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    client = LLMClient.from_env()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = REPORTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    accumulator = EvalAccumulator()
    predictions_path = run_dir / "predictions.jsonl"
    errors_path = run_dir / "errors.jsonl"
    failures = 0
    sample_mismatches = []

    with predictions_path.open("w", encoding="utf-8") as stream:
        for index, record in enumerate(joined, start=1):
            job_id = record["id"]
            console.print(f"[cyan]Evaluating {job_id}[/cyan] ({index}/{len(joined)})")
            expected = JobPostingLabel.model_validate(record["expected"])
            try:
                actual = generate_draft_label(record["text"], client=client, prompt_path=prompt_path)
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
            for mismatch in example_mismatches(expected, actual):
                if len(sample_mismatches) < 12:
                    sample_mismatches.append({"id": job_id, **mismatch})
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
    summary["prompt"] = str(prompt_path)
    summary["requested_examples"] = len(joined)
    summary["failed_examples"] = failures
    summary["sample_mismatches"] = sample_mismatches
    summary["predictions_path"] = str(predictions_path)
    if errors_path.exists():
        summary["errors_path"] = str(errors_path)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path = run_dir / "report.md"
    markdown_path.write_text(render_markdown_report(summary), encoding="utf-8")

    render_summary(summary)
    console.print(f"[green]Wrote report:[/green] {summary_path}")
    console.print(f"[green]Wrote markdown:[/green] {markdown_path}")


if __name__ == "__main__":
    run()
