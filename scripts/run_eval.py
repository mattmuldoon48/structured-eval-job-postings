import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from structured_eval.evaluate import (
    EvalAccumulator,
    estimate_cost,
    evaluate_quality_gates,
    example_mismatches,
    load_jsonl,
    score_prediction_records,
    summarize_usage,
)
from structured_eval.label_assist import generate_draft_label_with_usage
from structured_eval.llm_client import LLMClient
from structured_eval.schema import JobPostingLabel
from structured_eval.splits import load_split_map


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "job_postings.jsonl"
LABELED_PATH = ROOT / "data" / "labeled" / "labeled_jobs.jsonl"
SPLITS_PATH = ROOT / "data" / "splits" / "job_splits.jsonl"
PROMPTS_DIR = ROOT / "prompts"
REPORTS_DIR = ROOT / "reports" / "runs"
console = Console()


def build_joined_records(split_name: str | None = None) -> list[dict[str, Any]]:
    raw_by_id = {record["id"]: record for record in load_jsonl(RAW_PATH)}
    labels = load_jsonl(LABELED_PATH)
    split_map = load_split_map(SPLITS_PATH) if split_name else {}
    joined = []
    for label in labels:
        job_id = label["id"]
        if job_id not in raw_by_id:
            raise ValueError(f"Missing raw posting for labeled id {job_id}")
        record_split = split_map.get(job_id)
        if split_name and record_split != split_name:
            continue
        joined.append(
            {
                "id": job_id,
                "text": raw_by_id[job_id]["text"],
                "expected": label,
                "split": record_split,
            }
        )
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

    gate_config = summary.get("quality_gates") or {}
    gates_configured = gate_config.get("min_overall") is not None or bool(gate_config.get("min_metrics"))
    if summary.get("quality_gate_failures"):
        console.print("[red]Quality gates failed:[/red]")
        for failure in summary["quality_gate_failures"]:
            console.print(f"  - {failure}")
    elif gates_configured:
        console.print("[green]Quality gates passed.[/green]")

    if summary.get("usage"):
        usage = summary["usage"]
        console.print(
            "[bold]Usage:[/bold] "
            f"{usage['total_tokens']} tokens, "
            f"{usage['average_latency_seconds']:.2f}s avg latency"
        )
    if summary.get("cost_estimate"):
        cost = summary["cost_estimate"]
        console.print(f"[bold]Estimated cost:[/bold] ${cost['total_cost_usd']:.6f}")


def render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Structured Extraction Eval Report",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Model: `{summary['model']}`",
        f"- Prompt: `{summary['prompt']}`",
        f"- Split: `{summary.get('split')}`",
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

    if summary.get("quality_gates"):
        lines.extend(["", "## Quality Gates", ""])
        lines.append(f"- Minimum overall score: `{summary['quality_gates'].get('min_overall')}`")
        if summary["quality_gates"].get("min_metrics"):
            lines.append("- Minimum metric gates:")
            for gate in summary["quality_gates"]["min_metrics"]:
                lines.append(f"  - `{gate}`")
        if summary.get("quality_gate_failures"):
            lines.append("- Status: `failed`")
            for failure in summary["quality_gate_failures"]:
                lines.append(f"  - {failure}")
        else:
            lines.append("- Status: `passed`")

    if summary.get("usage"):
        usage = summary["usage"]
        lines.extend(
            [
                "",
                "## Usage",
                "",
                f"- Examples with usage: `{usage['examples']}`",
                f"- Prompt tokens: `{usage['prompt_tokens']}`",
                f"- Completion tokens: `{usage['completion_tokens']}`",
                f"- Total tokens: `{usage['total_tokens']}`",
                f"- Total latency seconds: `{usage['total_latency_seconds']:.2f}`",
                f"- Average latency seconds: `{usage['average_latency_seconds']:.2f}`",
            ]
        )
    if summary.get("cost_estimate"):
        cost = summary["cost_estimate"]
        lines.extend(
            [
                "",
                "## Cost Estimate",
                "",
                f"- Input cost per 1M tokens: `${cost['input_cost_per_1m']}`",
                f"- Output cost per 1M tokens: `${cost['output_cost_per_1m']}`",
                f"- Input cost USD: `${cost['input_cost_usd']:.6f}`",
                f"- Output cost USD: `${cost['output_cost_usd']:.6f}`",
                f"- Total cost USD: `${cost['total_cost_usd']:.6f}`",
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
    parser.add_argument("--split", default=None, help="Evaluate only records assigned to this split.")
    parser.add_argument(
        "--replay-predictions",
        default=None,
        help="Re-score an existing predictions.jsonl file without making model calls.",
    )
    parser.add_argument(
        "--prompt",
        default="extract_v2.txt",
        help="Prompt file under prompts/ or an explicit path.",
    )
    parser.add_argument("--min-overall", type=float, default=None, help="Fail if overall score is below this value.")
    parser.add_argument(
        "--min-metric",
        action="append",
        default=[],
        help="Fail if a metric is below a threshold, e.g. exact_accuracy.remote_policy=0.80.",
    )
    parser.add_argument(
        "--input-cost-per-1m",
        type=float,
        default=None,
        help="Input token cost in USD per 1M tokens for optional cost estimation.",
    )
    parser.add_argument(
        "--output-cost-per-1m",
        type=float,
        default=None,
        help="Output token cost in USD per 1M tokens for optional cost estimation.",
    )
    return parser.parse_args()


def resolve_prompt_path(prompt: str) -> Path:
    prompt_path = Path(prompt)
    if not prompt_path.is_absolute():
        prompt_path = PROMPTS_DIR / prompt_path
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path


def write_run_outputs(
    run_dir: Path,
    summary: dict[str, Any],
    prediction_records: list[dict[str, Any]],
) -> None:
    predictions_path = run_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as stream:
        for record in prediction_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary["predictions_path"] = str(predictions_path)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path = run_dir / "report.md"
    markdown_path.write_text(render_markdown_report(summary), encoding="utf-8")

    render_summary(summary)
    console.print(f"[green]Wrote report:[/green] {summary_path}")
    console.print(f"[green]Wrote markdown:[/green] {markdown_path}")


def build_summary(
    metric_summary: dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
    model: str,
    prompt: str,
    split: str | None,
    requested_examples: int,
    failed_examples: int,
    sample_mismatches: list[dict[str, Any]],
    usage_records: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = metric_summary
    summary["run_id"] = run_id
    summary["model"] = model
    summary["prompt"] = prompt
    summary["split"] = split
    summary["requested_examples"] = requested_examples
    summary["failed_examples"] = failed_examples
    summary["sample_mismatches"] = sample_mismatches
    summary["usage"] = summarize_usage(usage_records)
    summary["cost_estimate"] = estimate_cost(
        summary["usage"],
        input_cost_per_1m=args.input_cost_per_1m,
        output_cost_per_1m=args.output_cost_per_1m,
    )
    summary["quality_gates"] = {
        "min_overall": args.min_overall,
        "min_metrics": args.min_metric,
    }
    summary["quality_gate_failures"] = evaluate_quality_gates(
        summary,
        min_overall=args.min_overall,
        metric_gates=args.min_metric,
    )
    return summary


def run_live_eval(args: argparse.Namespace, run_id: str, run_dir: Path) -> dict[str, Any]:
    joined = build_joined_records(split_name=args.split)
    if args.limit is not None:
        joined = joined[: args.limit]
    prompt_path = resolve_prompt_path(args.prompt)
    client = LLMClient.from_env()

    accumulator = EvalAccumulator()
    errors_path = run_dir / "errors.jsonl"
    failures = 0
    sample_mismatches = []
    usage_records = []
    prediction_records = []

    for index, record in enumerate(joined, start=1):
        job_id = record["id"]
        console.print(f"[cyan]Evaluating {job_id}[/cyan] ({index}/{len(joined)})")
        expected = JobPostingLabel.model_validate(record["expected"])
        try:
            actual, llm_result = generate_draft_label_with_usage(
                record["text"],
                client=client,
                prompt_path=prompt_path,
            )
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
        usage = llm_result.usage_dict()
        usage_records.append(usage)
        prediction_records.append(
            {
                "id": job_id,
                "split": record.get("split"),
                "expected": expected.model_dump(mode="json"),
                "actual": actual.model_dump(mode="json"),
                "usage": usage,
            }
        )

    summary = build_summary(
        accumulator.summary(),
        args=args,
        run_id=run_id,
        model=client.model,
        prompt=str(prompt_path),
        split=args.split,
        requested_examples=len(joined),
        failed_examples=failures,
        sample_mismatches=sample_mismatches,
        usage_records=usage_records,
    )
    if errors_path.exists():
        summary["errors_path"] = str(errors_path)
    write_run_outputs(run_dir, summary, prediction_records)
    return summary


def run_replay_eval(args: argparse.Namespace, run_id: str, run_dir: Path) -> dict[str, Any]:
    replay_path = Path(args.replay_predictions)
    records = load_jsonl(replay_path)
    if args.split:
        records = [record for record in records if record.get("split") == args.split]
    if args.limit is not None:
        records = records[: args.limit]
    metric_summary, sample_mismatches, usage_records = score_prediction_records(records)
    summary = build_summary(
        metric_summary,
        args=args,
        run_id=run_id,
        model="replay",
        prompt=f"replay:{replay_path}",
        split=args.split,
        requested_examples=len(records),
        failed_examples=0,
        sample_mismatches=sample_mismatches,
        usage_records=usage_records,
    )
    summary["replay_source_path"] = str(replay_path)
    write_run_outputs(run_dir, summary, records)
    return summary


def run() -> None:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = REPORTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.replay_predictions:
        summary = run_replay_eval(args, run_id, run_dir)
    else:
        summary = run_live_eval(args, run_id, run_dir)

    if summary["quality_gate_failures"] or summary["failed_examples"]:
        sys.exit(1)


if __name__ == "__main__":
    run()
