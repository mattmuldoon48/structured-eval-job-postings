import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from structured_eval.evaluate import compare_summaries


ROOT = Path(__file__).resolve().parents[1]
RUN_EVAL_PATH = ROOT / "scripts" / "run_eval.py"
REPORTS_DIR = ROOT / "reports" / "runs"
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dev and test extraction benchmarks with matching settings.")
    parser.add_argument(
        "--prompt",
        default="extract_v2.txt",
        help="Prompt file under prompts/ or an explicit path.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate per split.",
    )
    parser.add_argument("--min-overall", type=float, default=None, help="Pass-through overall quality gate.")
    parser.add_argument(
        "--min-metric",
        action="append",
        default=[],
        help="Pass-through metric gate, e.g. exact_accuracy.remote_policy=0.80.",
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


def build_eval_command(
    python_executable: str,
    split: str,
    prompt: str,
    limit_per_split: int | None = None,
    min_overall: float | None = None,
    min_metrics: list[str] | None = None,
    input_cost_per_1m: float | None = None,
    output_cost_per_1m: float | None = None,
) -> list[str]:
    command = [
        python_executable,
        str(RUN_EVAL_PATH),
        "--split",
        split,
        "--prompt",
        prompt,
    ]
    if limit_per_split is not None:
        command.extend(["--limit", str(limit_per_split)])
    if min_overall is not None:
        command.extend(["--min-overall", str(min_overall)])
    for metric_gate in min_metrics or []:
        command.extend(["--min-metric", metric_gate])
    if input_cost_per_1m is not None:
        command.extend(["--input-cost-per-1m", str(input_cost_per_1m)])
    if output_cost_per_1m is not None:
        command.extend(["--output-cost-per-1m", str(output_cost_per_1m)])
    return command


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def latest_summary_since(start_time: float) -> Path:
    summaries = [
        path
        for path in REPORTS_DIR.glob("*/summary.json")
        if path.stat().st_mtime >= start_time
    ]
    if not summaries:
        raise RuntimeError("Eval completed but no new summary.json was found.")
    return max(summaries, key=lambda path: path.stat().st_mtime)


def run_split(split: str, args: argparse.Namespace) -> Path:
    console.print(f"[bold cyan]Running {split} split[/bold cyan]")
    started_at = time.time()
    command = build_eval_command(
        python_executable=sys.executable,
        split=split,
        prompt=args.prompt,
        limit_per_split=args.limit_per_split,
        min_overall=args.min_overall,
        min_metrics=args.min_metric,
        input_cost_per_1m=args.input_cost_per_1m,
        output_cost_per_1m=args.output_cost_per_1m,
    )
    subprocess.run(command, cwd=ROOT, check=True)
    return latest_summary_since(started_at)


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_comparison(dev_summary_path: Path, test_summary_path: Path) -> None:
    dev_summary = load_summary(dev_summary_path)
    test_summary = load_summary(test_summary_path)
    rows = compare_summaries(dev_summary, test_summary)
    rows.sort(key=lambda row: abs(row["delta"]), reverse=True)

    console.print(f"[bold]Dev summary:[/bold] {dev_summary_path}")
    console.print(f"[bold]Test summary:[/bold] {test_summary_path}")
    console.print(
        "[bold]Test minus dev overall delta:[/bold] "
        f"{test_summary.get('overall_mean_score', 0) - dev_summary.get('overall_mean_score', 0):+.3f}"
    )

    table = Table(title="Dev vs Test Benchmark")
    table.add_column("Metric")
    table.add_column("Dev", justify="right")
    table.add_column("Test", justify="right")
    table.add_column("Delta", justify="right")

    for row in rows[:15]:
        table.add_row(
            row["metric"],
            format_value(row["baseline"]),
            format_value(row["candidate"]),
            f"{row['delta']:+.3f}",
        )
    console.print(table)


def run() -> None:
    args = parse_args()
    dev_summary_path = run_split("dev", args)
    test_summary_path = run_split("test", args)
    render_comparison(dev_summary_path, test_summary_path)


if __name__ == "__main__":
    run()
