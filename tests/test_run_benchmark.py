import os
import subprocess
import sys
import time
from pathlib import Path

from scripts import run_benchmark, run_eval


def test_benchmark_keeps_its_own_summary_when_another_run_finishes(tmp_path, monkeypatch):
    reports = tmp_path / "runs"
    predictions = Path(__file__).parent / "fixtures" / "replay_predictions.jsonl"
    monkeypatch.setattr(run_eval, "REPORTS_DIR", reports)
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--limit-per-split", "1"])
    args = run_benchmark.parse_args()
    outputs = {}

    def run_child_with_interference(command, **kwargs):
        # Exercise the real CLI entry point with offline predictions instead of an API call.
        with monkeypatch.context() as child:
            child.setattr(
                sys, "argv",
                [*command[1:], "--replay-predictions", str(predictions)],
            )
            run_eval.run()
            owned_summary, = reports.glob("*/summary.json")
            outputs["owned"] = owned_summary
            outputs["original"] = owned_summary.read_bytes()

            # Another evaluation with the same split/settings finishes before discovery.
            child.setattr(
                sys, "argv",
                ["run_eval.py", "--replay-predictions", str(predictions), "--split", "dev", "--limit", "1"],
            )
            run_eval.run()
            other_summary, = set(reports.glob("*/summary.json")) - {owned_summary}
            timestamp = time.time() + 1
            os.utime(owned_summary, (timestamp, timestamp))
            os.utime(other_summary, (timestamp + 1, timestamp + 1))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_benchmark.subprocess, "run", run_child_with_interference)
    selected = run_benchmark.run_split("dev", args)

    assert selected == outputs["owned"]
    assert selected.read_bytes() == outputs["original"]
