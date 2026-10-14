import json
import sys
from datetime import datetime
from pathlib import Path

from scripts import run_eval


class FrozenClock:
    @staticmethod
    def now():
        return datetime(2026, 10, 14, 12, 0, 0)


def test_same_second_replays_preserve_separate_run_artifacts(tmp_path, monkeypatch):
    reports = tmp_path / "runs"
    predictions = Path(__file__).parent / "fixtures" / "replay_predictions.jsonl"
    monkeypatch.setattr(run_eval, "REPORTS_DIR", reports)
    monkeypatch.setattr(run_eval, "datetime", FrozenClock)

    for limit in (1, 2):
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_eval.py", "--replay-predictions", str(predictions), "--limit", str(limit)],
        )
        run_eval.run()
        if limit == 1:
            first_dir = next(reports.iterdir())
            original_files = {path.name: path.read_bytes() for path in first_dir.iterdir()}

    assert {path.name: path.read_bytes() for path in first_dir.iterdir()} == original_files
    second_dir, = set(reports.iterdir()) - {first_dir}
    assert json.loads(original_files["summary.json"])["examples"] == 1
    assert json.loads((second_dir / "summary.json").read_text(encoding="utf-8"))["examples"] == 2
