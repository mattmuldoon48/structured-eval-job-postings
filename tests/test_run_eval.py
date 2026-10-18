import json
import sys
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts import run_eval
from structured_eval.evaluate import score_prediction_records


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


@pytest.mark.parametrize(
    "extra_fields",
    [{}, {"employment_typ": "full_time"}],
    ids=["record-metadata", "unknown-label-field"],
)
def test_saved_labels_reach_scoring_without_relaxing_schema(
    tmp_path, monkeypatch, extra_fields
):
    raw_path = tmp_path / "raw.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    raw_path.write_text(
        json.dumps({"id": "job-test", "text": "Acme hybrid role"}) + "\n",
        encoding="utf-8",
    )
    labels_path.write_text(
        json.dumps({
            "id": "job-test", "company": "Acme", "remote_policy": "hybrid", **extra_fields
        }) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(run_eval, "RAW_PATH", raw_path)
    monkeypatch.setattr(run_eval, "LABELED_PATH", labels_path)
    record, = run_eval.build_joined_records()
    predictions = [{**record, "actual": {"company": "Acme", "remote_policy": "remote"}}]

    if extra_fields:
        with pytest.raises(ValidationError) as error:
            score_prediction_records(predictions)
        assert [(item["loc"], item["type"]) for item in error.value.errors()] == [
            (("employment_typ",), "extra_forbidden")
        ]
    else:
        summary, mismatches, _ = score_prediction_records(predictions)
        assert summary["examples"] == 1
        assert summary["exact_accuracy"]["company"] == 1.0
        assert summary["exact_accuracy"]["remote_policy"] == 0.0
        assert [
            (item["id"], item["field"], item["expected"], item["actual"])
            for item in mismatches
        ] == [("job-test", "remote_policy", "hybrid", "remote")]
