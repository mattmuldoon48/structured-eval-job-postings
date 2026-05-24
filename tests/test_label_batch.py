from scripts.label_batch import append_label
from structured_eval.schema import JobPostingLabel


def test_append_label_writes_job_id_and_label_fields(tmp_path, monkeypatch):
    labeled_path = tmp_path / "labels.jsonl"
    monkeypatch.setattr("scripts.label_batch.LABELED_PATH", labeled_path)
    label = JobPostingLabel(company="Acme", title="AI Engineer")

    append_label("job-001", label.model_dump(mode="json"))

    text = labeled_path.read_text(encoding="utf-8")
    assert '"id": "job-001"' in text
    assert '"company": "Acme"' in text
    assert "needs human review" in text
