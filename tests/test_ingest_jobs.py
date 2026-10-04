import json

from scripts import ingest_jobs


def test_ingest_allocates_unique_ids_after_reordered_records(tmp_path, monkeypatch):
    raw_path = tmp_path / "jobs.jsonl"
    original = [
        {"id": job_id, "text": "Existing posting"}
        for job_id in ["job-010", "job-002", "job-009"]
    ]
    raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in original), encoding="utf-8"
    )
    monkeypatch.setattr(ingest_jobs, "RAW_PATH", raw_path)
    responses = iter([
        "First new posting", "###END###", "y",
        "Second new posting", "###END###", "n",
    ])
    monkeypatch.setattr("builtins.input", lambda *_: next(responses))

    ingest_jobs.run()

    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    assert records[:3] == original
    assert records[3:] == [
        {"id": "job-011", "text": "First new posting"},
        {"id": "job-012", "text": "Second new posting"},
    ]
