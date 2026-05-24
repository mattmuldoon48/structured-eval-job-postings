from pathlib import Path

from structured_eval.splits import assign_split, build_split_records, load_split_map, normalized_ratios


def test_assign_split_is_deterministic():
    assert assign_split("job-001") == assign_split("job-001")


def test_normalized_ratios_sum_to_one():
    ratios = normalized_ratios({"dev": 7, "test": 3})

    assert ratios["dev"] == 0.7
    assert ratios["test"] == 0.3


def test_build_split_records_includes_ids_and_splits():
    records = build_split_records([{"id": "job-001"}, {"id": "job-002"}])

    assert records[0]["id"] == "job-001"
    assert records[0]["split"] in {"dev", "test"}


def test_load_split_map(tmp_path: Path):
    path = tmp_path / "splits.jsonl"
    path.write_text('{"id": "job-001", "split": "dev"}\n', encoding="utf-8")

    assert load_split_map(path) == {"job-001": "dev"}
