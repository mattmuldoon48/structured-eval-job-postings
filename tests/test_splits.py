from pathlib import Path

import pytest

from structured_eval.splits import assign_split, build_split_records, load_split_map, normalized_ratios


def test_assign_split_is_deterministic():
    assert assign_split("job-001") == assign_split("job-001")


def test_normalized_ratios_sum_to_one():
    ratios = normalized_ratios({"dev": 7, "test": 3})

    assert ratios["dev"] == 0.7
    assert ratios["test"] == 0.3


@pytest.mark.parametrize("invalid_ratio", [-0.1, float("inf"), float("nan")])
def test_normalized_ratios_reject_invalid_values(invalid_ratio):
    with pytest.raises(ValueError, match="Split ratios must be finite and non-negative"):
        normalized_ratios({"dev": 1.0, "test": invalid_ratio})


def test_build_split_records_includes_ids_and_splits():
    records = build_split_records([{"id": "job-001"}, {"id": "job-002"}])

    assert records[0]["id"] == "job-001"
    assert records[0]["split"] in {"dev", "test"}


def test_load_split_map(tmp_path: Path):
    path = tmp_path / "splits.jsonl"
    path.write_text('{"id": "job-001", "split": "dev"}\n', encoding="utf-8")

    assert load_split_map(path) == {"job-001": "dev"}


def test_load_split_map_rejects_duplicate_ids(tmp_path: Path):
    path = tmp_path / "splits.jsonl"
    path.write_text(
        '{"id": "job-001", "split": "dev"}\n'
        '{"id": "job-001", "split": "test"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Duplicate split ID 'job-001' on line 2; first seen on line 1",
    ):
        load_split_map(path)
