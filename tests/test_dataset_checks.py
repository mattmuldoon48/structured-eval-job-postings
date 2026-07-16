import pytest

from structured_eval.dataset_checks import duplicate_ids, validate_dataset_integrity


def test_duplicate_ids_returns_repeated_ids():
    assert duplicate_ids([{"id": "job-001"}, {"id": "job-001"}, {"id": "job-002"}]) == ["job-001"]


def test_duplicate_ids_ignores_mixed_invalid_ids():
    records = [
        {"id": "job-002"},
        {},
        {"id": None},
        {"id": ""},
        {"id": "  "},
        {"id": 2},
        {"id": "job-002"},
    ]

    assert duplicate_ids(records) == ["job-002"]


def test_validate_dataset_integrity_passes_matching_records():
    failures = validate_dataset_integrity(
        raw_records=[{"id": "job-001"}],
        labeled_records=[{"id": "job-001", "labeling_notes": None}],
        split_records=[{"id": "job-001", "split": "dev"}],
    )

    assert failures == []


def test_validate_dataset_integrity_reports_mismatches_and_drafts():
    failures = validate_dataset_integrity(
        raw_records=[{"id": "job-001"}, {"id": "job-002"}],
        labeled_records=[
            {"id": "job-001", "labeling_notes": "needs human review"},
            {"id": "job-003", "labeling_notes": None},
        ],
        split_records=[{"id": "job-001", "split": "holdout"}],
    )

    assert any("missing labels" in failure for failure in failures)
    assert any("without raw records" in failure for failure in failures)
    assert any("missing split assignments" in failure for failure in failures)
    assert any("draft" in failure for failure in failures)
    assert any("invalid split values" in failure for failure in failures)


@pytest.mark.parametrize(
    ("collection_name", "collection_argument", "record_defaults"),
    [
        ("raw", "raw_records", {}),
        ("labeled", "labeled_records", {"labeling_notes": None}),
        ("split", "split_records", {"split": "dev"}),
    ],
)
@pytest.mark.parametrize(
    "invalid_record",
    [{}, {"id": None}, {"id": ""}, {"id": " \t"}],
    ids=["missing", "null", "empty", "whitespace"],
)
def test_validate_dataset_integrity_reports_invalid_id_positions(
    collection_name, collection_argument, record_defaults, invalid_record
):
    collections = {
        "raw_records": [{"id": "job-001"}],
        "labeled_records": [{"id": "job-001", "labeling_notes": None}],
        "split_records": [{"id": "job-001", "split": "dev"}],
    }
    collections[collection_argument].append({**record_defaults, **invalid_record})

    failures = validate_dataset_integrity(**collections)

    assert any(
        collection_name in failure
        and "invalid ids" in failure
        and "position 2" in failure
        for failure in failures
    )
    assert not any("missing labels" in failure for failure in failures)
    assert not any("without raw records" in failure for failure in failures)
    assert not any("missing split assignments" in failure for failure in failures)


def test_validate_dataset_integrity_reports_aligned_invalid_ids():
    failures = validate_dataset_integrity(
        raw_records=[{"id": None}],
        labeled_records=[{"id": None, "labeling_notes": None}],
        split_records=[{"id": None, "split": "dev"}],
    )

    for collection_name in ("raw", "labeled", "split"):
        assert any(
            collection_name in failure
            and "invalid ids" in failure
            and "position 1" in failure
            for failure in failures
        )
    assert not any("missing" in failure for failure in failures)
    assert not any("without raw records" in failure for failure in failures)
