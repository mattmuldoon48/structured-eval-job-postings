from structured_eval.dataset_checks import duplicate_ids, validate_dataset_integrity


def test_duplicate_ids_returns_repeated_ids():
    assert duplicate_ids([{"id": "job-001"}, {"id": "job-001"}, {"id": "job-002"}]) == ["job-001"]


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
