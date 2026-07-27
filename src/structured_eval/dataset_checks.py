from collections import Counter
from typing import Any, TypeGuard


def _is_valid_record_id(record_id: Any) -> TypeGuard[str]:
    return isinstance(record_id, str) and bool(record_id.strip())


def _valid_ids(records: list[dict[str, Any]]) -> set[str]:
    return {
        record_id
        for record in records
        if _is_valid_record_id(record_id := record.get("id"))
    }


def duplicate_ids(records: list[dict[str, Any]]) -> list[str]:
    counts = Counter(
        record_id
        for record in records
        if _is_valid_record_id(record_id := record.get("id"))
    )
    return sorted(record_id for record_id, count in counts.items() if count > 1)


def validate_dataset_integrity(
    raw_records: list[dict[str, Any]],
    labeled_records: list[dict[str, Any]],
    split_records: list[dict[str, Any]],
) -> list[str]:
    failures = []
    raw_ids = _valid_ids(raw_records)
    labeled_ids = _valid_ids(labeled_records)
    split_ids = _valid_ids(split_records)

    for name, records in [
        ("raw", raw_records),
        ("labeled", labeled_records),
        ("split", split_records),
    ]:
        invalid_positions = [
            str(position)
            for position, record in enumerate(records, start=1)
            if not _is_valid_record_id(record.get("id"))
        ]
        if invalid_positions:
            location = (
                f"position {invalid_positions[0]}"
                if len(invalid_positions) == 1
                else f"positions {', '.join(invalid_positions)}"
            )
            failures.append(
                f"{name} records contain invalid ids at {location}; "
                "ids must be non-empty, non-whitespace strings"
            )
        duplicates = duplicate_ids(records)
        if duplicates:
            failures.append(f"{name} records contain duplicate ids: {', '.join(duplicates)}")

    missing_labels = sorted(raw_ids - labeled_ids)
    if missing_labels:
        failures.append(f"raw records missing labels: {', '.join(missing_labels)}")

    orphan_labels = sorted(labeled_ids - raw_ids)
    if orphan_labels:
        failures.append(f"labels without raw records: {', '.join(orphan_labels)}")

    missing_splits = sorted(raw_ids - split_ids)
    if missing_splits:
        failures.append(f"raw records missing split assignments: {', '.join(missing_splits)}")

    orphan_splits = sorted(split_ids - raw_ids)
    if orphan_splits:
        failures.append(f"split assignments without raw records: {', '.join(orphan_splits)}")

    draft_labels = sorted(
        record_id
        for record in labeled_records
        if _is_valid_record_id(record_id := record.get("id"))
        and "needs human review" in str(record.get("labeling_notes") or "").lower()
    )
    if draft_labels:
        failures.append(f"labels still marked as draft: {', '.join(draft_labels)}")

    invalid_splits = sorted(
        f"{record.get('id')}={record.get('split')}"
        for record in split_records
        if record.get("split") not in {"dev", "test"}
    )
    if invalid_splits:
        failures.append(f"invalid split values: {', '.join(invalid_splits)}")

    return failures
