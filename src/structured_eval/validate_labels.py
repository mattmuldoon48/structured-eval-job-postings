import json
from pathlib import Path

from .schema import JobPostingLabel


def validate_label_file(path: Path) -> list[tuple[int, str]]:
    errors: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as stream:
        for index, raw in enumerate(stream, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                example = json.loads(raw)
                record_id = example.get("id")
                if not isinstance(record_id, str) or not record_id.strip():
                    raise ValueError("record id must be a non-empty, non-whitespace string")
                label = {key: value for key, value in example.items() if key != "id"}
                JobPostingLabel.model_validate(label)
            except Exception as exc:
                errors.append((index, str(exc)))
    return errors


def load_labeled_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for raw in stream:
            raw = raw.strip()
            if not raw:
                continue
            records.append(json.loads(raw))
    return records
