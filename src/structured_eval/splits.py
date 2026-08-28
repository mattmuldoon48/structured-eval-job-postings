import math
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_SPLIT_RATIOS = {"dev": 0.7, "test": 0.3}


def normalized_ratios(ratios: dict[str, float]) -> dict[str, float]:
    if any(not math.isfinite(value) or value < 0 for value in ratios.values()):
        raise ValueError("Split ratios must be finite and non-negative")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value")
    return {name: value / total for name, value in ratios.items()}


def assign_split(record_id: str, ratios: dict[str, float] | None = None, seed: str = "job-posting-eval") -> str:
    ratios = normalized_ratios(ratios or DEFAULT_SPLIT_RATIOS)
    digest = hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:12], 16) / float(0xFFFFFFFFFFFF)

    cumulative = 0.0
    last_name = next(reversed(ratios))
    for name, ratio in ratios.items():
        cumulative += ratio
        if bucket <= cumulative:
            return name
    return last_name


def build_split_records(records: list[dict[str, Any]], seed: str = "job-posting-eval") -> list[dict[str, str]]:
    return [{"id": record["id"], "split": assign_split(record["id"], seed=seed)} for record in records]


def load_split_map(path: Path) -> dict[str, str]:
    split_map: dict[str, str] = {}
    first_seen_lines: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = record["id"]
            if record_id in first_seen_lines:
                raise ValueError(
                    f"Duplicate split ID {record_id!r} on line {line_number}; "
                    f"first seen on line {first_seen_lines[record_id]}"
                )
            first_seen_lines[record_id] = line_number
            split_map[record_id] = record["split"]
    return split_map
