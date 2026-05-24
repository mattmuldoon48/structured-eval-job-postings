import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_SPLIT_RATIOS = {"dev": 0.7, "test": 0.3}


def normalized_ratios(ratios: dict[str, float]) -> dict[str, float]:
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
    with path.open("r", encoding="utf-8") as stream:
        return {
            record["id"]: record["split"]
            for record in (json.loads(line) for line in stream if line.strip())
        }
