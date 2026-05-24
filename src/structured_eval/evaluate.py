import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import JobPostingLabel


EXACT_FIELDS = [
    "company",
    "title",
    "seniority",
    "employment_type",
    "location",
    "remote_policy",
    "salary_min",
    "salary_max",
    "required_years_experience",
    "security_clearance_required",
    "sponsorship_available",
]
LIST_FIELDS = ["required_skills", "nice_to_have_skills"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        normalized = re.sub(r"\s+", " ", value.strip().lower())
        return normalized or None
    return value


def normalize_list(values: list[str]) -> set[str]:
    return {normalize_scalar(value) for value in values if normalize_scalar(value)}


def list_f1(expected: list[str], actual: list[str]) -> float:
    expected_set = normalize_list(expected)
    actual_set = normalize_list(actual)
    if not expected_set and not actual_set:
        return 1.0
    if not expected_set or not actual_set:
        return 0.0
    overlap = len(expected_set & actual_set)
    precision = overlap / len(actual_set)
    recall = overlap / len(expected_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class EvalAccumulator:
    exact_totals: dict[str, int] = field(default_factory=lambda: {field: 0 for field in EXACT_FIELDS})
    exact_matches: dict[str, int] = field(default_factory=lambda: {field: 0 for field in EXACT_FIELDS})
    list_scores: dict[str, list[float]] = field(default_factory=lambda: {field: [] for field in LIST_FIELDS})

    def add(self, expected: JobPostingLabel, actual: JobPostingLabel) -> None:
        expected_data = expected.model_dump(mode="json")
        actual_data = actual.model_dump(mode="json")

        for field_name in EXACT_FIELDS:
            self.exact_totals[field_name] += 1
            if normalize_scalar(expected_data[field_name]) == normalize_scalar(actual_data[field_name]):
                self.exact_matches[field_name] += 1

        for field_name in LIST_FIELDS:
            self.list_scores[field_name].append(list_f1(expected_data[field_name], actual_data[field_name]))

    def summary(self) -> dict[str, Any]:
        exact_accuracy = {
            field_name: self.exact_matches[field_name] / self.exact_totals[field_name]
            for field_name in EXACT_FIELDS
            if self.exact_totals[field_name]
        }
        list_f1_scores = {
            field_name: sum(scores) / len(scores)
            for field_name, scores in self.list_scores.items()
            if scores
        }
        overall_scores = list(exact_accuracy.values()) + list(list_f1_scores.values())
        return {
            "examples": max(self.exact_totals.values(), default=0),
            "overall_mean_score": sum(overall_scores) / len(overall_scores) if overall_scores else 0.0,
            "exact_accuracy": exact_accuracy,
            "list_f1": list_f1_scores,
        }
