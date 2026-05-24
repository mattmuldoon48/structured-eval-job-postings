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
NORMALIZED_FIELDS = ["company", "title", "location"]
LIST_FIELDS = ["required_skills", "nice_to_have_skills"]
MISMATCH_EXACT_FIELDS = [
    "seniority",
    "employment_type",
    "remote_policy",
    "salary_min",
    "salary_max",
    "required_years_experience",
    "security_clearance_required",
    "sponsorship_available",
]
STOPWORDS = {
    "a",
    "an",
    "and",
    "architectures",
    "based",
    "development",
    "engineering",
    "experience",
    "framework",
    "frameworks",
    "in",
    "model",
    "models",
    "of",
    "or",
    "platform",
    "platforms",
    "system",
    "systems",
    "the",
    "to",
    "tools",
    "using",
    "with",
}
SKILL_SYNONYMS = {
    "genai": "generative ai",
    "generative": "generative ai",
    "llm": "llms",
    "rag": "retrieval augmented generation",
    "restful": "rest",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        normalized = re.sub(r"\s+", " ", value.strip().lower())
        return normalized or None
    return value


def normalized_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def token_set(value: str) -> set[str]:
    normalized = normalized_text(value)
    if normalized is None:
        return set()
    tokens = set()
    for token in normalized.split():
        token = SKILL_SYNONYMS.get(token, token)
        if token not in STOPWORDS:
            tokens.add(token)
    return tokens


def token_overlap_score(expected: Any, actual: Any) -> float:
    if expected is None and actual is None:
        return 1.0
    if expected is None or actual is None:
        return 0.0
    expected_tokens = token_set(str(expected))
    actual_tokens = token_set(str(actual))
    if not expected_tokens and not actual_tokens:
        return 1.0
    if not expected_tokens or not actual_tokens:
        return 0.0
    overlap = len(expected_tokens & actual_tokens)
    precision = overlap / len(actual_tokens)
    recall = overlap / len(expected_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def normalize_list(values: list[str]) -> set[str]:
    return {normalize_scalar(value) for value in values if normalize_scalar(value)}


def exact_list_f1(expected: list[str], actual: list[str]) -> float:
    expected_items = normalize_list(expected)
    actual_items = normalize_list(actual)
    if not expected_items and not actual_items:
        return 1.0
    if not expected_items or not actual_items:
        return 0.0
    overlap = len(expected_items & actual_items)
    precision = overlap / len(actual_items)
    recall = overlap / len(expected_items)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def soft_list_f1(expected: list[str], actual: list[str]) -> float:
    expected_items = [item for item in expected if normalized_text(item)]
    actual_items = [item for item in actual if normalized_text(item)]
    if not expected_items and not actual_items:
        return 1.0
    if not expected_items or not actual_items:
        return 0.0

    precision = sum(max(token_overlap_score(expected, actual) for expected in expected_items) for actual in actual_items)
    precision /= len(actual_items)
    recall = sum(max(token_overlap_score(expected, actual) for actual in actual_items) for expected in expected_items)
    recall /= len(expected_items)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def example_mismatches(expected: JobPostingLabel, actual: JobPostingLabel) -> list[dict[str, Any]]:
    expected_data = expected.model_dump(mode="json")
    actual_data = actual.model_dump(mode="json")
    mismatches = []

    for field_name in MISMATCH_EXACT_FIELDS:
        if normalize_scalar(expected_data[field_name]) != normalize_scalar(actual_data[field_name]):
            mismatches.append(
                {
                    "field": field_name,
                    "metric": "exact",
                    "score": 0.0,
                    "expected": expected_data[field_name],
                    "actual": actual_data[field_name],
                }
            )

    for field_name in NORMALIZED_FIELDS:
        score = token_overlap_score(expected_data[field_name], actual_data[field_name])
        if score < 1.0:
            mismatches.append(
                {
                    "field": field_name,
                    "metric": "normalized_text_score",
                    "score": score,
                    "expected": expected_data[field_name],
                    "actual": actual_data[field_name],
                }
            )

    for field_name in LIST_FIELDS:
        score = soft_list_f1(expected_data[field_name], actual_data[field_name])
        if score < 0.8:
            mismatches.append(
                {
                    "field": field_name,
                    "metric": "soft_list_f1",
                    "score": score,
                    "expected": expected_data[field_name],
                    "actual": actual_data[field_name],
                }
            )

    return mismatches


def parse_metric_gate(raw: str) -> tuple[str, str, float]:
    try:
        metric_path, threshold_text = raw.split("=", maxsplit=1)
        metric_group, field_name = metric_path.split(".", maxsplit=1)
        threshold = float(threshold_text)
    except ValueError as exc:
        raise ValueError(
            "Metric gates must use the form metric_group.field=value, "
            "for example exact_accuracy.remote_policy=0.80"
        ) from exc
    if not 0 <= threshold <= 1:
        raise ValueError("Metric gate threshold must be between 0 and 1")
    return metric_group, field_name, threshold


def evaluate_quality_gates(
    summary: dict[str, Any],
    min_overall: float | None = None,
    metric_gates: list[str] | None = None,
) -> list[str]:
    failures = []

    if min_overall is not None and summary["overall_mean_score"] < min_overall:
        failures.append(
            f"overall_mean_score {summary['overall_mean_score']:.3f} is below required {min_overall:.3f}"
        )

    for raw_gate in metric_gates or []:
        metric_group, field_name, threshold = parse_metric_gate(raw_gate)
        try:
            score = summary[metric_group][field_name]
        except KeyError as exc:
            raise ValueError(f"Unknown metric gate target: {metric_group}.{field_name}") from exc
        if score < threshold:
            failures.append(
                f"{metric_group}.{field_name} {score:.3f} is below required {threshold:.3f}"
            )

    return failures


@dataclass
class EvalAccumulator:
    exact_totals: dict[str, int] = field(default_factory=lambda: {field: 0 for field in EXACT_FIELDS})
    exact_matches: dict[str, int] = field(default_factory=lambda: {field: 0 for field in EXACT_FIELDS})
    normalized_scores: dict[str, list[float]] = field(default_factory=lambda: {field: [] for field in NORMALIZED_FIELDS})
    exact_list_scores: dict[str, list[float]] = field(default_factory=lambda: {field: [] for field in LIST_FIELDS})
    soft_list_scores: dict[str, list[float]] = field(default_factory=lambda: {field: [] for field in LIST_FIELDS})

    def add(self, expected: JobPostingLabel, actual: JobPostingLabel) -> None:
        expected_data = expected.model_dump(mode="json")
        actual_data = actual.model_dump(mode="json")

        for field_name in EXACT_FIELDS:
            self.exact_totals[field_name] += 1
            if normalize_scalar(expected_data[field_name]) == normalize_scalar(actual_data[field_name]):
                self.exact_matches[field_name] += 1

        for field_name in NORMALIZED_FIELDS:
            self.normalized_scores[field_name].append(
                token_overlap_score(expected_data[field_name], actual_data[field_name])
            )

        for field_name in LIST_FIELDS:
            self.exact_list_scores[field_name].append(
                exact_list_f1(expected_data[field_name], actual_data[field_name])
            )
            self.soft_list_scores[field_name].append(
                soft_list_f1(expected_data[field_name], actual_data[field_name])
            )

    def summary(self) -> dict[str, Any]:
        exact_accuracy = {
            field_name: self.exact_matches[field_name] / self.exact_totals[field_name]
            for field_name in EXACT_FIELDS
            if self.exact_totals[field_name]
        }
        normalized_scores = {
            field_name: sum(scores) / len(scores)
            for field_name, scores in self.normalized_scores.items()
            if scores
        }
        exact_list_f1_scores = {
            field_name: sum(scores) / len(scores)
            for field_name, scores in self.exact_list_scores.items()
            if scores
        }
        soft_list_f1_scores = {
            field_name: sum(scores) / len(scores)
            for field_name, scores in self.soft_list_scores.items()
            if scores
        }
        overall_scores = (
            [score for field_name, score in exact_accuracy.items() if field_name not in NORMALIZED_FIELDS]
            + list(normalized_scores.values())
            + list(soft_list_f1_scores.values())
        )
        return {
            "examples": max(self.exact_totals.values(), default=0),
            "overall_mean_score": sum(overall_scores) / len(overall_scores) if overall_scores else 0.0,
            "exact_accuracy": exact_accuracy,
            "normalized_text_score": normalized_scores,
            "exact_list_f1": exact_list_f1_scores,
            "soft_list_f1": soft_list_f1_scores,
        }
