import pytest

from structured_eval.evaluate import (
    compare_summaries,
    estimate_cost,
    evaluate_quality_gates,
    example_mismatches,
    flatten_summary_metrics,
    parse_metric_gate,
    score_prediction_records,
    soft_list_f1,
    summarize_usage,
    token_overlap_score,
)
from structured_eval.schema import JobPostingLabel


def test_token_overlap_scores_partial_location_match():
    assert token_overlap_score("Seattle, WA", "Seattle") > 0.6


def test_soft_list_f1_scores_related_skill_phrases():
    score = soft_list_f1(
        ["model optimization", "LLMs", "AI integration"],
        ["model performance optimization", "LLM integrations"],
    )

    assert score > 0.5


def test_example_mismatches_includes_partial_text_and_skill_misses():
    expected = JobPostingLabel(
        company="Acme Corp",
        title="Senior AI Engineer",
        location="Seattle, WA",
        required_skills=["RAG pipelines", "Python"],
    )
    actual = JobPostingLabel(
        company="Acme",
        title="Senior AI Engineer",
        location="Seattle",
        required_skills=["Java"],
    )

    mismatches = example_mismatches(expected, actual)
    fields = {mismatch["field"] for mismatch in mismatches}

    assert "company" in fields
    assert "location" in fields
    assert "required_skills" in fields


def test_parse_metric_gate_returns_metric_group_field_and_threshold():
    assert parse_metric_gate("exact_accuracy.remote_policy=0.80") == (
        "exact_accuracy",
        "remote_policy",
        0.8,
    )


def test_parse_metric_gate_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        parse_metric_gate("exact_accuracy.remote_policy=1.2")


def test_evaluate_quality_gates_reports_failures():
    summary = {
        "overall_mean_score": 0.84,
        "exact_accuracy": {"remote_policy": 0.79},
    }

    failures = evaluate_quality_gates(
        summary,
        min_overall=0.85,
        metric_gates=["exact_accuracy.remote_policy=0.80"],
    )

    assert len(failures) == 2


def test_evaluate_quality_gates_passes_when_thresholds_met():
    summary = {
        "overall_mean_score": 0.86,
        "exact_accuracy": {"remote_policy": 0.82},
    }

    failures = evaluate_quality_gates(
        summary,
        min_overall=0.85,
        metric_gates=["exact_accuracy.remote_policy=0.80"],
    )

    assert failures == []


def test_summarize_usage_aggregates_latency_and_tokens():
    summary = summarize_usage(
        [
            {
                "latency_seconds": 1.5,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
            {
                "latency_seconds": 2.5,
                "prompt_tokens": 200,
                "completion_tokens": 30,
                "total_tokens": 230,
            },
        ]
    )

    assert summary["examples"] == 2
    assert summary["total_latency_seconds"] == 4.0
    assert summary["average_latency_seconds"] == 2.0
    assert summary["prompt_tokens"] == 300
    assert summary["completion_tokens"] == 50
    assert summary["total_tokens"] == 350


def test_estimate_cost_returns_none_without_rates():
    assert estimate_cost({"prompt_tokens": 100, "completion_tokens": 50}) is None


def test_estimate_cost_uses_input_and_output_rates():
    estimate = estimate_cost(
        {"prompt_tokens": 1_000_000, "completion_tokens": 500_000},
        input_cost_per_1m=0.4,
        output_cost_per_1m=1.6,
    )

    assert estimate["input_cost_usd"] == 0.4
    assert estimate["output_cost_usd"] == 0.8
    assert estimate["total_cost_usd"] == pytest.approx(1.2)


def test_score_prediction_records_scores_replay_records():
    records = [
        {
            "id": "job-001",
            "expected": JobPostingLabel(
                company="Acme",
                title="AI Engineer",
                required_skills=["Python"],
            ).model_dump(mode="json"),
            "actual": JobPostingLabel(
                company="Acme",
                title="AI Engineer",
                required_skills=["Python"],
            ).model_dump(mode="json"),
            "usage": {
                "latency_seconds": 1.0,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
    ]

    summary, mismatches, usage_records = score_prediction_records(records)

    assert summary["examples"] == 1
    assert summary["overall_mean_score"] == 1.0
    assert mismatches == []
    assert usage_records == [records[0]["usage"]]


def test_flatten_summary_metrics_includes_nested_scores_and_usage():
    metrics = flatten_summary_metrics(
        {
            "overall_mean_score": 0.8,
            "exact_accuracy": {"remote_policy": 0.9},
            "soft_list_f1": {"required_skills": 0.5},
            "usage": {"total_tokens": 100, "average_latency_seconds": 1.25},
            "cost_estimate": {"total_cost_usd": 0.001},
        }
    )

    assert metrics["overall_mean_score"] == 0.8
    assert metrics["exact_accuracy.remote_policy"] == 0.9
    assert metrics["soft_list_f1.required_skills"] == 0.5
    assert metrics["usage.total_tokens"] == 100
    assert metrics["usage.average_latency_seconds"] == 1.25
    assert metrics["cost.total_cost_usd"] == 0.001


def test_compare_summaries_returns_candidate_minus_baseline_delta():
    comparison = compare_summaries(
        {"overall_mean_score": 0.8, "exact_accuracy": {"remote_policy": 0.7}},
        {"overall_mean_score": 0.9, "exact_accuracy": {"remote_policy": 0.6}},
    )
    by_metric = {row["metric"]: row for row in comparison}

    assert by_metric["overall_mean_score"]["delta"] == pytest.approx(0.1)
    assert by_metric["exact_accuracy.remote_policy"]["delta"] == pytest.approx(-0.1)
