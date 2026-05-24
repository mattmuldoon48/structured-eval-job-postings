from structured_eval.llm_client import LLMResult


def test_llm_result_usage_dict_contains_usage_fields():
    result = LLMResult(
        content="{}",
        latency_seconds=1.25,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    assert result.usage_dict() == {
        "latency_seconds": 1.25,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
