import pytest

from structured_eval.llm_client import LLMClient, LLMResult


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


def test_extract_json_returns_object_from_prose_response():
    raw = 'Here is the label: {"company": "Acme", "remote_policy": "remote"} Thanks.'

    assert LLMClient.extract_json(raw) == '{"company": "Acme", "remote_policy": "remote"}'


def test_extract_json_returns_object_from_fenced_response():
    raw = '```json\n{"company": "Acme", "required_skills": ["python"]}\n```'

    assert LLMClient.extract_json(raw) == '{"company": "Acme", "required_skills": ["python"]}'


def test_extract_json_returns_first_of_adjacent_objects():
    raw = '{"company": "First"}{"company": "Second"}'

    assert LLMClient.extract_json(raw) == '{"company": "First"}'


def test_extract_json_skips_invalid_brace_prose_before_valid_object():
    raw = 'Use {company name} as context, then return {"company": "Acme"}.'

    assert LLMClient.extract_json(raw) == '{"company": "Acme"}'


def test_extract_json_handles_nested_and_string_braces():
    raw = 'Result: {"metadata": {"note": "keep {these} braces"}, "valid": true} trailing'

    assert LLMClient.extract_json(raw) == (
        '{"metadata": {"note": "keep {these} braces"}, "valid": true}'
    )


def test_extract_json_raises_when_response_has_no_object():
    with pytest.raises(ValueError, match="Could not find JSON object"):
        LLMClient.extract_json("I could not extract a structured job label.")