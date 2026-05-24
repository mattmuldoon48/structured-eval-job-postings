import json
from pathlib import Path
from typing import Any

from .llm_client import LLMClient, LLMResult
from .schema import JobPostingLabel


PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "extract_v4.txt"


def _load_prompt_template(prompt_path: Path | None = None) -> str:
    return (prompt_path or PROMPT_PATH).read_text(encoding="utf-8")


def _normalize_response(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = LLMClient.extract_json(raw)
        return json.loads(cleaned)


def generate_draft_label(
    job_text: str,
    client: LLMClient | None = None,
    prompt_path: Path | None = None,
) -> JobPostingLabel:
    if client is None:
        client = LLMClient.from_env()

    prompt_template = _load_prompt_template(prompt_path)
    prompt = prompt_template.replace("{job_text}", job_text)
    messages = [
        {
            "role": "system",
            "content": "You are a structured data extraction assistant. Return only valid JSON."
        },
        {"role": "user", "content": prompt},
    ]

    response_text = client.generate(messages)
    parsed = _normalize_response(response_text)
    return JobPostingLabel.model_validate(parsed)


def generate_draft_label_with_usage(
    job_text: str,
    client: LLMClient | None = None,
    prompt_path: Path | None = None,
) -> tuple[JobPostingLabel, LLMResult]:
    if client is None:
        client = LLMClient.from_env()

    prompt_template = _load_prompt_template(prompt_path)
    prompt = prompt_template.replace("{job_text}", job_text)
    messages = [
        {
            "role": "system",
            "content": "You are a structured data extraction assistant. Return only valid JSON."
        },
        {"role": "user", "content": prompt},
    ]

    result = client.generate_with_usage(messages)
    parsed = _normalize_response(result.content)
    return JobPostingLabel.model_validate(parsed), result
