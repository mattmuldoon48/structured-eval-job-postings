import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMResult:
    content: str
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def usage_dict(self) -> dict[str, Any]:
        return {
            "latency_seconds": self.latency_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMClient:
    api_key: str
    model: str
    provider: str = "openai"

    @classmethod
    def from_env(cls) -> "LLMClient":
        api_key = os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required in environment")
        return cls(api_key=api_key, model=model)

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        return self.generate_with_usage(messages, temperature=temperature).content

    def generate_with_usage(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> LLMResult:
        if self.provider != "openai":
            raise NotImplementedError("Only OpenAI provider is implemented")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package is required for LLM calls") from exc

        client = OpenAI(api_key=self.api_key)
        started_at = time.perf_counter()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=1000,
        )
        latency_seconds = time.perf_counter() - started_at
        usage = getattr(response, "usage", None)
        return LLMResult(
            content=response.choices[0].message.content,
            latency_seconds=latency_seconds,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )

    @staticmethod
    def extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`\n")
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start > end:
            raise ValueError("Could not find JSON object in model response")
        return text[start : end + 1]

    def generate_json(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> Any:
        raw = self.generate(messages, temperature=temperature)
        content = self.extract_json(raw)
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response was not valid JSON") from exc
