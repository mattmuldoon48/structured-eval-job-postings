"""Structured evaluation package for job posting label assistance."""

from .schema import JobPostingLabel
from .label_assist import generate_draft_label
from .llm_client import LLMClient
