# Portfolio Blurb

Use or adapt this text for a resume, portfolio page, LinkedIn, or GitHub project description.

## GitHub Description

Python evaluation harness for LLM structured extraction from job postings, with human-reviewed labels, Pydantic validation, prompt versioning, dev/test benchmarks, replay scoring, and CI quality gates.

## Resume Bullets

- Built a production-style LLM evaluation harness in Python for structured job-posting extraction, covering schema design, labeling workflow, prompt iteration, and benchmark reporting.
- Created a 64-example reviewed JSONL dataset for AI/ML job postings and evaluated extraction quality across fields including seniority, salary, remote policy, sponsorship, and skill lists.
- Implemented Pydantic validation, deterministic dev/test splits, exact/normalized/F1 metrics, replay-mode scoring, mismatch analysis, and CI checks for data integrity and regression safety.
- Improved prompt performance through iterative evaluation, reaching `0.920` overall score and `0.906` held-out test score with `gpt-4.1-mini`.

## Case Study Summary

This project explores what it takes to move beyond a simple "LLM returns JSON" demo and build a repeatable evaluation loop for structured extraction.

I built a Python harness that uses a Pydantic schema to extract structured fields from real AI/ML job postings, including company, title, seniority, employment type, location, remote policy, compensation, required experience, skills, clearance, and sponsorship. The workflow supports LLM-generated draft labels, human review, JSONL ground truth, deterministic dev/test splits, live model evaluation, replay scoring, and field-level mismatch analysis.

The benchmark currently uses 64 reviewed labels and scores `gpt-4.1-mini` with the `extract_v4.txt` prompt at `0.920` overall, with a `0.906` held-out test score. Prompt iteration improved seniority and skill-list extraction, while the current error profile points to remote/location policy as the next highest-value target.

The repo includes CI checks for unit tests, label validation, dataset integrity, and replay-mode eval gates so the project can catch schema, data, and scoring regressions without requiring live API calls in CI.

## Interview Talking Points

- The project is intentionally small and inspectable: JSONL, Pydantic, direct OpenAI SDK calls, and focused scripts.
- The evaluation treats different field types differently: exact match for enums/numbers/booleans, normalized text overlap for fuzzy fields, and exact plus soft F1 for skill lists.
- Replay mode separates scoring logic from model calls, which makes CI cheaper and more reliable.
- The dev/test split prevents prompt iteration from becoming purely anecdotal.
- The biggest remaining challenge is policy ambiguity around remote/hybrid/onsite roles and location formatting.
