# Project Summary

`structured-eval-job-postings` is a Python evaluation harness for measuring how well an LLM extracts structured fields from real AI/ML job postings.

## Problem

LLM demos often stop at "the model returned JSON." In production, that is not enough. Structured extraction systems need:

- a trusted schema
- reviewed ground truth
- repeatable evaluation
- field-level metrics
- prompt/version comparison
- regression checks that do not require live model calls

This project implements that loop for job-posting extraction.

## What It Does

The system extracts fields such as:

- company, title, seniority, employment type
- location and remote policy
- salary range
- required years of experience
- required and preferred skills
- clearance and sponsorship signals

It supports both human-in-the-loop labeling and benchmark evaluation:

```text
raw postings -> draft labels -> human review -> validated labels -> prompt eval -> benchmark report
```

## Technical Highlights

- Pydantic schema with enum and value validation
- JSONL dataset workflow for raw, labeled, and split records
- OpenAI SDK integration without a large orchestration framework
- Prompt versioning from `extract_v1.txt` through `extract_v4.txt`
- Deterministic dev/test split
- Strict exact metrics for enum, boolean, numeric, and salary fields
- Normalized text metrics for company/title/location
- Exact and soft F1 metrics for skill lists
- Replay mode for scoring saved predictions without model calls
- GitHub Actions CI with unit tests, label validation, and replay eval quality gate

## Current Dataset

- 64 reviewed labels
- 43 dev examples
- 21 test examples
- Roles include AI/ML engineering, agentic AI, MLOps, AI platform, full-stack AI, robotics, healthcare AI, fintech AI, and government/clearance roles

## Current Result

Using `gpt-4.1-mini` and `extract_v4.txt`:

| Evaluation | Score |
| --- | ---: |
| Full dataset overall | 0.920 |
| Dev overall | 0.927 |
| Test overall | 0.906 |
| Test gap | -0.021 |

Selected field scores:

| Field / metric | Score |
| --- | ---: |
| Company exact accuracy | 0.984 |
| Seniority exact accuracy | 0.938 |
| Employment type exact accuracy | 0.969 |
| Required years exact accuracy | 0.953 |
| Required skills soft F1 | 0.792 |
| Nice-to-have skills soft F1 | 0.680 |

## What Improved

Prompt iteration produced measurable gains:

- `extract_v2.txt` tightened conservative extraction behavior.
- `extract_v3.txt` fixed seniority by adding explicit years-of-experience rules.
- `extract_v4.txt` refined skill-list extraction and improved skill F1.

The benchmark now shows a healthy dev/test gap and gives a clear error profile for future work.

## Remaining Work

The biggest remaining quality target is remote/location extraction. Many postings mix office location, remote eligibility, relocation language, occasional travel, and work-setting labels in inconsistent ways. That makes `remote_policy` and exact `location` harder than salary or seniority.

Good next improvements:

- tighten remote/location labeling policy
- add a lightweight report viewer
- expand to about 100 reviewed examples
- add provider abstraction for Anthropic/Gemini
