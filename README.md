# structured-eval-job-postings

[![CI](https://github.com/mattmuldoon48/structured-eval-job-postings/actions/workflows/ci.yml/badge.svg)](https://github.com/mattmuldoon48/structured-eval-job-postings/actions/workflows/ci.yml)

A production-style Python LLM evaluation harness for structured extraction from job postings. It combines human-reviewed labels, Pydantic schema validation, prompt versioning, deterministic dev/test splits, and benchmark reports for measuring structured-output quality over time.

Current benchmark: `gpt-4.1-mini` with `extract_v4.txt` scores `0.920` overall on 64 reviewed job postings, with a `0.906` held-out test score.

## Why this exists

This project demonstrates applied LLM engineering for AI roles by focusing on:
- structured output design with Pydantic schemas
- annotation workflow for clean ground truth
- validation and regression testing
- direct SDK usage without a large framework
- clear, professional project structure

## Workflow

```text
raw job postings
  -> LLM draft labels
  -> human review
  -> validated JSONL ground truth
  -> prompt eval runs
  -> dev/test benchmark comparison
```

The goal is not just to call an LLM. The goal is to build a small, inspectable evaluation loop that can tell whether prompt changes improve extraction quality without silently breaking fields like seniority, remote policy, salary, or skill lists.

## Results

Latest eval snapshot using `gpt-4.1-mini` and `extract_v4.txt` on 64 examples:

| Metric | Score |
| --- | ---: |
| Overall mean score | 0.920 |
| Company exact accuracy | 0.984 |
| Title exact accuracy | 0.938 |
| Seniority exact accuracy | 0.938 |
| Employment type exact accuracy | 0.969 |
| Remote policy exact accuracy | 0.766 |
| Salary min/max exact accuracy | 0.984 / 1.000 |
| Required years exact accuracy | 0.953 |
| Company normalized text score | 0.997 |
| Title normalized text score | 0.983 |
| Location normalized text score | 0.913 |
| Required skills soft F1 | 0.792 |
| Nice-to-have skills soft F1 | 0.680 |

Latest dev/test benchmark:

| Split | Examples | Overall | Required skills soft F1 | Nice-to-have skills soft F1 | Seniority accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev | 43 | 0.927 | 0.826 | 0.739 | 0.930 |
| test | 21 | 0.906 | 0.722 | 0.560 | 0.952 |

See [docs/project_summary.md](docs/project_summary.md) for a reviewer-friendly overview and [docs/benchmark_snapshot.md](docs/benchmark_snapshot.md) for the current benchmark summary and error analysis.

## Known limitations and next improvements

- Remote policy is the weakest high-level field because job postings often mix remote, hybrid, onsite, and location-eligibility language.
- Location extraction still has edge cases around multi-location roles, remote roles with state restrictions, and recruiter-provided locations that conflict with the body text.
- Required vs nice-to-have skills are difficult to separate when postings list tools, responsibilities, and preferences in the same paragraph.
- The dataset is intentionally small and inspectable; adding 20-40 more diverse postings would make the held-out benchmark more representative.
- The next prompt pass should focus on remote policy rules, location normalization, and cleaner skill-list boundaries.

## Setup

1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -e .
```

3. Copy the example environment file

```bash
cp .env.example .env
```

4. Set your OpenAI credentials in `.env`

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-mini
```

## Quick commands

Run the local non-API checks:

```bash
make ci
```

Run the standard live dev/test benchmark:

```bash
make benchmark
```

Live labeling and eval commands require an OpenAI API key. CI uses replay-mode scoring where possible so the public checks do not require API credentials or spend tokens.

## Data layout

- `data/raw/job_postings.jsonl` — raw job posting inputs
- `data/labeled/labeled_jobs.jsonl` — accepted labels
- `data/splits/job_splits.jsonl` — deterministic dev/test split assignments
- `prompts/extract_v1.txt` — baseline extraction prompt template
- `prompts/extract_v2.txt` — conservative extraction prompt template
- `prompts/extract_v3.txt` — prompt with explicit seniority rules
- `prompts/extract_v4.txt` — default prompt with refined skill-list rules

Current dataset:

- 64 raw job postings
- 64 validated reviewed labels
- deterministic split: 43 dev examples, 21 test examples
- AI/ML, LLM, agentic AI, MLOps, and full-stack AI engineering roles

Example structured label:

```json
{
  "id": "job-001",
  "company": "example company",
  "title": "machine learning engineer",
  "seniority": "senior",
  "employment_type": "full_time",
  "location": "Philadelphia, PA",
  "remote_policy": "hybrid",
  "salary_min": 120000,
  "salary_max": 160000,
  "required_years_experience": 5,
  "required_skills": ["Python", "AWS", "LLMs", "MLOps"],
  "nice_to_have_skills": ["LangGraph", "RAG", "Kubernetes"],
  "security_clearance_required": false,
  "sponsorship_available": false,
  "labeling_notes": null
}
```

## Labeling workflow

1. Add raw job postings to `data/raw/job_postings.jsonl` as JSONL records with `id` and `text`.
2. Run `python scripts/label_next.py`.
3. Review the LLM draft label, accept it, or edit fields.
4. Approved labels are appended to `data/labeled/labeled_jobs.jsonl`.

For larger batches, generate draft labels for all currently unlabeled raw jobs:

```bash
python scripts/label_batch.py
```

Treat batch labels as draft ground truth until reviewed.

Use `labeling_notes` for ambiguous human decisions that should be visible during later prompt or policy reviews; leave it null when the label is straightforward.

Review draft labels one at a time:

```bash
python scripts/review_drafts.py
```

## Validation

Run label validation for all labeled examples:

```bash
python scripts/validate_labels.py
```

Run the local non-API checks:

```bash
python -m pytest -q
python scripts/validate_labels.py
python scripts/check_dataset.py
```

Or use the convenience command:

```bash
make ci
```

GitHub Actions runs these checks on pushes and pull requests.
CI also runs a replay-mode eval quality gate against a small fixture, which exercises scoring without requiring an API key.

## Evaluation

Run the extraction eval against the labeled dataset with the default prompt, `extract_v4.txt`:

```bash
python scripts/run_eval.py
```

Run a specific prompt version:

```bash
python scripts/run_eval.py --prompt extract_v2.txt
```

Run a split-specific eval:

```bash
python scripts/create_splits.py
python scripts/run_eval.py --split dev
python scripts/run_eval.py --split test
```

Run the standard dev/test benchmark and compare the splits:

```bash
python scripts/run_benchmark.py --prompt extract_v4.txt
```

Run with quality gates:

```bash
python scripts/run_eval.py \
  --min-overall 0.85 \
  --min-metric exact_accuracy.remote_policy=0.80 \
  --min-metric soft_list_f1.required_skills=0.60
```

Estimate run cost from token usage by passing current model rates:

```bash
python scripts/run_eval.py \
  --input-cost-per-1m 0.40 \
  --output-cost-per-1m 1.60
```

Cost estimates are advisory run metadata only: they depend on the rates passed at run time and do not change scoring, quality gates, or saved labels.

Replay an existing prediction file without making model calls:

```bash
python scripts/run_eval.py \
  --replay-predictions reports/runs/<run-id>/predictions.jsonl
```

Replay mode is the safest default when reviewing scoring changes: it validates and re-scores the saved `expected` and `actual` labels locally without making model calls. It does not reload ground truth from `data/labeled/labeled_jobs.jsonl`, so label corrections made after the original run are not reflected in replay scores. Use a live eval to measure prompt or model-output changes.

Likewise, `--split` filters the saved records' `split` field rather than consulting current split assignments. A full-dataset live run saves `split: null`; replay that file without `--split`, or use predictions from a split-specific run.

Compare two eval runs:

```bash
python scripts/compare_runs.py \
  reports/runs/<baseline-run-id>/summary.json \
  reports/runs/<candidate-run-id>/summary.json
```

Analyze mismatches from a prediction file:

```bash
python scripts/analyze_run.py reports/runs/<run-id>/predictions.jsonl
```

For a quick smoke test:

```bash
python scripts/run_eval.py --limit 1
python scripts/run_benchmark.py --limit-per-split 1
```

The eval runner:

- generates model predictions for each raw job posting
- validates predictions against the Pydantic schema
- compares predictions to labeled ground truth
- can run against the full dataset or a deterministic split
- reports strict exact-match metrics for enums, numbers, and booleans
- reports normalized text scores for company, title, and location
- reports exact and soft F1 scores for skill lists
- can fail the run when quality gates are not met
- records token usage and latency for each model call
- can estimate run cost from explicit per-token pricing inputs
- can replay saved predictions to regenerate reports without API calls
- can compare summary metrics across runs
- can summarize field-level mismatches from saved predictions
- can run a matched dev/test benchmark from one command
- writes run artifacts under `reports/runs/`

### Run artifact guide

Each `reports/runs/<run-id>/` directory contains:

- `summary.json` — machine-readable aggregate metrics, run configuration, usage, cost estimate, sample mismatches, and quality-gate results
- `report.md` — the same run summarized for human review
- `predictions.jsonl` — one expected/predicted record per completed example; use this file for replay and mismatch analysis
- `errors.jsonl` — written only when a live run stops on a model, parsing, or validation exception

Check `failed_examples` in `summary.json` and review `errors.jsonl` before treating a live run as complete. A partial run can still contain valid predictions for examples completed before the failure.

Historical prompt comparison on the original 34-example set:

| Prompt | Overall | Required skills soft F1 | Nice-to-have skills soft F1 | Remote policy accuracy |
| --- | ---: | ---: | ---: | ---: |
| `extract_v1.txt` | 0.854 | 0.573 | 0.442 | 0.794 |
| `extract_v2.txt` | 0.864 | 0.661 | 0.473 | 0.824 |
| `extract_v3.txt` | 0.887 | 0.647 | 0.469 | 0.824 |
| `extract_v4.txt` | 0.889 | 0.674 | 0.479 | 0.853 |

## What comes next

High-value next improvements:
- tighten remote/location policy and labels, currently the largest non-skill error source
- add a small Streamlit or static HTML report viewer for benchmark artifacts
- add a provider abstraction for Anthropic/Gemini alongside OpenAI
- expand to about 100 reviewed labels once the current policy is stable

## Notes

Keep the labeling workflow human-in-the-loop: only corrected labels should become ground truth.
