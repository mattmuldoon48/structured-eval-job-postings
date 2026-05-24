# structured-eval-job-postings

A production-style Python LLM evaluation harness for structured extraction from job postings. This repository supports an LLM-assisted labeling workflow for generating draft labels, reviewing corrections, and building a trusted ground-truth dataset.

## Why this exists

This project demonstrates applied LLM engineering for AI roles by focusing on:
- structured output design with Pydantic schemas
- annotation workflow for clean ground truth
- validation and regression testing
- direct SDK usage without a large framework
- clear, professional project structure

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

## Data layout

- `data/raw/job_postings.jsonl` — raw job posting inputs
- `data/labeled/labeled_jobs.jsonl` — accepted labels
- `prompts/extract_v1.txt` — baseline extraction prompt template
- `prompts/extract_v2.txt` — more conservative extraction prompt template

Current dataset:

- 34 raw job postings
- 34 validated human-reviewed labels
- AI/ML, LLM, agentic AI, MLOps, and full-stack AI engineering roles

## Labeling workflow

1. Add raw job postings to `data/raw/job_postings.jsonl` as JSONL records with `id` and `text`.
2. Run `python scripts/label_next.py`.
3. Review the LLM draft label, accept it, or edit fields.
4. Approved labels are appended to `data/labeled/labeled_jobs.jsonl`.

## Validation

Run label validation for all labeled examples:

```bash
python scripts/validate_labels.py
```

Run the local non-API checks:

```bash
python -m pytest -q
python scripts/validate_labels.py
```

GitHub Actions runs these checks on pushes and pull requests.

## Evaluation

Run the extraction eval against the labeled dataset with the default prompt, `extract_v2.txt`:

```bash
python scripts/run_eval.py
```

Run a specific prompt version:

```bash
python scripts/run_eval.py --prompt extract_v2.txt
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

Replay an existing prediction file without making model calls:

```bash
python scripts/run_eval.py \
  --replay-predictions reports/runs/<run-id>/predictions.jsonl
```

For a quick smoke test:

```bash
python scripts/run_eval.py --limit 1
```

The eval runner:

- generates model predictions for each raw job posting
- validates predictions against the Pydantic schema
- compares predictions to labeled ground truth
- reports strict exact-match metrics for enums, numbers, and booleans
- reports normalized text scores for company, title, and location
- reports exact and soft F1 scores for skill lists
- can fail the run when quality gates are not met
- records token usage and latency for each model call
- can estimate run cost from explicit per-token pricing inputs
- can replay saved predictions to regenerate reports without API calls
- writes run artifacts under `reports/runs/`

Latest eval snapshot using `gpt-4.1-mini` and `extract_v2.txt` on 34 examples:

| Metric | Score |
| --- | ---: |
| Overall mean score | 0.864 |
| Company exact accuracy | 0.971 |
| Title exact accuracy | 0.882 |
| Seniority exact accuracy | 0.588 |
| Employment type exact accuracy | 0.941 |
| Remote policy exact accuracy | 0.824 |
| Salary min/max exact accuracy | 1.000 / 1.000 |
| Required years exact accuracy | 0.941 |
| Company normalized text score | 0.994 |
| Title normalized text score | 0.968 |
| Location normalized text score | 0.841 |
| Required skills soft F1 | 0.661 |
| Nice-to-have skills soft F1 | 0.473 |

Prompt comparison:

| Prompt | Overall | Required skills soft F1 | Nice-to-have skills soft F1 | Remote policy accuracy |
| --- | ---: | ---: | ---: | ---: |
| `extract_v1.txt` | 0.854 | 0.573 | 0.442 | 0.794 |
| `extract_v2.txt` | 0.864 | 0.661 | 0.473 | 0.824 |

## What comes next

Future enhancements may include:
- held-out train/test splits
- CI integration for quality gates
- report comparison utilities across runs
- support for multiple LLM providers

## Notes

Keep the labeling workflow human-in-the-loop: only corrected labels should become ground truth.
