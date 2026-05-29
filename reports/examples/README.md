# Benchmark Artifact Examples

This directory is for small, sanitized examples of benchmark artifacts that reviewers can inspect without running live API calls.

No real benchmark artifact bundle is currently committed here. Do not treat this directory as evidence for the README benchmark numbers until a real run bundle is added.

A real artifact bundle should include:

- `summary.json` — run metadata, examples scored, field metrics, quality gates, usage, and cost estimate when available
- `report.md` — markdown rendering of the same run for quick review
- `predictions.jsonl` — per-example expected and actual labels, with any sensitive or proprietary text removed
- prompt name or prompt hash
- model name
- exact command used to produce the run
- run date

Live runs write full artifacts under `reports/runs/<run-id>/`. That path is gitignored so local benchmark output is not accidentally committed. Copy only intentional, reviewed, sanitized examples into this directory.
