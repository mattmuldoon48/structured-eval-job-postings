# Eval Protocol

This project evaluates structured extraction from job postings using reviewed JSONL labels, deterministic splits, and field-level scoring.

## Label creation

1. Raw postings live in `data/raw/job_postings.jsonl` with stable `id` values and posting text.
2. Draft labels can be generated with `scripts/label_next.py` or `scripts/label_batch.py` using the current extraction prompt.
3. Batch-generated labels are marked with `needs human review` in `labeling_notes` and should not be treated as ground truth until reviewed.
4. Reviewed labels live in `data/labeled/labeled_jobs.jsonl` and are validated against `JobPostingLabel` in `src/structured_eval/schema.py`.
5. Dataset integrity checks verify that raw records, labels, and split assignments have matching IDs and that no labels remain marked as draft.

## Dev/test split

Split assignments live in `data/splits/job_splits.jsonl`. They are generated deterministically from each job ID using `src/structured_eval/splits.py`, which hashes `seed:id` and maps the result into the configured split ratios.

The current checked-in split contains 43 dev examples and 21 test examples. The split is deterministic so future reruns keep the same examples in each split unless the seed, ratios, or input IDs change.

## Prompt iteration split

Prompt work should be iterated on the dev split. The test split should be used as a held-out check after prompt changes are selected.

The historical prompt table in the README is useful for context, but the repository does not currently commit the original per-run artifacts for those historical numbers. Treat the current dev/test protocol as the reproducible workflow for future prompt changes.

## Overall score calculation

`overall_mean_score` is an equal-weight mean of these component scores from `src/structured_eval/evaluate.py`:

- exact accuracy for structured enum, numeric, and boolean fields: `seniority`, `employment_type`, `remote_policy`, `salary_min`, `salary_max`, `required_years_experience`, `security_clearance_required`, and `sponsorship_available`
- normalized text score for `company`, `title`, and `location`
- soft list F1 for `required_skills` and `nice_to_have_skills`

Exact list F1 is reported for skill lists but is not included in `overall_mean_score`; the overall score uses soft list F1 for those fields.

## Test split limitations

The test split has 21 examples. That is enough to catch large regressions and show the evaluation loop, but it is not large enough to make high-confidence claims about production performance across all job-posting formats, locations, industries, and compensation styles.

Small test-set changes can move field metrics noticeably, especially for sparse fields such as salary, sponsorship, clearance, and nice-to-have skills. Results should be read as portfolio-scale evidence of methodology, not as a production benchmark.

## Improvements with more time

- Commit sanitized benchmark artifact bundles for important runs, including `summary.json`, `report.md`, `predictions.jsonl`, prompt version, model, command, and run date.
- Expand the reviewed dataset with more remote/hybrid, multi-location, salary, sponsorship, and clearance examples.
- Add a second label-review pass or spot audit for ambiguous fields such as remote policy and location.
- Keep prompt iteration on dev and reserve test for final checks; add a larger frozen test split before making stronger claims.
- Use provider-native structured output or JSON schema mode where available instead of relying only on post-hoc JSON extraction.
