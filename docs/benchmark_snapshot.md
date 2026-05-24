# Benchmark Snapshot

Current benchmark for `structured-eval-job-postings`.

## Run Context

- Dataset: 64 reviewed AI/ML job postings
- Split: 43 dev examples, 21 test examples
- Model: `gpt-4.1-mini`
- Prompt: `extract_v4.txt`
- Full replay score: `0.920`
- Dev score: `0.927`
- Test score: `0.906`
- Test gap: `-0.021`

## Field Metrics

| Metric | Score |
| --- | ---: |
| Company exact accuracy | 0.984 |
| Title exact accuracy | 0.938 |
| Seniority exact accuracy | 0.938 |
| Employment type exact accuracy | 0.969 |
| Location exact accuracy | 0.797 |
| Remote policy exact accuracy | 0.766 |
| Salary min/max exact accuracy | 0.984 / 1.000 |
| Required years exact accuracy | 0.953 |
| Security clearance exact accuracy | 1.000 |
| Sponsorship exact accuracy | 0.984 |
| Company normalized text score | 0.997 |
| Title normalized text score | 0.983 |
| Location normalized text score | 0.913 |
| Required skills exact F1 | 0.711 |
| Required skills soft F1 | 0.792 |
| Nice-to-have skills exact F1 | 0.654 |
| Nice-to-have skills soft F1 | 0.680 |

## Prompt Iteration

Historical comparison on the original 34-example dataset:

| Prompt | Overall | Required skills soft F1 | Nice-to-have skills soft F1 | Remote policy accuracy |
| --- | ---: | ---: | ---: | ---: |
| `extract_v1.txt` | 0.854 | 0.573 | 0.442 | 0.794 |
| `extract_v2.txt` | 0.864 | 0.661 | 0.473 | 0.824 |
| `extract_v3.txt` | 0.887 | 0.647 | 0.469 | 0.824 |
| `extract_v4.txt` | 0.889 | 0.674 | 0.479 | 0.853 |

## Current Error Profile

The main remaining mismatch groups are:

| Field | Mismatches | Notes |
| --- | ---: | --- |
| `required_skills` | 30 | Mostly canonical phrasing and list-boundary differences. |
| `nice_to_have_skills` | 28 | Optional sections vary widely; some labels intentionally omit broad preferences. |
| `remote_policy` | 15 | Ambiguous postings mix remote, onsite, eligibility geography, and occasional travel. |
| `location` | 13 | Mostly exact-format misses; normalized location score is much stronger. |

## Interpretation

The eval harness is now stable enough for prompt regression testing: dev/test scores are close, schema validation catches malformed outputs, and replay mode lets CI exercise scoring without model calls. The highest-value next quality improvement is remote/location policy, not broad prompt rewrites.
