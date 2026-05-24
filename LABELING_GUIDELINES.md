# Labeling Guidelines

This document explains how to label each field for structured extraction from job postings.

## Fields

- `company`
  - Use the employer name from the posting header or signature.
  - If not available, use `null`.

- `title`
  - Use the official job title as advertised.
  - If the posting is generic or missing a title, use `null`.

- `seniority`
  - Choose one of: `intern`, `entry`, `mid`, `senior`, `staff`, `manager`, `director`, `executive`, `unknown`.
  - Derive from explicit level language such as "Senior", "Lead", "Director", or from years of experience.
  - Use `unknown` if the level is unclear.

- `employment_type`
  - Choose one of: `full_time`, `part_time`, `contract`, `internship`, `temporary`, `unknown`.
  - If the posting is unclear or uses generic terms, use `unknown`.

- `location`
  - Use the city, region, or country as listed.
  - If the posting is fully remote, use `null` and rely on `remote_policy`.

- `remote_policy`
  - Choose one of: `onsite`, `hybrid`, `remote`, `unknown`.
  - If the posting explicitly says work from home, use `remote`.
  - If it says a mix of office and remote, use `hybrid`.

- `salary_min` and `salary_max`
  - Use numeric values only.
  - If the salary range is not stated clearly, use `null`.
  - Do not infer salary values that are not explicitly in the posting.

- `required_years_experience`
  - Use a numeric value if the posting specifies required experience.
  - For a range like "3-5 years", use the minimum or the explicitly required floor.
  - Use `null` if not stated.

- `required_skills`
  - List explicit must-have skills from the posting.
  - Normalize skill phrases into concise terms (e.g. `python`, `machine learning`, `cloud`).

- `nice_to_have_skills`
  - List optional or preferred skills, not required.
  - If the posting does not distinguish optional skills, keep this list empty.

- `security_clearance_required`
  - Use `true` when the posting explicitly requests clearance.
  - Use `false` when the posting does not require it or does not mention it.

- `sponsorship_available`
  - Use `true` when sponsorship is explicitly offered.
  - Use `false` when sponsorship is explicitly denied.
  - Use `null` when the posting does not mention sponsorship.

- `labeling_notes`
  - Use this field for clarifications or uncertainty about the label.
  - Keep it short and factual.
