import json
from pathlib import Path

import pytest
from structured_eval.label_assist import _load_prompt_template
from structured_eval.schema import JobPostingLabel, Seniority


def test_valid_sample_label_passes():
    sample = {
        "company": "Acme Corp",
        "title": "Senior Software Engineer",
        "seniority": "senior",
        "employment_type": "full_time",
        "location": "Seattle, WA",
        "remote_policy": "hybrid",
        "salary_min": 130000,
        "salary_max": 160000,
        "required_years_experience": 5,
        "required_skills": ["python", "distributed systems", "cloud infrastructure"],
        "nice_to_have_skills": ["kubernetes"],
        "security_clearance_required": False,
        "sponsorship_available": False,
        "labeling_notes": "Draft label from test data."
    }

    label = JobPostingLabel.model_validate(sample)
    assert label.company == "Acme Corp"
    assert label.seniority == Seniority.senior
    assert label.salary_min == 130000


def test_invalid_enum_fails():
    sample = {
        "company": "Acme Corp",
        "title": "Software Engineer",
        "seniority": "expert",
        "employment_type": "full_time",
        "location": "Seattle, WA",
        "remote_policy": "hybrid",
        "salary_min": 130000,
        "salary_max": 160000,
        "required_years_experience": 5,
        "required_skills": ["python"],
        "nice_to_have_skills": [],
        "security_clearance_required": False,
        "sponsorship_available": False,
        "labeling_notes": None
    }

    with pytest.raises(Exception):
        JobPostingLabel.model_validate(sample)


def test_negative_salary_fails():
    sample = {
        "company": "Acme Corp",
        "title": "Software Engineer",
        "seniority": "entry",
        "employment_type": "full_time",
        "location": "Seattle, WA",
        "remote_policy": "onsite",
        "salary_min": -50000,
        "salary_max": 120000,
        "required_years_experience": 2,
        "required_skills": ["python"],
        "nice_to_have_skills": [],
        "security_clearance_required": False,
        "sponsorship_available": None,
        "labeling_notes": None
    }

    with pytest.raises(Exception):
        JobPostingLabel.model_validate(sample)


def test_salary_max_less_than_salary_min_fails():
    sample = {
        "company": "Acme Corp",
        "title": "Software Engineer",
        "seniority": "entry",
        "employment_type": "full_time",
        "location": "Seattle, WA",
        "remote_policy": "onsite",
        "salary_min": 150000,
        "salary_max": 120000,
        "required_years_experience": 2,
        "required_skills": ["python"],
        "nice_to_have_skills": [],
        "security_clearance_required": False,
        "sponsorship_available": None,
        "labeling_notes": None
    }

    with pytest.raises(Exception):
        JobPostingLabel.model_validate(sample)


def test_extraction_prompt_template_loads():
    template = _load_prompt_template()
    assert "{job_text}" in template


def test_extraction_prompt_template_loads_custom_path():
    template = _load_prompt_template(Path("prompts/extract_v4.txt"))
    assert "{job_text}" in template
