from pathlib import Path

import pytest
from pydantic import ValidationError
import scripts.validate_labels as validate_labels_command
from structured_eval.label_assist import _load_prompt_template
from structured_eval.schema import JobPostingLabel, Seniority
from structured_eval.validate_labels import validate_label_file


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


def test_unknown_field_fails():
    sample = {
        "company": "Acme Corp",
        "employment_typ": "full_time",
    }

    with pytest.raises(ValidationError, match="employment_typ"):
        JobPostingLabel.model_validate(sample)


def test_label_file_allows_record_id_and_rejects_other_unknown_fields(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(
        '{"id":"job-001","company":"Acme"}\n'
        '{"id":"job-002","employment_typ":"full_time"}\n',
        encoding="utf-8",
    )

    errors = validate_label_file(path)

    assert len(errors) == 1
    assert errors[0][0] == 2
    assert "employment_typ" in errors[0][1]


@pytest.mark.parametrize(
    ("field", "value"),
    [("required_skills", [""]), ("nice_to_have_skills", ["   "])],
)
def test_blank_skill_entries_fail(field, value):
    with pytest.raises(ValidationError, match="Skill entries must not be blank"):
        JobPostingLabel.model_validate({field: value})


def test_skill_categories_must_not_overlap():
    with pytest.raises(ValidationError, match="must not overlap: python"):
        JobPostingLabel.model_validate(
            {
                "required_skills": ["Python"],
                "nice_to_have_skills": [" python "],
            }
        )


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


@pytest.mark.parametrize(
    "record",
    ['{"company":"Acme"}', '{"id":"   ","company":"Acme"}'],
)
def test_label_file_requires_nonblank_record_ids(tmp_path, record):
    path = tmp_path / "labels.jsonl"
    path.write_text(record + "\n", encoding="utf-8")

    errors = validate_label_file(path)

    assert errors == [
        (1, "record id must be a non-empty, non-whitespace string"),
    ]


def test_label_file_rejects_duplicate_record_ids(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(
        '{"id":"job-001","company":"Acme"}\n'
        '{"id":"job-001","company":"Beta"}\n',
        encoding="utf-8",
    )

    errors = validate_label_file(path)

    assert errors == [
        (2, "duplicate record id 'job-001'; first seen on line 1"),
    ]


def test_validate_label_command_fails_when_dataset_is_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        validate_labels_command,
        "LABELED_PATH",
        tmp_path / "missing.jsonl",
    )

    with pytest.raises(SystemExit) as exc_info:
        validate_labels_command.run()

    assert exc_info.value.code == 1
