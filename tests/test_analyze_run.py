from scripts.analyze_run import collect_mismatches
from structured_eval.schema import JobPostingLabel


def test_collect_mismatches_groups_by_field():
    expected = JobPostingLabel(
        company="Acme Corp",
        title="Senior AI Engineer",
        location="Philadelphia",
        remote_policy="hybrid",
        required_skills=["Python", "RAG"],
    )
    actual = JobPostingLabel(
        company="Acme",
        title="Senior AI Engineer",
        location="Remote",
        remote_policy="remote",
        required_skills=["Java"],
    )

    grouped = collect_mismatches(
        [
            {
                "id": "job-001",
                "expected": expected.model_dump(mode="json"),
                "actual": actual.model_dump(mode="json"),
            }
        ]
    )

    assert "company" in grouped
    assert "location" in grouped
    assert "remote_policy" in grouped
    assert "required_skills" in grouped
    assert grouped["remote_policy"][0]["id"] == "job-001"
