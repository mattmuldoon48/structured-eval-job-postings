from structured_eval.evaluate import example_mismatches, soft_list_f1, token_overlap_score
from structured_eval.schema import JobPostingLabel


def test_token_overlap_scores_partial_location_match():
    assert token_overlap_score("Seattle, WA", "Seattle") > 0.6


def test_soft_list_f1_scores_related_skill_phrases():
    score = soft_list_f1(
        ["model optimization", "LLMs", "AI integration"],
        ["model performance optimization", "LLM integrations"],
    )

    assert score > 0.5


def test_example_mismatches_includes_partial_text_and_skill_misses():
    expected = JobPostingLabel(
        company="Acme Corp",
        title="Senior AI Engineer",
        location="Seattle, WA",
        required_skills=["RAG pipelines", "Python"],
    )
    actual = JobPostingLabel(
        company="Acme",
        title="Senior AI Engineer",
        location="Seattle",
        required_skills=["Java"],
    )

    mismatches = example_mismatches(expected, actual)
    fields = {mismatch["field"] for mismatch in mismatches}

    assert "company" in fields
    assert "location" in fields
    assert "required_skills" in fields
