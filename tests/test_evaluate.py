from structured_eval.evaluate import soft_list_f1, token_overlap_score


def test_token_overlap_scores_partial_location_match():
    assert token_overlap_score("Seattle, WA", "Seattle") > 0.6


def test_soft_list_f1_scores_related_skill_phrases():
    score = soft_list_f1(
        ["model optimization", "LLMs", "AI integration"],
        ["model performance optimization", "LLM integrations"],
    )

    assert score > 0.5
