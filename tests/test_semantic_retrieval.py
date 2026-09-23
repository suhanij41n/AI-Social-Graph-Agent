import pytest

pytestmark = pytest.mark.integration


def test_semantic_search_interests_surfaces_storytelling_related_branches():
    """Spec §8: a query about 'creative storytelling' should retrieve interests
    related to film, photography, dance, or music even without exact keyword
    overlap — demonstrating semantic (not just keyword) retrieval."""
    from app.retrieval.semantic import semantic_search

    hits = semantic_search("creative storytelling", entity_type="interests", top_k=10)
    names = {h.payload["name"] for h in hits}
    storytelling_related = {"Arts", "Visual Arts", "Film", "Screenwriting", "Cinematography",
                             "Music", "Painting", "Illustration", "Editing"}
    assert names & storytelling_related, f"expected overlap with {storytelling_related}, got {names}"


def test_semantic_search_returns_scored_results_in_descending_order():
    from app.retrieval.semantic import semantic_search

    hits = semantic_search("AI and machine learning", entity_type="interests", top_k=5)
    assert len(hits) > 0
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_semantic_search_top_hit_is_relevant_for_direct_match():
    from app.retrieval.semantic import semantic_search

    hits = semantic_search("dance", entity_type="interests", top_k=3)
    top_names = {h.entity_id for h in hits}
    assert any("dance" in h.payload["name"].lower() for h in hits), top_names
