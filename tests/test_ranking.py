from app.ranking.explain import compare_candidates
from app.ranking.features import RankingFeatures
from app.ranking.scorer import score_candidates


def make_features(person_id: str, **overrides) -> RankingFeatures:
    defaults = dict(
        semantic_relevance=0.0, graph_proximity=0.0, relationship_strength=0.0,
        shared_interests=0.0, shared_communities=0.0, goal_alignment=0.0,
        location_relevance=0.0, recency=0.0,
    )
    defaults.update(overrides)
    return RankingFeatures(person_id=person_id, raw={}, **defaults)


def test_higher_semantic_relevance_ranks_first_when_all_else_equal():
    a = make_features("a", semantic_relevance=0.9)
    b = make_features("b", semantic_relevance=0.1)
    ranked = score_candidates([a, b], profile="hybrid")
    assert ranked[0].person_id == "a"


def test_scoring_is_deterministic_across_runs():
    candidates = [make_features(f"p{i}", semantic_relevance=i / 10) for i in range(10)]
    ranked_1 = score_candidates(candidates, profile="hybrid")
    ranked_2 = score_candidates(candidates, profile="hybrid")
    assert [r.person_id for r in ranked_1] == [r.person_id for r in ranked_2]
    assert [r.score for r in ranked_1] == [r.score for r in ranked_2]


def test_tie_break_is_deterministic_by_person_id():
    # identical features -> ties broken by person_id (per config/ranking_weights.yaml)
    a = make_features("zzz", semantic_relevance=0.5)
    b = make_features("aaa", semantic_relevance=0.5)
    ranked = score_candidates([a, b], profile="hybrid")
    assert ranked[0].person_id == "aaa"


def test_semantic_only_profile_ignores_graph_signals():
    a = make_features("a", semantic_relevance=0.1, graph_proximity=1.0, shared_communities=1.0)
    b = make_features("b", semantic_relevance=0.9, graph_proximity=0.0, shared_communities=0.0)
    ranked = score_candidates([a, b], profile="semantic_only")
    assert ranked[0].person_id == "b"


def test_graph_only_profile_ignores_semantic_relevance():
    a = make_features("a", semantic_relevance=0.9, graph_proximity=0.1, relationship_strength=0.1)
    b = make_features("b", semantic_relevance=0.0, graph_proximity=1.0, relationship_strength=1.0)
    ranked = score_candidates([a, b], profile="graph_only")
    assert ranked[0].person_id == "b"


def test_empty_candidate_list_returns_empty():
    assert score_candidates([], profile="hybrid") == []


def test_compare_candidates_identifies_deciding_signals():
    a = make_features("a", semantic_relevance=0.9, graph_proximity=0.9)
    b = make_features("b", semantic_relevance=0.1, graph_proximity=0.1)
    ranked = score_candidates([a, b], profile="hybrid")
    ranked_a = next(r for r in ranked if r.person_id == "a")
    ranked_b = next(r for r in ranked if r.person_id == "b")
    explanation = compare_candidates(ranked_a, ranked_b)
    assert explanation.candidate_a == "a"
    assert len(explanation.deciding_signals) > 0
    assert all(d.advantage == "a" for d in explanation.deciding_signals)
