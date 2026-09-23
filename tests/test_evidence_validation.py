from app.agent.evidence import CalculatedMetric, DatabaseFact, EvidenceBundle
from app.agent.nodes import validate_evidence_node


def _state_with_bundles(bundles: dict, ranked: list[dict]) -> dict:
    return {
        "graph_evidence": {"_evidence_bundles": bundles},
        "ranked_candidates": ranked,
    }


def test_candidate_with_evidence_is_kept():
    bundle = EvidenceBundle(person_id="p1")
    bundle.facts.append(DatabaseFact(subject_id="p1", fact_type="shared_interest",
                                      description="Shares interest in AI with you."))
    state = _state_with_bundles({"p1": bundle}, [{"person_id": "p1", "score": 0.8}])
    result = validate_evidence_node(state)
    assert len(result["ranked_candidates"]) == 1
    assert result["ranked_candidates"][0]["person_id"] == "p1"


def test_candidate_with_empty_bundle_is_dropped():
    empty_bundle = EvidenceBundle(person_id="p2")
    state = _state_with_bundles({"p2": empty_bundle}, [{"person_id": "p2", "score": 0.5}])
    result = validate_evidence_node(state)
    assert result["ranked_candidates"] == []


def test_candidate_with_no_bundle_at_all_is_dropped():
    state = _state_with_bundles({}, [{"person_id": "p3", "score": 0.5}])
    result = validate_evidence_node(state)
    assert result["ranked_candidates"] == []


def test_mixed_candidates_only_evidence_backed_survive():
    good_bundle = EvidenceBundle(person_id="good")
    good_bundle.metrics.append(CalculatedMetric(subject_id="good", metric_name="network_degree",
                                                  value=2, description="2-degree connection."))
    empty_bundle = EvidenceBundle(person_id="bad")
    state = _state_with_bundles(
        {"good": good_bundle, "bad": empty_bundle},
        [{"person_id": "good", "score": 0.9}, {"person_id": "bad", "score": 0.7}],
    )
    result = validate_evidence_node(state)
    ids = [rc["person_id"] for rc in result["ranked_candidates"]]
    assert ids == ["good"]


def test_evidence_bundle_summary_lines_reflects_all_tiers():
    bundle = EvidenceBundle(person_id="p1")
    bundle.facts.append(DatabaseFact(subject_id="p1", fact_type="x", description="fact line"))
    bundle.metrics.append(CalculatedMetric(subject_id="p1", metric_name="y", value=1, description="metric line"))
    lines = bundle.summary_lines()
    assert "fact line" in lines
    assert "metric line" in lines
