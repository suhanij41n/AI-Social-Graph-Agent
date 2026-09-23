import pytest

from app.agent.nodes import no_results_node, understand_query_node, why_not_node


def test_no_results_node_produces_helpful_message():
    result = no_results_node({})
    assert "couldn't find" in result["final_response"].lower()
    assert result["ranked_candidates"] == []


def test_why_not_node_without_previous_context_is_graceful():
    state = {"previous_ranked_candidates": None, "compare_pair": None, "user_query": "why not?"}
    result = why_not_node(state)
    assert "previous" in result["final_response"].lower()
    assert "final_response" in result


def test_why_not_node_with_unknown_person_in_pair_is_graceful():
    state = {
        "previous_ranked_candidates": [{"person_id": "p1", "score": 0.5, "normalized_features": {}, "weights_used": {}}],
        "compare_pair": ("p1", "does_not_exist"),
        "user_query": "why not?",
    }
    result = why_not_node(state)
    assert "final_response" in result
    assert "ranking data" in result["final_response"].lower()


def test_understand_query_node_handles_malformed_llm_json(mocker):
    """Simulated LLM failure: the model returns non-JSON. Must degrade gracefully,
    not raise, and fall back to a sane default intent."""
    fake_response = mocker.Mock()
    fake_response.content = "not valid json at all"
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value = fake_response
    mocker.patch("app.agent.nodes.get_llm", return_value=mock_llm)

    fake_session = mocker.Mock()
    state = {"user_query": "asdkjaslkdj gibberish query???", "_session": fake_session}

    result = understand_query_node(state)
    assert result["intent"] == "cross_domain"  # documented fallback
    assert result["candidate_people"] == []
    assert result["errors"] == []


def test_understand_query_node_handles_neo4j_connection_failure(mocker):
    """Simulated Neo4j failure while resolving person names: must not crash the node,
    propagates as a normal exception the caller can catch (not silently swallowed)."""
    fake_response = mocker.Mock()
    fake_response.content = '{"intent": "relationship_lookup", "person_names": ["Maya"], "interest_names": [], "community_names": [], "location": null, "goal_text": null}'
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value = fake_response
    mocker.patch("app.agent.nodes.get_llm", return_value=mock_llm)

    failing_session = mocker.Mock()
    failing_session.run.side_effect = ConnectionError("Neo4j unreachable")

    state = {"user_query": "How am I connected to Maya?", "_session": failing_session}

    with pytest.raises(ConnectionError):
        understand_query_node(state)


def test_empty_query_string_is_handled_by_understand_query(mocker):
    fake_response = mocker.Mock()
    fake_response.content = '{"intent": "cross_domain", "person_names": [], "interest_names": [], "community_names": [], "location": null, "goal_text": null}'
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value = fake_response
    mocker.patch("app.agent.nodes.get_llm", return_value=mock_llm)

    fake_session = mocker.Mock()
    result = understand_query_node({"user_query": "", "_session": fake_session})
    assert result["intent"] == "cross_domain"
