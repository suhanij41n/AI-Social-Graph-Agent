from app.agent.routing import (
    ANALYTICS,
    GRAPH_RETRIEVAL,
    PATH_ANALYSIS,
    SEMANTIC_RETRIEVAL,
    TEMPORAL,
    tool_nodes_for_intent,
)


def test_relationship_lookup_only_uses_path_analysis():
    assert tool_nodes_for_intent("relationship_lookup") == [PATH_ANALYSIS]


def test_bridge_detection_does_not_use_semantic_retrieval():
    nodes = tool_nodes_for_intent("bridge_detection")
    assert SEMANTIC_RETRIEVAL not in nodes
    assert ANALYTICS in nodes
    assert GRAPH_RETRIEVAL in nodes


def test_goal_discovery_uses_semantic_and_graph_and_path():
    nodes = tool_nodes_for_intent("goal_discovery")
    assert SEMANTIC_RETRIEVAL in nodes
    assert GRAPH_RETRIEVAL in nodes
    assert PATH_ANALYSIS in nodes
    assert ANALYTICS not in nodes


def test_temporal_query_uses_temporal_node():
    nodes = tool_nodes_for_intent("temporal_query")
    assert TEMPORAL in nodes


def test_semantic_similarity_only_uses_semantic_retrieval():
    assert tool_nodes_for_intent("semantic_similarity") == [SEMANTIC_RETRIEVAL]


def test_why_not_uses_no_tool_nodes():
    assert tool_nodes_for_intent("why_not") == []


def test_unknown_intent_falls_back_to_graph_and_semantic():
    nodes = tool_nodes_for_intent("nonexistent_intent")  # type: ignore[arg-type]
    assert GRAPH_RETRIEVAL in nodes
    assert SEMANTIC_RETRIEVAL in nodes


def test_not_every_query_uses_every_tool_node():
    """Spec §10: 'do not force every query through every tool.'"""
    all_nodes = {GRAPH_RETRIEVAL, SEMANTIC_RETRIEVAL, ANALYTICS, PATH_ANALYSIS, TEMPORAL}
    for intent in ["relationship_lookup", "semantic_similarity", "temporal_query"]:
        nodes = set(tool_nodes_for_intent(intent))
        assert nodes != all_nodes
        assert len(nodes) < len(all_nodes)
