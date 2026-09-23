"""Pure-logic intent -> tool-node dispatch (spec §10: "do not force every
query through every tool"). No LLM call here — this is a deterministic table
so routing correctness is unit-testable without mocking the LLM.
"""
from __future__ import annotations

from app.agent.graph_state import Intent

GRAPH_RETRIEVAL = "graph_retrieval_node"
SEMANTIC_RETRIEVAL = "semantic_retrieval_node"
ANALYTICS = "analytics_node"
PATH_ANALYSIS = "path_analysis_node"
TEMPORAL = "temporal_node"

INTENT_TO_TOOL_NODES: dict[Intent, list[str]] = {
    "relationship_lookup": [PATH_ANALYSIS],
    "community_discovery": [GRAPH_RETRIEVAL, ANALYTICS],
    "bridge_detection": [GRAPH_RETRIEVAL, ANALYTICS],
    "goal_discovery": [SEMANTIC_RETRIEVAL, GRAPH_RETRIEVAL, PATH_ANALYSIS],
    "outside_network": [SEMANTIC_RETRIEVAL, GRAPH_RETRIEVAL, PATH_ANALYSIS],
    "semantic_similarity": [SEMANTIC_RETRIEVAL],
    "location_query": [GRAPH_RETRIEVAL],
    "temporal_query": [TEMPORAL, GRAPH_RETRIEVAL],
    "cross_domain": [SEMANTIC_RETRIEVAL, GRAPH_RETRIEVAL],
    "why_not": [],  # handled separately, skips retrieval entirely
}


def tool_nodes_for_intent(intent: Intent) -> list[str]:
    return INTENT_TO_TOOL_NODES.get(intent, [GRAPH_RETRIEVAL, SEMANTIC_RETRIEVAL])
