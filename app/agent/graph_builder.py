"""Compiles the LangGraph StateGraph and exposes run_agent() as the entrypoint."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.graph_state import AgentState
from app.agent.nodes import (
    community_summary_node,
    fuse_evidence_node,
    generate_explanation_node,
    no_results_node,
    rank_candidates_node,
    run_tools_node,
    understand_query_node,
    validate_evidence_node,
    why_not_node,
)
from app.graph.connection import get_session
from app.graph.export import export_person_graph
from app.logging_config import get_logger

logger = get_logger(__name__)


def _route_after_understand(state: AgentState) -> str:
    if state["intent"] == "why_not":
        return "why_not_response"
    entities = state.get("entities", {})
    if (
        state["intent"] == "community_discovery"
        and not entities.get("community_names")
        and not entities.get("person_names")
    ):
        return "community_summary_response"
    return "run_tools"


def _route_after_validate(state: AgentState) -> str:
    if not state.get("ranked_candidates"):
        return "no_results_response"
    return "generate_explanation"


def build_agent():
    builder = StateGraph(AgentState)

    builder.add_node("understand_query", understand_query_node)
    builder.add_node("run_tools", run_tools_node)
    builder.add_node("fuse_evidence", fuse_evidence_node)
    builder.add_node("rank_candidates", rank_candidates_node)
    builder.add_node("validate_evidence", validate_evidence_node)
    builder.add_node("generate_explanation", generate_explanation_node)
    builder.add_node("no_results_response", no_results_node)
    builder.add_node("why_not_response", why_not_node)
    builder.add_node("community_summary_response", community_summary_node)

    builder.add_edge(START, "understand_query")
    builder.add_conditional_edges(
        "understand_query", _route_after_understand,
        ["run_tools", "why_not_response", "community_summary_response"],
    )
    builder.add_edge("run_tools", "fuse_evidence")
    builder.add_edge("fuse_evidence", "rank_candidates")
    builder.add_edge("rank_candidates", "validate_evidence")
    builder.add_conditional_edges(
        "validate_evidence", _route_after_validate, ["generate_explanation", "no_results_response"]
    )
    builder.add_edge("generate_explanation", END)
    builder.add_edge("no_results_response", END)
    builder.add_edge("why_not_response", END)
    builder.add_edge("community_summary_response", END)

    return builder.compile()


_AGENT = None


def get_agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()
    return _AGENT


def run_agent(
    query: str,
    user_id: str = "p_0001",
    previous_ranked_candidates: list[dict] | None = None,
    compare_pair: tuple[str, str] | None = None,
    ranking_profile: str | None = None,
) -> AgentState:
    agent = get_agent()
    with get_session() as session:
        g = export_person_graph(session)
        initial_state: AgentState = {
            "user_query": query,
            "user_id": user_id,
            "previous_ranked_candidates": previous_ranked_candidates,
            "compare_pair": compare_pair,
            "ranking_profile": ranking_profile,
            "_session": session,
            "_graph": g,
        }
        result = agent.invoke(initial_state)
    return result
