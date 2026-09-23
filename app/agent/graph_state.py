"""LangGraph agent state (spec §10)."""
from __future__ import annotations

from typing import Any, Literal, TypedDict

Intent = Literal[
    "relationship_lookup",
    "community_discovery",
    "bridge_detection",
    "goal_discovery",
    "outside_network",
    "semantic_similarity",
    "location_query",
    "temporal_query",
    "cross_domain",
    "why_not",
]


class Entities(TypedDict, total=False):
    person_names: list[str]
    interest_names: list[str]
    community_names: list[str]
    location: str | None
    goal_text: str | None


class Constraints(TypedDict, total=False):
    location: str | None
    max_degree: int | None
    days: int | None
    top_k: int


class AgentState(TypedDict, total=False):
    user_query: str
    user_id: str  # the primary demo user issuing the query
    ranking_profile: str | None  # overrides config/ranking_weights.yaml active_profile (used by evaluation)
    intent: Intent
    entities: Entities
    constraints: Constraints

    candidate_people: list[str]  # person ids
    graph_evidence: dict[str, Any]
    semantic_evidence: dict[str, Any]
    ranking_features: dict[str, Any]
    ranked_candidates: list[dict]

    final_response: str
    tool_calls_log: list[str]
    errors: list[str]

    # for "why not" follow-up queries against the previous turn's results
    previous_ranked_candidates: list[dict] | None
    compare_pair: tuple[str, str] | None

    # runtime-only handles (not persisted/serialized), set once per run_agent() call
    _session: Any
    _graph: Any
