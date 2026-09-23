"""The agent's 11 explicit tools (spec §11).

Each tool is a thin adapter: it takes a Neo4j session and/or NetworkX graph
plus simple arguments, and returns plain data. All real logic lives in
app/graph, app/analytics, and app/retrieval so it's testable independently
of the LLM/LangGraph runtime. Tool selection itself is a deterministic
dispatch (app/agent/routing.py) driven by the LLM's intent classification,
rather than LLM function-calling — this keeps tool execution reliable and
independently testable while still satisfying "the agent decides which
tools it needs" (the LLM decides *intent*, which determines *which tools run*).
"""
from __future__ import annotations

import networkx as nx

from app.analytics.algorithms import (
    betweenness_centrality,
    degree_centrality,
    degree_of_connection,
    pagerank,
    shortest_path,
)
from app.analytics.bridge_detection import find_bridge_people
from app.graph.export import export_community_names, export_person_community_memberships
from app.graph.queries import (
    find_direct_connections,
    find_shared_communities,
    find_shared_interests,
    get_person_profile,
    get_temporal_relationships as _get_temporal_relationships,
    search_people_by_filters,
)
from app.retrieval.graph_retrieval import classify_network_distance, outside_network_candidates
from app.retrieval.semantic import semantic_search as _semantic_search


def search_people(session, city: str | None = None, interest_name: str | None = None) -> list[dict]:
    if not city and not interest_name:
        return []  # no filters: don't return the whole population as "candidates"
    return search_people_by_filters(session, city=city, interest_name=interest_name)


def semantic_search(query_text: str, entity_type: str = "people", top_k: int = 15) -> list[dict]:
    hits = _semantic_search(query_text, entity_type=entity_type, top_k=top_k)
    return [{"entity_id": h.entity_id, "score": h.score, "payload": h.payload} for h in hits]


def find_connections(session, person_id: str) -> list[dict]:
    return find_direct_connections(session, person_id)


def find_shortest_path(g: nx.Graph, source_id: str, target_id: str) -> dict:
    path = shortest_path(g, source_id, target_id)
    degree = degree_of_connection(g, source_id, target_id)
    return {"path": path, "degree": degree}


def find_shared_interests_tool(session, person_a: str, person_b: str) -> list[str]:
    return find_shared_interests(session, person_a, person_b)


def find_shared_communities_tool(session, person_a: str, person_b: str) -> list[str]:
    return find_shared_communities(session, person_a, person_b)


def resolve_community_name(session, raw_name: str, score_threshold: float = 0.25) -> str | None:
    """Resolve a possibly-generic phrase (e.g. 'dance') to the closest actual
    community name, using the community embeddings already in Qdrant. Falls
    back to an exact/substring match against real community names first
    since that's cheaper and more precise when the LLM got it right."""
    names = list(export_community_names(session).values())
    lowered = raw_name.lower()
    for name in names:
        if lowered == name.lower() or lowered in name.lower():
            return name
    hits = _semantic_search(raw_name, entity_type="communities", top_k=1)
    if hits and hits[0].score >= score_threshold:
        return hits[0].payload["name"]
    return None


def find_bridge_people_tool(
    session, g: nx.Graph, community_a_name: str, community_b_name: str, top_k: int = 5
) -> list[dict]:
    resolved_a = resolve_community_name(session, community_a_name)
    resolved_b = resolve_community_name(session, community_b_name)
    names = export_community_names(session)
    id_by_name = {v: k for k, v in names.items()}
    ca = id_by_name.get(resolved_a) if resolved_a else None
    cb = id_by_name.get(resolved_b) if resolved_b else None
    if ca is None or cb is None:
        return []
    memberships = export_person_community_memberships(session)
    results = find_bridge_people(g, memberships, ca, cb, top_k=top_k)
    return [
        {"person_id": r.person_id, "betweenness": r.betweenness,
         "communities_bridged": [resolved_a, resolved_b],
         "connecting_edges": r.connecting_edges}
        for r in results
    ]


def get_person_profile_tool(session, person_id: str) -> dict | None:
    return get_person_profile(session, person_id)


def analyze_community(session, g: nx.Graph, community_name: str) -> dict:
    resolved = resolve_community_name(session, community_name)
    names = export_community_names(session)
    id_by_name = {v: k for k, v in names.items()}
    community_id = id_by_name.get(resolved) if resolved else None
    if community_id is None:
        return {"community_name": community_name, "found": False}
    memberships = export_person_community_memberships(session)
    member_ids = [p for p, comms in memberships.items() if community_id in comms]
    subgraph_members = [m for m in member_ids if m in g]
    degree = degree_centrality(g)
    pr = pagerank(g)
    top_by_degree = sorted(subgraph_members, key=lambda p: degree.get(p, 0), reverse=True)[:5]
    top_by_pagerank = sorted(subgraph_members, key=lambda p: pr.get(p, 0), reverse=True)[:5]
    return {
        "community_name": community_name,
        "found": True,
        "member_count": len(member_ids),
        "top_hubs_by_degree": top_by_degree,
        "top_influencers_by_pagerank": top_by_pagerank,
    }


def find_relevant_outside_network(
    g: nx.Graph, user_id: str, semantic_hits: list[dict], max_known_degree: int = 1
) -> list[str]:
    ids = [h["entity_id"] for h in semantic_hits]
    return outside_network_candidates(g, user_id, ids, max_known_degree=max_known_degree)


def get_temporal_relationships(session, person_id: str, days: int = 30) -> list[dict]:
    return _get_temporal_relationships(session, person_id, days=days)
