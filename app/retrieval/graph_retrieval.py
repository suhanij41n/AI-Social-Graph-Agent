"""Graph-based candidate retrieval: who is connected, how far, via what."""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from app.analytics.algorithms import degree_of_connection, shortest_path
from app.graph.queries import (
    find_direct_connections,
    find_shared_communities,
    find_shared_interests,
    get_person_profile,
    search_people_by_filters,
)


@dataclass
class GraphCandidate:
    person_id: str
    degree: int | None  # hops from the requesting user; None = disconnected in current graph
    path: list[str] | None
    relationship_strength: float | None  # strength on the direct edge, if 1st-degree
    shared_interests: list[str] = field(default_factory=list)
    shared_communities: list[str] = field(default_factory=list)


def candidates_by_filters(session, city: str | None = None, interest_name: str | None = None) -> list[dict]:
    if not city and not interest_name:
        return []  # no filters specified: avoid returning the whole population as "candidates"
    return search_people_by_filters(session, city=city, interest_name=interest_name)


def classify_network_distance(
    g: nx.Graph, user_id: str, candidate_ids: list[str]
) -> dict[str, GraphCandidate]:
    """For each candidate, compute hop distance and shortest path from the user."""
    result: dict[str, GraphCandidate] = {}
    for cid in candidate_ids:
        degree = degree_of_connection(g, user_id, cid)
        path = shortest_path(g, user_id, cid)
        strength = None
        if degree == 1:
            edge = g.get_edge_data(user_id, cid)
            strength = edge.get("weight") if edge else None
        result[cid] = GraphCandidate(person_id=cid, degree=degree, path=path, relationship_strength=strength)
    return result


def enrich_with_shared_context(session, user_id: str, candidates: dict[str, GraphCandidate]) -> None:
    for cid, cand in candidates.items():
        cand.shared_interests = find_shared_interests(session, user_id, cid)
        cand.shared_communities = find_shared_communities(session, user_id, cid)


def outside_network_candidates(
    g: nx.Graph, user_id: str, semantic_candidate_ids: list[str], max_known_degree: int = 1
) -> list[str]:
    """From a semantic candidate pool, keep only those NOT within max_known_degree of the user
    (i.e. true outside-network discovery, not people already directly known)."""
    result = []
    for cid in semantic_candidate_ids:
        degree = degree_of_connection(g, user_id, cid)
        if degree is None or degree > max_known_degree:
            result.append(cid)
    return result
