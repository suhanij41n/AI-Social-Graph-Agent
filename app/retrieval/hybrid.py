"""Merges graph-based and semantic candidates into one unified candidate set,
carrying the raw per-signal evidence each retrieval mode contributed.

Graph retrieval answers: who is connected, how far, via which communities.
Semantic retrieval answers: who is meaning-relevant even without exact keyword
or graph overlap. Combining both lets goal/cross-domain queries surface
people that neither signal alone would rank highly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from app.graph.queries import get_person_by_id
from app.retrieval.graph_retrieval import GraphCandidate, classify_network_distance, enrich_with_shared_context
from app.retrieval.semantic import SemanticHit, semantic_search


@dataclass
class UnifiedCandidate:
    person_id: str
    name: str
    city: str
    role: str
    semantic_score: float = 0.0
    network_degree: int | None = None  # hops from user; None = disconnected
    path: list[str] | None = None
    relationship_strength: float | None = None
    shared_interests: list[str] = field(default_factory=list)
    shared_communities: list[str] = field(default_factory=list)
    source: set[str] = field(default_factory=set)  # {"graph", "semantic"}


def build_candidate_pool(
    session,
    g: nx.Graph,
    user_id: str,
    graph_candidate_ids: list[str] | None = None,
    semantic_query: str | None = None,
    semantic_top_k: int = 20,
) -> dict[str, UnifiedCandidate]:
    graph_candidate_ids = graph_candidate_ids or []
    semantic_hits: list[SemanticHit] = []
    if semantic_query:
        semantic_hits = semantic_search(semantic_query, entity_type="people", top_k=semantic_top_k)

    semantic_scores = {h.entity_id: h.score for h in semantic_hits}
    all_ids = set(graph_candidate_ids) | set(semantic_scores) - {user_id}

    graph_info: dict[str, GraphCandidate] = classify_network_distance(g, user_id, list(all_ids))
    enrich_with_shared_context(session, user_id, graph_info)

    pool: dict[str, UnifiedCandidate] = {}
    for pid in all_ids:
        person = get_person_by_id(session, pid)
        if person is None:
            continue
        gc = graph_info[pid]
        source = set()
        if pid in graph_candidate_ids:
            source.add("graph")
        if pid in semantic_scores:
            source.add("semantic")
        pool[pid] = UnifiedCandidate(
            person_id=pid,
            name=person["name"],
            city=person["city"],
            role=person["role"],
            semantic_score=semantic_scores.get(pid, 0.0),
            network_degree=gc.degree,
            path=gc.path,
            relationship_strength=gc.relationship_strength,
            shared_interests=gc.shared_interests,
            shared_communities=gc.shared_communities,
            source=source,
        )
    return pool
