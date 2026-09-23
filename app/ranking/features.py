"""Computes the 7 raw ranking signals per candidate (spec §9).

Each function is independently unit-testable and returns a value already
close to [0, 1] so app/ranking/scorer.py's min-max normalization has
sane, comparable inputs across a candidate set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from app.graph.queries import get_last_interaction
from app.retrieval.embeddings import embed_text
from app.retrieval.hybrid import UnifiedCandidate

TODAY = date(2026, 8, 24)

# hop distance -> proximity score; disconnected (None) -> 0.0
_DEGREE_PROXIMITY = {0: 1.0, 1: 1.0, 2: 0.6, 3: 0.3}


@dataclass
class QueryContext:
    goal_text: str | None = None
    location: str | None = None
    user_city: str | None = None
    user_country: str | None = None


@dataclass
class RankingFeatures:
    person_id: str
    semantic_relevance: float
    graph_proximity: float
    relationship_strength: float
    shared_interests: float
    shared_communities: float
    goal_alignment: float
    location_relevance: float
    recency: float
    raw: dict = field(default_factory=dict)


def _graph_proximity(degree: int | None) -> float:
    if degree is None:
        return 0.0
    return _DEGREE_PROXIMITY.get(degree, 0.1)


def _normalized_count(items: list, cap: int = 3) -> float:
    return min(1.0, len(items) / cap)


def _location_relevance(candidate: UnifiedCandidate, ctx: QueryContext) -> float:
    if ctx.location:
        return 1.0 if candidate.city.lower() == ctx.location.lower() else 0.0
    if ctx.user_city:
        return 1.0 if candidate.city.lower() == ctx.user_city.lower() else 0.0
    return 0.0


def _recency_score(session, user_id: str, candidate: UnifiedCandidate) -> float:
    if candidate.network_degree != 1:
        return 0.0
    last_interaction = get_last_interaction(session, user_id, candidate.person_id)
    if not last_interaction:
        return 0.0
    days_since = (TODAY - date.fromisoformat(last_interaction)).days
    return math.exp(-max(days_since, 0) / 365.0)


def _goal_alignment(candidate: UnifiedCandidate, ctx: QueryContext) -> float:
    if not ctx.goal_text:
        return candidate.semantic_score
    candidate_text = f"{candidate.name}, {candidate.role}. " + ", ".join(
        candidate.shared_interests + candidate.shared_communities
    )
    import numpy as np
    goal_vec = embed_text(ctx.goal_text)
    cand_vec = embed_text(candidate_text)
    return float(np.dot(goal_vec, cand_vec))


def compute_features(session, user_id: str, candidate: UnifiedCandidate, ctx: QueryContext) -> RankingFeatures:
    return RankingFeatures(
        person_id=candidate.person_id,
        semantic_relevance=max(0.0, candidate.semantic_score),
        graph_proximity=_graph_proximity(candidate.network_degree),
        relationship_strength=candidate.relationship_strength or 0.0,
        shared_interests=_normalized_count(candidate.shared_interests),
        shared_communities=_normalized_count(candidate.shared_communities, cap=2),
        goal_alignment=max(0.0, _goal_alignment(candidate, ctx)),
        location_relevance=_location_relevance(candidate, ctx),
        recency=_recency_score(session, user_id, candidate),
        raw={
            "network_degree": candidate.network_degree,
            "path": candidate.path,
            "shared_interests": candidate.shared_interests,
            "shared_communities": candidate.shared_communities,
            "source": sorted(candidate.source),
        },
    )
