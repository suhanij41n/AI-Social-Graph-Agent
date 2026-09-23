"""Combines ranking features into a final score using config/ranking_weights.yaml.

Deterministic and reproducible: normalization is min-max over the current
candidate set, weights come from a fixed config, and ties break on a fixed
key order (spec §9).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import get_ranking_weights
from app.ranking.features import RankingFeatures

SIGNAL_NAMES = [
    "semantic_relevance", "graph_proximity", "relationship_strength",
    "shared_interests", "shared_communities", "goal_alignment",
    "location_relevance", "recency",
]


@dataclass
class RankedCandidate:
    person_id: str
    score: float
    features: RankingFeatures
    normalized_features: dict[str, float]
    weights_used: dict[str, float]


def _normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 if hi > 0 else 0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def score_candidates(
    features_list: list[RankingFeatures], profile: str | None = None
) -> list[RankedCandidate]:
    config = get_ranking_weights()
    profile = profile or config.get("active_profile", "hybrid")
    weights = config["profiles"][profile]
    normalize = config.get("normalize", True)
    tie_break_keys = config.get("tie_break", ["graph_proximity", "person_id"])

    if not features_list:
        return []

    raw_matrix = {
        name: [getattr(f, name) for f in features_list] for name in SIGNAL_NAMES
    }
    if normalize:
        norm_matrix = {name: _normalize(vals) for name, vals in raw_matrix.items()}
    else:
        norm_matrix = raw_matrix

    ranked: list[RankedCandidate] = []
    for idx, f in enumerate(features_list):
        norm_features = {name: norm_matrix[name][idx] for name in SIGNAL_NAMES}
        total = sum(weights.get(name, 0.0) * norm_features[name] for name in SIGNAL_NAMES)
        ranked.append(RankedCandidate(
            person_id=f.person_id,
            score=round(total, 6),
            features=f,
            normalized_features=norm_features,
            weights_used=weights,
        ))

    def sort_key(rc: RankedCandidate):
        key = [-rc.score]
        for tb in tie_break_keys:
            if tb == "person_id":
                key.append(rc.person_id)
            else:
                key.append(-rc.normalized_features.get(tb, 0.0))
        return tuple(key)

    ranked.sort(key=sort_key)
    return ranked
