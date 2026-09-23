"""'Why this / why not' comparisons based on actual ranking features (spec §18)."""
from __future__ import annotations

from dataclasses import dataclass

from app.ranking.scorer import SIGNAL_NAMES, RankedCandidate

SIGNAL_LABELS = {
    "semantic_relevance": "semantic relevance to your query",
    "graph_proximity": "closeness in the network",
    "relationship_strength": "relationship strength",
    "shared_interests": "shared interests",
    "shared_communities": "shared communities",
    "goal_alignment": "alignment with your stated goal",
    "location_relevance": "location match",
    "recency": "recency of interaction",
}


@dataclass
class FeatureDiff:
    signal: str
    label: str
    a_value: float
    b_value: float
    advantage: str  # "a", "b", or "tie"
    weight: float


@dataclass
class WhyNotExplanation:
    candidate_a: str  # ranked higher
    candidate_b: str  # ranked lower
    score_a: float
    score_b: float
    diffs: list[FeatureDiff]
    deciding_signals: list[FeatureDiff]  # top signals that favored A


def compare_candidates(a: RankedCandidate, b: RankedCandidate) -> WhyNotExplanation:
    diffs: list[FeatureDiff] = []
    for signal in SIGNAL_NAMES:
        av = a.normalized_features[signal]
        bv = b.normalized_features[signal]
        weight = a.weights_used.get(signal, 0.0)
        if abs(av - bv) < 1e-9:
            advantage = "tie"
        else:
            advantage = "a" if av > bv else "b"
        diffs.append(FeatureDiff(signal=signal, label=SIGNAL_LABELS[signal],
                                  a_value=av, b_value=bv, advantage=advantage, weight=weight))

    deciding = sorted(
        [d for d in diffs if d.advantage == "a"],
        key=lambda d: d.weight * (d.a_value - d.b_value),
        reverse=True,
    )[:3]

    return WhyNotExplanation(
        candidate_a=a.person_id, candidate_b=b.person_id,
        score_a=a.score, score_b=b.score, diffs=diffs, deciding_signals=deciding,
    )
