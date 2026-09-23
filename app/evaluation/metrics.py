"""Evaluation metrics (spec §20)."""
from __future__ import annotations

from dataclasses import dataclass


def precision_at_k(retrieved: list[str], gold: list[str], k: int) -> float | None:
    """None means "not applicable" — used for open-ended queries (goal-based/outside-
    network discovery) that have no fixed gold answer set. Returning 0.0/1.0 for those
    would be misleading (an empty gold list would make ANY non-empty retrieval score
    0% "precision," inverting the intuition that returning good candidates is good)."""
    if not gold:
        return None
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r in gold)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float | None:
    if not gold:
        return None
    top_k = set(retrieved[:k])
    hits = sum(1 for g in gold if g in top_k)
    return hits / len(gold)


def mrr(retrieved: list[str], gold: list[str]) -> float | None:
    if not gold:
        return None
    for rank, person_id in enumerate(retrieved, start=1):
        if person_id in gold:
            return 1.0 / rank
    return 0.0


def path_correctness(agent_path: list[str] | None, gold_path: list[str] | None) -> bool:
    if gold_path is None:
        return agent_path is None
    if agent_path is None:
        return False
    # any valid shortest path of the same length counts as correct (multiple shortest paths may exist)
    return len(agent_path) == len(gold_path) and agent_path[0] == gold_path[0] and agent_path[-1] == gold_path[-1]


def evidence_groundedness(final_response: str, evidence_lines: list[str]) -> float:
    """Fraction of evidence-line "concepts" (by simple substring presence of key terms)
    that appear reflected in the response. Pragmatic, not a full NLI check."""
    if not evidence_lines:
        return 1.0 if not final_response.strip() else 0.0
    response_lower = final_response.lower()
    grounded = 0
    for line in evidence_lines:
        keywords = [w.strip(".,()") for w in line.lower().split() if len(w) > 4]
        if any(kw in response_lower for kw in keywords):
            grounded += 1
    return grounded / len(evidence_lines)


def hallucination_rate(final_response: str, valid_names: set[str]) -> float:
    """Rough check: fraction of capitalized bigrams in the response that don't match
    any valid candidate name. A simple, explainable proxy rather than full NER."""
    import re
    candidate_names = re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", final_response)
    if not candidate_names:
        return 0.0
    unmatched = [n for n in candidate_names if n not in valid_names]
    return len(unmatched) / len(candidate_names)


@dataclass
class LatencyBreakdown:
    total_seconds: float


def measure_latency(fn, *args, **kwargs) -> tuple[object, LatencyBreakdown]:
    import time
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, LatencyBreakdown(total_seconds=elapsed)
