"""Evidence objects, separated into 4 explicit tiers (spec §17):

  DatabaseFact      - a fact read directly from Neo4j (a relationship, a
                       community membership, a profile field).
  CalculatedMetric   - a value computed by a graph algorithm (betweenness,
                       PageRank, shortest-path length, community id).
  SemanticScore      - a cosine-similarity score from the vector store.
  LLMInterpretation  - the LLM's own phrasing/summary, explicitly marked as
                       interpretation rather than fact.

Keeping these separate lets app/agent/nodes.py::validate_evidence mechanically
check that every claim in a final response traces back to a DatabaseFact,
CalculatedMetric, or SemanticScore, and that LLMInterpretation is never the
sole source of a factual claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EvidenceKind = Literal["database_fact", "calculated_metric", "semantic_score", "llm_interpretation"]


@dataclass
class DatabaseFact:
    subject_id: str
    fact_type: str  # e.g. "relationship", "membership", "profile_field"
    description: str
    data: dict = field(default_factory=dict)
    kind: EvidenceKind = "database_fact"


@dataclass
class CalculatedMetric:
    subject_id: str
    metric_name: str  # e.g. "betweenness_centrality", "shortest_path", "network_degree"
    value: object
    description: str
    kind: EvidenceKind = "calculated_metric"


@dataclass
class SemanticScore:
    subject_id: str
    query_text: str
    score: float
    description: str
    kind: EvidenceKind = "semantic_score"


@dataclass
class LLMInterpretation:
    subject_id: str
    text: str
    kind: EvidenceKind = "llm_interpretation"


@dataclass
class EvidenceBundle:
    """All evidence gathered for one candidate, keyed for validation."""
    person_id: str
    facts: list[DatabaseFact] = field(default_factory=list)
    metrics: list[CalculatedMetric] = field(default_factory=list)
    semantic: list[SemanticScore] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.facts or self.metrics or self.semantic)

    def summary_lines(self) -> list[str]:
        lines = [f.description for f in self.facts]
        lines += [m.description for m in self.metrics]
        lines += [s.description for s in self.semantic]
        return lines
