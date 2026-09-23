"""Semantic (vector) search over embedded entities."""
from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.embeddings import embed_text
from app.retrieval.vector_store import search


@dataclass
class SemanticHit:
    entity_id: str
    score: float
    payload: dict


def semantic_search(query_text: str, entity_type: str = "people", top_k: int = 10) -> list[SemanticHit]:
    """entity_type must be one of: people, interests, skills, projects, communities."""
    query_vector = embed_text(query_text)
    hits = search(entity_type, query_vector, top_k=top_k)
    return [
        SemanticHit(entity_id=h.payload["entity_id"], score=h.score, payload=h.payload)
        for h in hits
    ]
