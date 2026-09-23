"""Qdrant client wrapper. One collection per entity type."""
from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import get_settings

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

COLLECTIONS = ["people", "interests", "skills", "projects", "communities"]


@lru_cache
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    if settings.qdrant_mode == "embedded":
        return QdrantClient(path=settings.qdrant_path)
    return QdrantClient(url=settings.qdrant_url)


def ensure_collections() -> None:
    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    for name in COLLECTIONS:
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(size=EMBEDDING_DIM, distance=qmodels.Distance.COSINE),
            )


def upsert_points(collection: str, ids: list[str], vectors, payloads: list[dict]) -> None:
    client = get_qdrant_client()
    # Qdrant point IDs must be int or UUID; use a stable positive int hash of the string id
    point_ids = [_stable_point_id(i) for i in ids]
    points = [
        qmodels.PointStruct(id=pid, vector=vec.tolist(), payload={**payload, "entity_id": original_id})
        for pid, vec, payload, original_id in zip(point_ids, vectors, payloads, ids)
    ]
    client.upsert(collection_name=collection, points=points)


def _stable_point_id(text_id: str) -> int:
    import hashlib
    return int(hashlib.sha1(text_id.encode()).hexdigest()[:12], 16)


def search(collection: str, query_vector, top_k: int = 10, query_filter: qmodels.Filter | None = None):
    client = get_qdrant_client()
    return client.query_points(
        collection_name=collection,
        query=query_vector.tolist(),
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    ).points
