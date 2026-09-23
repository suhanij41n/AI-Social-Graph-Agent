"""Wraps sentence-transformers for embedding bios, interests, skills,
project descriptions, and community descriptions.

Embeddings turn text into vectors such that semantically similar text ends
up close together in vector space, measured by cosine similarity
(the cosine of the angle between two vectors, in [-1, 1], where 1 = identical
direction/meaning). This lets us retrieve "a photographer interested in
storytelling" for a query about "creative storytelling" even though the
words don't overlap, because their sentence embeddings point in a similar
direction. See docs/ARCHITECTURE.md for more detail.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]
