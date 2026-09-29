"""Semantic (embedding-based) retrieval — Story 3.1.

Uses sentence-transformers with all-MiniLM-L6-v2 as the default model.
Brute-force cosine similarity over stored passage embeddings.
No vector database — appropriate for the MVP corpus size.

The embedding model is swappable through the EmbeddingModel interface
defined in interface.py. The retriever depends on the interface, not on
any specific model implementation.
"""

from __future__ import annotations

import math
from typing import Any

from app.retrieval.interface import EmbeddingModel, RetrievalResult, SemanticRetriever


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """Concrete embedding model using sentence-transformers.

    Default model: all-MiniLM-L6-v2 (local, CPU, 384-dimensional).
    Swappable: any implementation of EmbeddingModel can be used instead.

    Model loading is lazy — the model is loaded on the first call to embed().
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (384 for all-MiniLM-L6-v2)."""
        if self._dim is None:
            self._ensure_model_loaded()
        assert self._dim is not None
        return self._dim

    def embed(self, text: str) -> list[float]:
        """Convert text to a dense vector using the sentence-transformers model.

        Args:
            text: The text to embed.

        Returns:
            List of floats of length ``dimension``.
        """
        self._ensure_model_loaded()
        assert self._model is not None
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _ensure_model_loaded(self) -> None:
        """Load the sentence-transformers model if not already loaded."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)
        self._dim = self._model.get_embedding_dimension()


class SemanticRetrieverImpl(SemanticRetriever):
    """Brute-force cosine similarity semantic retriever.

    Indexes passage embeddings and retrieves by computing cosine similarity
    between the query embedding and each indexed passage embedding.

    Scoring:
    - Cosine similarity is computed between query and passage embeddings.
    - Scores are normalized to [0, 1] using: normalized = (cos + 1) / 2.
    - Ties are broken deterministically by passage_id ascending.

    Empty index behavior: returns an empty list.
    """

    def __init__(self, embedding_model: EmbeddingModel | None = None) -> None:
        self._embedding_model = embedding_model or SentenceTransformerEmbeddingModel()
        # Internal index: dict[passage_id -> embedding]
        self._index: dict[str, list[float]] = {}

    def index_passages(self, passages: list[dict[str, Any]]) -> None:
        """Index passages for semantic retrieval.

        Each dict must contain:: {"id": str, "text": str, "embedding": list[float] | None}

        Passages without a stored embedding are skipped. If the same passage ID
        appears multiple times, the last entry wins.
        """
        for p in passages:
            pid = p.get("id")
            emb = p.get("embedding")
            if pid is not None and emb is not None and isinstance(emb, list):
                self._index[pid] = emb

    def remove_passages(self, passage_ids: list[str]) -> None:
        """Remove indexed passages by ID."""
        for pid in passage_ids:
            self._index.pop(pid, None)

    def search(self, query: str, top_n: int = 20) -> list[RetrievalResult]:
        """Return the top *top_n* passages ranked by semantic similarity.

        Args:
            query: The search query text.
            top_n: Maximum number of results to return (default 20).

        Returns:
            List of RetrievalResult sorted by score descending, ties broken
            by passage_id ascending. Empty list when the index is empty.
        """
        if not self._index:
            return []

        query_embedding = self._embedding_model.embed(query)
        query_norm = math.sqrt(sum(v * v for v in query_embedding))
        if query_norm == 0.0:
            return []

        results: list[RetrievalResult] = []
        for pid, passage_embedding in self._index.items():
            passage_norm = math.sqrt(sum(v * v for v in passage_embedding))
            if passage_norm == 0.0:
                continue

            dot_product = sum(
                q * p for q, p in zip(query_embedding, passage_embedding)
            )
            cosine = dot_product / (query_norm * passage_norm)
            # Normalize from [-1, 1] to [0, 1]
            normalized_score = max(0.0, (cosine + 1.0) / 2.0)

            results.append(
                RetrievalResult(
                    passage_id=pid,
                    score=normalized_score,
                    index_type="semantic",
                )
            )

        # Sort: score descending, then passage_id ascending for ties
        results.sort(key=lambda r: (-r.score, r.passage_id))
        return results[:top_n]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value in [-1, 1]. Returns 0.0 if either vector has zero norm.
    """
    norm_a = math.sqrt(sum(v * v for v in vec_a))
    norm_b = math.sqrt(sum(v * v for v in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return dot / (norm_a * norm_b)
