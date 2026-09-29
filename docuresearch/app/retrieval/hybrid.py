"""Hybrid retrieval — Story 3.3.

Combines semantic and keyword retrieval signals into a single ranked result set.

Design:
- Delegates to independently usable SemanticRetriever and KeywordRetriever
- Normalizes each signal to [0, 1] before combination
- Default weighting: 0.5 semantic + 0.5 keyword (configurable via HybridWeighting)
- Deduplicates passage IDs, keeping the higher combined score
- Deterministic tie-breaking by passage_id ascending
- Compatible with DocumentScope via ScopedHybridRetriever

Normalization:
  Semantic scores are already in [0, 1] (cosine → (cos+1)/2).
  Keyword scores are normalized to [0, 1] by dividing by the max score in
  the result set (done by BM25RetrieverImpl.search()).
  Both are treated as equally scaled 0–1 signals and combined as:
      combined = semantic_weight * semantic_score + keyword_weight * keyword_score

A passage that appears in only one result set gets a zero contribution from
the missing signal, so strong performance in one signal is still reflected.
"""

from __future__ import annotations

from typing import Any

from app.retrieval.interface import (
    DocumentScope,
    HybridWeighting,
    KeywordRetriever,
    RetrievalResult,
    SemanticRetriever,
)


class HybridRetriever:
    """Combines semantic and keyword retrieval into a unified ranked result.

    Args:
        semantic: A SemanticRetriever instance (already configured).
        keyword: A KeywordRetriever instance (already configured).
        weighting: How to blend the two signals. Defaults to 0.5/0.5.
    """

    def __init__(
        self,
        semantic: SemanticRetriever,
        keyword: KeywordRetriever,
        weighting: HybridWeighting | None = None,
    ) -> None:
        self._semantic = semantic
        self._keyword = keyword
        self._weighting = weighting or HybridWeighting()

    def index_passages(self, passages: list[dict[str, Any]]) -> None:
        """Index passages into both the semantic and keyword indexes.

        Passages are indexed independently into each retriever. Both must
        receive the full passage list so that each can build its own index
        representation (embeddings for semantic, tokenized text for keyword).
        """
        self._semantic.index_passages(passages)
        self._keyword.index_passages(passages)

    def remove_passages(self, passage_ids: list[str]) -> None:
        """Remove passages from both indexes."""
        self._semantic.remove_passages(passage_ids)
        self._keyword.remove_passages(passage_ids)

    def search(self, query: str, top_n: int = 20) -> list[RetrievalResult]:
        """Combine semantic and keyword results into a single ranked list.

        Steps:
        1. Run both retrievers with the same query.
        2. Normalize each result set independently (already [0,1] per retriever).
        3. Look up each passage's score from each signal.
        4. Compute combined score = w_s * s_score + w_k * k_score.
        5. Deduplicate by passage_id, keeping the max combined score.
        6. Sort by combined score desc, then passage_id asc.
        7. Return top_n results.

        Args:
            query: The search query text.
            top_n: Maximum number of results to return.

        Returns:
            List of RetrievalResult with index_type="hybrid".
        """
        # Request FULL result sets from both retrievers so that the hybrid
        # layer can consider every indexed passage for combination. The
        # individual retrievers' own top_n limits are bypassed here so that
        # a passage ranked below top_n in one signal but highly in the other
        # is not silently excluded before scoring.
        semantic_results = self._semantic.search(query)
        keyword_results = self._keyword.search(query)

        # Build score lookup dicts: passage_id -> score for each signal
        semantic_scores: dict[str, float] = {
            r.passage_id: r.score for r in semantic_results
        }
        keyword_scores: dict[str, float] = {
            r.passage_id: r.score for r in keyword_results
        }

        # All unique passage IDs across both signals
        all_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())

        if not all_ids:
            return []

        w_s = self._weighting.semantic_weight
        w_k = self._weighting.keyword_weight

        combined: list[RetrievalResult] = []
        for pid in all_ids:
            s_score = semantic_scores.get(pid, 0.0)
            k_score = keyword_scores.get(pid, 0.0)
            # Both s_score and k_score are already in [0, 1]
            combined_score = w_s * s_score + w_k * k_score
            combined.append(
                RetrievalResult(
                    passage_id=pid,
                    score=combined_score,
                    index_type="hybrid",
                )
            )

        # Sort: score descending, passage_id ascending for ties
        combined.sort(key=lambda r: (-r.score, r.passage_id))
        return combined[:top_n]


class ScopedHybridRetriever:
    """Wrapper that applies document scope to hybrid retrieval.

    Filters the passage corpus to only those matching the scope BEFORE
    indexing, so both the semantic and keyword indexes only contain
    in-scope passages.

    Usage:
        hybrid = HybridRetriever(semantic, keyword)
        scoped = ScopedHybridRetriever(hybrid)
        scoped.index_passages(passages, DocumentScope(mode="specific", document_id="doc-1"))
        results = scoped.search("query", top_n=20)
    """

    def __init__(self, base: HybridRetriever) -> None:
        self._base = base

    def index_passages(
        self, passages: list[dict[str, Any]], scope: DocumentScope
    ) -> None:
        """Index passages filtered by document scope into both retrievers."""
        if scope.mode == "all":
            self._base.index_passages(passages)
        elif scope.mode == "specific" and scope.document_id:
            filtered = [
                p for p in passages
                if p.get("document_id") == scope.document_id
            ]
            self._base.index_passages(filtered)
        else:
            self._base.index_passages([])

    def remove_passages(self, passage_ids: list[str]) -> None:
        """Remove passages from both scoped indexes."""
        self._base.remove_passages(passage_ids)

    def search(self, query: str, top_n: int = 20) -> list[RetrievalResult]:
        """Search within the scoped hybrid index."""
        return self._base.search(query, top_n=top_n)
