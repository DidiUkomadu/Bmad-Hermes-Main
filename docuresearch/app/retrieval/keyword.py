"""Keyword / BM25 retrieval — Story 3.2.

Uses rank_bm25 (BM25L) for BM25-based keyword retrieval.
Swappable through the KeywordRetriever interface defined in interface.py.

Tokenization: lowercase + alphanumeric tokenization (consistent with the
project's existing tokenization approach in evaluation/runner.py).

Indexing: BM25 index is rebuilt from scratch on each call to index_passages().
This follows the approved implementation recommendation of full reindex when
passages are added or removed.

Empty index behavior: returns an empty list.
Empty query behavior: returns an empty list.

Note on BM25L vs BM25Okapi:
  BM25L uses a different IDF formula that avoids zero IDF values for terms
  appearing in more than half the corpus. BM25Okapi (the default rank_bm25
  variant) can produce zero IDF for frequent terms, which would cause all
  scores to be zero. BM25L is the better choice for the MVP corpus.
"""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25L

from app.retrieval.interface import KeywordRetriever, RetrievalResult


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing.

    Lowercases and splits on non-alphanumeric characters, keeping tokens
    with length > 1. Consistent with the project's existing tokenization
    in evaluation/runner.py.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class BM25RetrieverImpl(KeywordRetriever):
    """BM25 keyword retriever using rank_bm25 (BM25L).

    Tokenizes passage text and builds a BM25 index. Queries are tokenized
    the same way and scored against the index.

    Scoring:
    - Raw BM25 scores from BM25L are normalized to [0, 1] by dividing by
      the maximum score in the result set. When all scores are zero, results
      receive a score of 0.0.
    - Ties are broken deterministically by passage_id ascending.

    Empty index behavior: returns an empty list.
    Empty query behavior: returns an empty list (no tokens to match).
    """

    def __init__(self) -> None:
        self._tokenized_passages: list[list[str]] = []
        self._passage_ids: list[str] = []
        self._bm25_index: BM25L | None = None

    def index_passages(self, passages: list[dict[str, Any]]) -> None:
        """Index passages for BM25 retrieval.

        Each dict must contain:: {"id": str, "text": str}

        The BM25 index is rebuilt from scratch. Passages are stored in the
        order they appear in the input list.
        """
        self._tokenized_passages = []
        self._passage_ids = []

        for p in passages:
            pid = p.get("id")
            text = p.get("text", "")
            if pid is not None and isinstance(text, str):
                tokens = _tokenize(text)
                if tokens:  # Skip passages with no meaningful tokens
                    self._tokenized_passages.append(tokens)
                    self._passage_ids.append(pid)

        if self._tokenized_passages:
            self._bm25_index = BM25L(self._tokenized_passages)
        else:
            self._bm25_index = None

    def remove_passages(self, passage_ids: list[str]) -> None:
        """Remove indexed passages by ID and rebuild the index."""
        remove_set = set(passage_ids)
        new_tokenized: list[list[str]] = []
        new_ids: list[str] = []

        for tokens, pid in zip(self._tokenized_passages, self._passage_ids):
            if pid not in remove_set:
                new_tokenized.append(tokens)
                new_ids.append(pid)

        self._tokenized_passages = new_tokenized
        self._passage_ids = new_ids

        if self._tokenized_passages:
            self._bm25_index = BM25L(self._tokenized_passages)
        else:
            self._bm25_index = None

    def search(self, query: str, top_n: int = 20) -> list[RetrievalResult]:
        """Return the top *top_n* passages ranked by BM25 relevance.

        Args:
            query: The search query text.
            top_n: Maximum number of results to return (default 20).

        Returns:
            List of RetrievalResult sorted by score descending, ties broken
            by passage_id ascending. Empty list when the index is empty or
            the query has no tokens.
        """
        if self._bm25_index is None:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Get raw BM25 scores for all passages
        raw_scores = self._bm25_index.get_scores(query_tokens)

        # Collect results with non-zero scores
        results: list[RetrievalResult] = []
        for i, score in enumerate(raw_scores):
            if score > 0.0 and i < len(self._passage_ids):
                results.append(
                    RetrievalResult(
                        passage_id=self._passage_ids[i],
                        score=score,
                        index_type="keyword",
                    )
                )

        if not results:
            return []

        # Normalize scores to [0, 1] by creating new RetrievalResult objects
        max_score = max(r.score for r in results)
        if max_score > 0.0:
            results = [
                RetrievalResult(
                    passage_id=r.passage_id,
                    score=r.score / max_score,
                    index_type=r.index_type,
                )
                for r in results
            ]

        # Sort: score descending, then passage_id ascending for ties
        results.sort(key=lambda r: (-r.score, r.passage_id))
        return results[:top_n]
