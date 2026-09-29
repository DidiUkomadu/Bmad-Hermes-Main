"""Document scope filtering — Story 3.4.

Implements document scope as a retrieval constraint that filters the passage
corpus BEFORE retrieval scoring, not as a silent post-filter.

Supports:
- All documents (no scope restriction)
- Specific document (only passages from that document)

The scope is applied at index time for semantic and keyword retrieval, and
is compatible with both retrieval backends. An unknown document ID in
specific scope returns an empty result set cleanly.
"""

from __future__ import annotations

from typing import Any

from app.retrieval.interface import (
    DocumentScope,
    KeywordRetriever,
    RetrievalResult,
    SemanticRetriever,
)


class ScopedSemanticRetriever:
    """Wrapper that applies document scope to semantic retrieval.

    Filters the passage corpus to only those matching the scope BEFORE
    indexing, so retrieval scoring only considers in-scope passages.

    Usage:
        base = SemanticRetrieverImpl()
        scoped = ScopedSemanticRetriever(base)
        scoped.index_passages(passages, scope=DocumentScope(mode="specific", document_id="doc-1"))
        results = scoped.search("query", top_n=20)
    """

    def __init__(self, base: SemanticRetriever) -> None:
        self._base = base

    def index_passages(
        self, passages: list[dict[str, Any]], scope: DocumentScope
    ) -> None:
        """Index passages filtered by document scope.

        When scope.mode is "all", all passages are indexed.
        When scope.mode is "specific", only passages matching the
        document_id are indexed. Unknown document IDs result in an
        empty index.
        """
        if scope.mode == "all":
            self._base.index_passages(passages)
        elif scope.mode == "specific" and scope.document_id:
            filtered = [
                p for p in passages
                if p.get("document_id") == scope.document_id
            ]
            self._base.index_passages(filtered)
        else:
            # Unknown or invalid scope — index nothing
            self._base.index_passages([])

    def remove_passages(self, passage_ids: list[str]) -> None:
        """Remove passages from the index."""
        self._base.remove_passages(passage_ids)

    def search(self, query: str, top_n: int = 20) -> list[RetrievalResult]:
        """Search within the scoped index."""
        return self._base.search(query, top_n=top_n)


class ScopedKeywordRetriever:
    """Wrapper that applies document scope to keyword (BM25) retrieval.

    Filters the passage corpus to only those matching the scope BEFORE
    indexing, so retrieval scoring only considers in-scope passages.

    Usage:
        base = BM25RetrieverImpl()
        scoped = ScopedKeywordRetriever(base)
        scoped.index_passages(passages, scope=DocumentScope(mode="specific", document_id="doc-1"))
        results = scoped.search("query", top_n=20)
    """

    def __init__(self, base: KeywordRetriever) -> None:
        self._base = base

    def index_passages(
        self, passages: list[dict[str, Any]], scope: DocumentScope
    ) -> None:
        """Index passages filtered by document scope.

        When scope.mode is "all", all passages are indexed.
        When scope.mode is "specific", only passages matching the
        document_id are indexed. Unknown document IDs result in an
        empty index.
        """
        if scope.mode == "all":
            self._base.index_passages(passages)
        elif scope.mode == "specific" and scope.document_id:
            filtered = [
                p for p in passages
                if p.get("document_id") == scope.document_id
            ]
            self._base.index_passages(filtered)
        else:
            # Unknown or invalid scope — index nothing
            self._base.index_passages([])

    def remove_passages(self, passage_ids: list[str]) -> None:
        """Remove passages from the index."""
        self._base.remove_passages(passage_ids)

    def search(self, query: str, top_n: int = 20) -> list[RetrievalResult]:
        """Search within the scoped index."""
        return self._base.search(query, top_n=top_n)
