"""Retrieval abstractions and shared types.

Defines the interfaces that make the retrieval pipeline swappable:

- EmbeddingModel: text → vector (swappable embedding backend)
- SemanticRetriever: embedding-based retrieval (abstract)
- KeywordRetriever: lexical/BM25 retrieval (abstract)
- RetrievalResult: ranked passage result with score and source type
- DocumentScope: all-documents vs specific-document constraint
- HybridWeighting: configurable semantic/keyword blend
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RetrievalResult:
    """A single ranked retrieval result.

    Attributes:
        passage_id: Stable passage identifier (resolvable via the content store).
        score: Normalized relevance score in [0, 1].
        index_type: Which retrieval signal produced this result:
            ``"semantic"``, ``"keyword"``, or ``"hybrid"``.
    """

    passage_id: str
    score: float
    index_type: str


class EmbeddingModel(Protocol):
    """Swappable abstraction for text → vector embedding.

    Implementations must be deterministic for the same input text so that
    retrieval results are reproducible across runs (important for evaluation).
    """

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        ...

    def embed(self, text: str) -> list[float]:
        """Convert *text* to a dense vector representation.

        Returns:
            List of floats of length ``dimension``.
        """
        ...


class SemanticRetriever:
    """Abstract semantic (embedding-based) retriever.

    Concrete implementations must:
    - embed queries via the configured EmbeddingModel
    - retrieve passages by vector similarity
    - return results with scores normalized to [0, 1]
    """

    def index_passages(self, passages: list[dict]) -> None:
        """Index a list of passages for semantic retrieval.

        Each dict must contain at least::
            {"id": str, "text": str, "embedding": list[float] | None}

        Passages without a stored embedding are skipped.
        """
        raise NotImplementedError

    def remove_passages(self, passage_ids: list[str]) -> None:
        """Remove indexed passages by ID."""
        raise NotImplementedError

    def search(self, query: str, top_n: int = 20) -> list[RetrievalResult]:
        """Return the top *top_n* passages ranked by semantic similarity.

        Returns an empty list when the corpus is empty or the query
        produces no matches.
        """
        raise NotImplementedError


class KeywordRetriever:
    """Abstract keyword / lexical retriever.

    Concrete implementations must:
    - index passage text
    - retrieve passages matching a query's terms
    - return results with scores normalized to [0, 1]
    """

    def index_passages(self, passages: list[dict]) -> None:
        """Index a list of passages for keyword retrieval.

        Each dict must contain at least::
            {"id": str, "text": str}
        """
        raise NotImplementedError

    def remove_passages(self, passage_ids: list[str]) -> None:
        """Remove indexed passages by ID."""
        raise NotImplementedError

    def search(self, query: str, top_n: int = 20) -> list[RetrievalResult]:
        """Return the top *top_n* passages ranked by keyword relevance.

        Returns an empty list when the corpus is empty or the query
        matches no passages.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class DocumentScope:
    """Constraint that limits retrieval to a subset of documents.

    Attributes:
        mode: ``"all"`` to search every document, or ``"specific"`` to
            restrict to a single document.
        document_id: The document ID when ``mode == "specific"``.
    """

    mode: str  # "all" | "specific"
    document_id: str | None = None


@dataclass(frozen=True)
class HybridWeighting:
    """Configurable relative weight of semantic vs keyword signals.

    Attributes:
        semantic_weight: Weight for the semantic (embedding) score.
        keyword_weight: Weight for the keyword (BM25) score.
    """

    semantic_weight: float = 0.5
    keyword_weight: float = 0.5
