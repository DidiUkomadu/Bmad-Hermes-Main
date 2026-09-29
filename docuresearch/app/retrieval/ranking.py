"""Final ranking and candidate set — Story 3.5.

Applies the final ranking, candidate limit, and latency measurement to
retrieval results. This layer is separate from the semantic, keyword, and
hybrid retrievers so it can be reused by later generation/API layers.

Ranking rules:
- Sort by score descending.
- Tie-break deterministically by passage_id ascending.
- Truncate to max_candidates (default 20).
- Measure and record retrieval latency.

Empty input returns an empty list safely.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.retrieval.interface import RetrievalResult


@dataclass
class RankedResult:
    """A retrieval result with ranking metadata.

    Attributes:
        results: The ranked list of RetrievalResult objects.
        max_candidates: The maximum candidate set size that was enforced.
        retrieval_latency_seconds: Wall-clock time for the retrieval operation.
    """

    results: list[RetrievalResult]
    max_candidates: int
    retrieval_latency_seconds: float


class RankingLayer:
    """Final ranking layer for retrieval results.

    Takes a list of RetrievalResult objects (from any retrieval source),
    applies deterministic ranking, enforces the candidate set size limit,
    and measures latency.

    The ranking is stable and deterministic: equal scores are broken by
    passage_id ascending, ensuring reproducible results across runs.

    Args:
        max_candidates: Maximum number of candidates to return (default 20).
    """

    def __init__(self, max_candidates: int = 20) -> None:
        if max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {max_candidates}")
        self._max_candidates = max_candidates

    def rank(
        self, results: list[RetrievalResult], latency_seconds: float | None = None
    ) -> RankedResult:
        """Rank, deduplicate-safe (caller must dedup before calling), and limit.

        Args:
            results: Retrieval results to rank. Should already be deduplicated
                by passage_id (the hybrid retriever handles this).
            latency_seconds: Optional pre-measured latency. If None, the
                ranking operation itself is timed (excluding caller's retrieval
                time). For end-to-end latency, pass the retrieval latency from
                the caller.

        Returns:
            RankedResult with the top max_candidates results.
        """
        if latency_seconds is None:
            start = time.monotonic()
            # Ranking is fast; measure just the sort+slice
            ranked = self._rank_results(results)
            elapsed = time.monotonic() - start
        else:
            ranked = self._rank_results(results)
            elapsed = latency_seconds

        return RankedResult(
            results=ranked,
            max_candidates=self._max_candidates,
            retrieval_latency_seconds=elapsed,
        )

    def rank_with_latency(
        self, results: list[RetrievalResult]
    ) -> RankedResult:
        """Rank results and measure the ranking operation latency.

        Convenience method that times the ranking operation.
        """
        return self.rank(results)

    def _rank_results(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Sort and truncate results.

        Sorting: score descending, passage_id ascending for ties.
        Truncation: keep at most max_candidates results.
        """
        # Sort: score desc, then passage_id asc for deterministic ties
        sorted_results = sorted(
            results,
            key=lambda r: (-r.score, r.passage_id),
        )
        return sorted_results[: self._max_candidates]

    @property
    def max_candidates(self) -> int:
        """The configured maximum candidate set size."""
        return self._max_candidates

    def configure(self, max_candidates: int) -> None:
        """Update the maximum candidate set size."""
        if max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {max_candidates}")
        self._max_candidates = max_candidates
