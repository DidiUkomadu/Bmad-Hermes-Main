"""Citation generation and resolution — Story 4.6.

Validates and resolves citations in a GeneratedAnswer against the retrieved
context (RetrievedPassage objects from the retrieval layer).

Responsibilities:
- Validate that every citation's passage_id exists in the retrieved context.
- Reject/ignore citations for unknown passage IDs (not in retrieved context).
|- Resolve document_name, location, and excerpt from the trusted RetrievedPassage,
  ignoring any LLM-supplied values for these fields.
|- Each citation's excerpt is the actual passage text from the RetrievedPassage
  (Epics & Stories §4.3, §4.6), not a model-generated paraphrase.
|- Preserve answer_text and evidence_quality unchanged.
|- Deduplicate citations by passage_id (keep first occurrence).
|- Handle malformed citations safely (skip invalid entries, keep valid ones).

Does NOT:
- Rewrite answer_text (Story 4.6 is about citation validity, not answer editing).
- Modify evidence_quality or evidence_quality_narrative.
- Populate excerpt from the store (that is the citation resolver's job,
  Story 4.6 resolver / Epic 6 / app.citation.resolver).
- Implement citation UI (Epic 6).

Architecture §2.6, §9.8; Epics & Stories story 4.6.
"""

from __future__ import annotations

from collections import OrderedDict

from app.generation.interface import (
    Citation,
    GeneratedAnswer,
)
from app.generation.prompt import (
    RetrievedContext,
    RetrievedPassage,
)


def resolve_citations(
    answer: GeneratedAnswer,
    context: RetrievedContext | None = None,
) -> GeneratedAnswer:
    """Validate and resolve citations against the retrieved context.

    For each citation in the answer:
    - If the citation's passage_id exists in the retrieved context,
      resolve document_name and location from the trusted RetrievedPassage.
      The LLM-supplied document_name and location are ignored in favour of
      the trusted values from the retrieval layer.
    - If the citation's passage_id does NOT exist in the retrieved context,
      the citation is dropped (invalid reference, not supported by retrieved
      passages).
    - If context is None or has no passages, all citations are dropped
      (no trusted source to validate against).

    Citations are deduplicated by passage_id (first occurrence kept).

    The answer_text, evidence_quality, evidence_quality_narrative, and
    is_abstention fields are preserved unchanged.

    Excerpt is resolved from the RetrievedPassage text (the actual passage text
    from the retrieval layer), satisfying Epics & Stories §4.3 and §4.6.

    Args:
        answer: The GeneratedAnswer whose citations should be validated.
        context: The RetrievedContext from the retrieval layer. When None
            or passages is empty, all citations are dropped.

    Returns:
        A new GeneratedAnswer with validated/resolved citations.
        The original is not modified (GeneratedAnswer is frozen).
    """
    if context is None or not context.passages:
        # No trusted source — drop all citations
        resolved = []
    else:
        # Build lookup: passage_id -> RetrievedPassage
        passage_map = _build_passage_map(context.passages)

        resolved = []
        seen_ids: set[str] = set()
        for cit in answer.citations:
            pid = cit.passage_id
            if pid in passage_map:
                # Resolve from trusted source, ignore LLM-supplied values
                trusted = passage_map[pid]
                resolved_cit = Citation(
                    passage_id=pid,
                    document_name=trusted.document_name,
                    location=trusted.location,
                    excerpt=trusted.text,  # actual passage text from trusted source
                )
                if pid not in seen_ids:
                    resolved.append(resolved_cit)
                    seen_ids.add(pid)
            # else: unknown passage_id — drop the citation

    return GeneratedAnswer(
        answer_text=answer.answer_text,
        citations=resolved,
        evidence_quality=answer.evidence_quality,
        evidence_quality_narrative=answer.evidence_quality_narrative,
        is_abstention=answer.is_abstention,
        generation_latency_seconds=answer.generation_latency_seconds,
    )


def _build_passage_map(passages: list[RetrievedPassage]) -> dict[str, RetrievedPassage]:
    """Build a passage_id -> RetrievedPassage lookup from a passage list.

    If duplicate passage_ids exist, the last occurrence wins (consistent
    with how the retrieval layer returns results).
    """
    return OrderedDict((p.passage_id, p) for p in passages)


def citation_passage_ids(answer: GeneratedAnswer) -> list[str]:
    """Return the ordered list of passage_ids from a GeneratedAnswer's citations.

    Used by tests and the evaluation runner to check which passages are cited.
    """
    return [c.passage_id for c in answer.citations]


def has_valid_citations(answer: GeneratedAnswer) -> bool:
    """Check whether a GeneratedAnswer has at least one citation.

    Convenience check for tests.
    """
    return len(answer.citations) > 0
