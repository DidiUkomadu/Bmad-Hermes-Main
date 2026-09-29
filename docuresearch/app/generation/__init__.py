"""Generation package — Stories 4.1–4.6.

Model-agnostic LLM-backed answer generation for DocuResearch.

Layers:
- interface.py  — LLMInterface protocol, GeneratedAnswer, Citation, EvidenceQuality, LLMResponse
- prompt.py     — build_prompt(context, config) → str (8-section prompt construction)
- schema.py     — parse_llm_output(raw_text) → GeneratedAnswer | None (structured output parsing)
- post_process.py — finalize_generation(answer, context) → GeneratedAnswer (abstention + evidence enforcement)
- citation.py   — resolve_citations(answer, context) → GeneratedAnswer (citation validation + resolution)

Stories implemented:
- 4.1: Model-agnostic LLM interface
- 4.2: Prompt construction (8-section structure)
- 4.3: Structured output schema and parsing
- 4.4: Abstention logic (post-processing enforcement)
- 4.5: Partial and conflicting evidence handling (post-processing enforcement)
- 4.6: Citation generation (validation against retrieved context, resolution from trusted source)

Deferred:
- Epic 5: Conversation
- Epic 6: Citation UI
- Epic 7: API integration
"""

from __future__ import annotations

from app.generation.citation import (
    citation_passage_ids,
    has_valid_citations,
    resolve_citations,
)
from app.generation.interface import (
    Citation,
    EvidenceQuality,
    GeneratedAnswer,
    GenerationConfig,
    LLMInterface,
    LLMResponse,
)
from app.generation.post_process import (
    classify_evidence_quality,
    finalize_generation,
    is_abstention_answer,
)

__all__ = [
    # interface.py
    "Citation",
    "EvidenceQuality",
    "GeneratedAnswer",
    "GenerationConfig",
    "LLMInterface",
    "LLMResponse",
    # post_process.py
    "classify_evidence_quality",
    "finalize_generation",
    "is_abstention_answer",
    # citation.py
    "citation_passage_ids",
    "has_valid_citations",
    "resolve_citations",
]
