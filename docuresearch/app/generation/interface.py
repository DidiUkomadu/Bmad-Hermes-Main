"""Model-agnostic LLM interface — Story 4.1.

Defines the LLMInterface protocol and supporting types that make the
generation layer swappable across LLM providers.

The generation layer depends on this interface, not on any specific
provider. Swapping the model is implementing a new LLMInterface or
changing which implementation is instantiated.

Initial implementation uses the Hermes agent free model access as the
LLM backend. The interface is designed to be replaced later without
changing prompt construction or output parsing.

Do NOT implement:
- Epic 4.4 (abstention logic)
- Epic 4.5 (partial/conflicting evidence handling)
- Epic 4.6 (citation generation)
- Epic 5 (conversation)
- Epic 6 (citation UI)
- Epic 7 (API integration)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


@dataclass(frozen=True)
class GeneratedAnswer:
    """Structured answer produced by the generation layer.

    Attributes:
        answer_text: The generated answer text.
        citations: Citations supporting the answer.
            Citation.excerpt is populated by
            ``app.generation.citation.resolve_citations`` from the
            RetrievedPassage text (Story 4.6), not by the LLM.
        evidence_quality: The evidence quality category.
        evidence_quality_narrative: Narrative explaining the evidence quality.
        is_abstention: Whether the system is abstaining from a definitive answer.
        generation_latency_seconds: Wall-clock time for the generation step.
    """

    answer_text: str
    citations: list[Citation]
    evidence_quality: EvidenceQuality
    evidence_quality_narrative: str
    is_abstention: bool
    generation_latency_seconds: float


@dataclass(frozen=True)
class Citation:
    """A citation referencing a retrieved passage.

    Attributes:
        passage_id: The stable passage ID from the content store.
        document_name: Human-readable document name.
        location: Page, section, or paragraph locator.
        excerpt: Actual passage text from the store.
            Populated by ``app.generation.citation.resolve_citations``
            (Story 4.6) from the RetrievedPassage.text in the retrieval
            context. Not set by the LLM. Empty string at generation time.
    """

    passage_id: str
    document_name: str
    location: str
    excerpt: str = ""


class EvidenceQuality(StrEnum):
    """Evidence quality categories from the PRFAQ / architecture.

    Values:
        SUFFICIENT   — documents support the answer directly
        PARTIAL      — documents support part of the answer
        INSUFFICIENT — documents do not contain enough to answer
        CONFLICTING  — documents disagree on the point
    """

    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check whether *value* is a recognised evidence quality category."""
        try:
            cls(value)
            return True
        except ValueError:
            return False


@dataclass
class LLMResponse:
    """Response from an LLMInterface.generate() call.

    Attributes:
        raw_text: The LLM's raw output text.
        structured: Parsed structured output, or None if parsing failed.
        parse_error: Error description if parsing failed, None otherwise.
        latency_seconds: Wall-clock time for the LLM call.
    """

    raw_text: str
    structured: GeneratedAnswer | None = None
    parse_error: str | None = None
    latency_seconds: float = 0.0


class LLMInterface(Protocol):
    """Model-agnostic interface for LLM-backed answer generation.

    Implementations must:
    - Accept a prompt string and optional generation kwargs
    - Return an LLMResponse with raw_text and optionally structured output
    - Set parse_error when structured parsing fails (structured=None)
    - Record generation latency

    The generation layer (prompt.py, schema parsing) depends on this
    interface, not on any specific provider.
    """

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a response for the given prompt.

        Args:
            prompt: The full prompt to send to the LLM.
            **kwargs: Provider-specific generation parameters
                (e.g. temperature, max_tokens).

        Returns:
            LLMResponse with raw_text and optionally structured output.
        """
        ...


class GenerationConfig:
    """Configuration for prompt construction and generation behaviour.

    Attributes:
        max_passages_in_prompt: Maximum number of retrieved passages to
            include in the prompt. Downstream callers may pass fewer.
        default_evidence_quality: Default evidence quality when the LLM
            does not produce one (used as a fallback only).
        require_json_output: Whether to instruct the LLM to output JSON.
    """

    def __init__(
        self,
        max_passages_in_prompt: int = 10,
        default_evidence_quality: str = EvidenceQuality.INSUFFICIENT,
        require_json_output: bool = True,
    ) -> None:
        if max_passages_in_prompt < 1:
            raise ValueError(
                f"max_passages_in_prompt must be >= 1, got {max_passages_in_prompt}"
            )
        if not EvidenceQuality.is_valid(default_evidence_quality):
            raise ValueError(
                f"default_evidence_quality must be one of {sorted(self._all_values())}, "
                f"got {default_evidence_quality!r}"
            )
        self.max_passages_in_prompt = max_passages_in_prompt
        self.default_evidence_quality = default_evidence_quality
        self.require_json_output = require_json_output

    @staticmethod
    def _all_values() -> set[str]:
        return {
            EvidenceQuality.SUFFICIENT.value,
            EvidenceQuality.PARTIAL.value,
            EvidenceQuality.INSUFFICIENT.value,
            EvidenceQuality.CONFLICTING.value,
        }
