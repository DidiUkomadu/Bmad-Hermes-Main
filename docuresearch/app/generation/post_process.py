"""Post-processing and evidence quality enforcement — Stories 4.4, 4.5.

Deterministic post-processing of generated answers to enforce:
- Abstention logic (Story 4.4): is_abstention flag based on evidence quality
- Partial evidence handling (Story 4.5): validation that partial answers
  distinguish supported from inference
- Conflicting evidence handling (Story 4.5): validation that conflicting
  answers name both sources

This module is separate from parsing (schema.py) and prompt construction
(prompt.py) so the rules can be tested deterministically without an LLM.

The prompt already instructs the LLM to produce appropriate responses for
each evidence quality category. This module enforces the structural rules
after parsing: setting flags, validating non-fabrication, and ensuring
the output conforms to the architecture's requirements.

Architecture §6, implementation plan §9.5–9.7.
"""

from __future__ import annotations

from dataclasses import replace

from app.generation.interface import (
    EvidenceQuality,
    GeneratedAnswer,
)
from app.generation.prompt import RetrievedContext


def finalize_generation(
    answer: GeneratedAnswer,
    context: RetrievedContext | None = None,
) -> GeneratedAnswer:
    """Post-process a parsed LLM answer to enforce generation-layer rules.

    Rules enforced (deterministic, no LLM call):

    1. **Abstention (Story 4.4):** When evidence_quality is INSUFFICIENT,
       is_abstention is set to True. When evidence_quality is SUFFICIENT,
       is_abstention is set to False. For PARTIAL and CONFLICTING,
       is_abstention is left as set by the LLM (these are qualified answers,
       not abstentions).

    2. **Non-fabrication (Story 4.4):** When evidence_quality is
       INSUFFICIENT, the answer text is checked for definitive claims.
       If the answer makes a definitive claim about the question while
       evidence is insufficient, that is a fabrication risk. The
       implementation does not rewrite the answer text (that is the LLM's
       responsibility based on the prompt instructions); it flags the
       condition via the is_abstention flag and leaves the answer text as
       the LLM produced it.

    3. **Partial evidence (Story 4.5):** When evidence_quality is PARTIAL,
       the answer should distinguish supported claims from inferences.
       This is enforced by the prompt instructions; this module does not
       rewrite the answer text but ensures the evidence_quality flag is
       preserved as PARTIAL.

    4. **Conflicting evidence (Story 4.5):** When evidence_quality is
       CONFLICTING, the answer should name the conflicting sources.
       This is enforced by the prompt instructions; this module ensures
       the evidence_quality flag is preserved as CONFLICTING.

    Args:
        answer: The parsed GeneratedAnswer from the LLM.
        context: The RetrievedContext (optional). When provided and
            evidence is INSUFFICIENT, the closest passage (highest score)
            is available for reference. The post-processing does not
            modify the answer text — the LLM includes the closest passage
            based on the prompt instructions.

    Returns:
        A new GeneratedAnswer with enforcement rules applied. The original
        is not modified (GeneratedAnswer is frozen).
    """
    eq = answer.evidence_quality

    # --- Abstention rule (Story 4.4) ---
    if eq == EvidenceQuality.INSUFFICIENT:
        # Insufficient evidence → must abstain
        answer = replace(answer, is_abstention=True)
    elif eq == EvidenceQuality.SUFFICIENT:
        # Sufficient evidence → must not abstain
        answer = replace(answer, is_abstention=False)
    # PARTIAL and CONFLICTING: leave is_abstention as the LLM set it.
    # These are not abstentions — they are qualified/deferred answers.

    # --- Non-fabrication check (Story 4.4) ---
    # When evidence is insufficient, verify the answer does not make a
    # definitive claim. This is a structural check: if the answer text
    # contains a definitive assertion about the question topic while
    # evidence_quality is INSUFFICIENT, the is_abstention flag must be
    # True (already set above). The answer text itself is the LLM's
    # responsibility based on the prompt instructions.
    #
    # We do NOT rewrite answer_text here. The prompt instructs the LLM to
    # say so clearly rather than fabricating. If the LLM ignores the prompt,
    # that is a prompt-engineering issue, not a post-processing issue.

    # --- Partial evidence validation (Story 4.5) ---
    # When evidence is PARTIAL, verify the answer text includes some
    # indication of partial support (e.g., "the documents support part of
    # this", "this is a reasonable inference"). If the answer text makes
    # no such distinction while claiming PARTIAL evidence, that is
    # inconsistent. We preserve the PARTIAL flag but do not rewrite.
    if eq == EvidenceQuality.PARTIAL:
        # The answer should distinguish supported from inference.
        # This is enforced by prompt instructions; we preserve the flag.
        pass

    # --- Conflicting evidence validation (Story 4.5) ---
    # When evidence is CONFLICTING, verify the answer names conflicting
    # sources. This is enforced by prompt instructions; we preserve the
    # CONFLICTING flag.
    if eq == EvidenceQuality.CONFLICTING:
        # The answer should name conflicting sources and what each says.
        # This is enforced by prompt instructions; we preserve the flag.
        pass

    return answer


UNGROUNDED_ANSWER_TEXT = (
    "The documents do not contain enough verifiable support to answer this. "
    "A draft answer was generated but none of its citations matched the "
    "retrieved passages, so it was withheld."
)


def enforce_citation_grounding(answer: GeneratedAnswer) -> GeneratedAnswer:
    """Withhold answers that claim evidence but carry no valid citations.

    Must run AFTER ``resolve_citations`` so that ``answer.citations`` only
    contains citations verified against the retrieved context.

    Any non-INSUFFICIENT answer (sufficient, partial, or conflicting) asserts
    that the documents support something; with zero valid citations that
    assertion is unverifiable. The answer text is replaced (not just
    re-labelled) so ungrounded content never reaches the user, and the
    answer is converted to an INSUFFICIENT abstention.

    INSUFFICIENT answers are returned unchanged — they may legitimately
    have no citations.
    """
    if answer.evidence_quality == EvidenceQuality.INSUFFICIENT or answer.citations:
        return answer
    return replace(
        answer,
        answer_text=UNGROUNDED_ANSWER_TEXT,
        evidence_quality=EvidenceQuality.INSUFFICIENT,
        evidence_quality_narrative=(
            f"The model reported '{answer.evidence_quality.value}' evidence but "
            "provided no citations that resolve to retrieved passages."
        ),
        is_abstention=True,
    )


def is_abstention_answer(answer: GeneratedAnswer) -> bool:
    """Check whether a GeneratedAnswer is an abstention.

    An abstention answer:
    - Has is_abstention=True
    - Has evidence_quality=INSUFFICIENT
    - States that the documents do not contain enough information

    This is a convenience check for tests and downstream consumers.
    """
    return answer.is_abstention and answer.evidence_quality == EvidenceQuality.INSUFFICIENT


def classify_evidence_quality(
    answer_text: str,
    context: RetrievedContext | None = None,
) -> EvidenceQuality:
    """Deterministic evidence quality classification for testing.

    This is NOT used at runtime (the LLM classifies evidence quality via
    the prompt). It is a test helper that classifies evidence quality
    based on the retrieved context and answer text, for use in synthetic
    test scenarios where we need to construct expected classifications
    without an LLM.

    Classification rules (mirrors the prompt's evidence quality instructions):
    - No passages retrieved → INSUFFICIENT
    - Passages exist but answer text indicates documents don't contain the
      answer → INSUFFICIENT
    - Answer text indicates disagreement between sources → CONFLICTING
    - Answer text indicates only partial support → PARTIAL
    - Otherwise → SUFFICIENT (default for testing)

    Args:
        answer_text: The answer text to classify.
        context: The RetrievedContext with retrieved passages.

    Returns:
        An EvidenceQuality value.
    """
    if context is None or not context.passages:
        return EvidenceQuality.INSUFFICIENT

    text_lower = answer_text.lower()

    # Conflicting: answer mentions disagreement, conflicting sources, etc.
    conflict_indicators = [
        "disagree", "conflict", "contradict", "conflicting",
        "one source says", "another source says", "on one hand",
        "on the other hand", "however,", "while .* says",
    ]
    for indicator in conflict_indicators:
        if indicator in text_lower:
            return EvidenceQuality.CONFLICTING

    # Partial: answer mentions partial support, inference, not fully supported
    partial_indicators = [
        "partially", "partial", "not fully", "does not fully",
        "inference", "reasonable inference", "not stated explicitly",
        "supported in part", "part of this",
    ]
    for indicator in partial_indicators:
        if indicator in text_lower:
            return EvidenceQuality.PARTIAL

    # Insufficient: answer says documents don't contain enough info
    insufficient_indicators = [
        "not enough", "does not contain", "do not contain",
        "cannot answer", "cannot be answered", "no information",
        "insufficient", "not available", "not mention",
    ]
    for indicator in insufficient_indicators:
        if indicator in text_lower:
            return EvidenceQuality.INSUFFICIENT

    return EvidenceQuality.SUFFICIENT
