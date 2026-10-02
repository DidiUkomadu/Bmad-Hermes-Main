"""Prompt construction — Story 4.2.

Builds the full prompt for the LLM from a RetrievedContext and GenerationConfig.

The prompt follows the 8-section structure resolved in Epics & Stories
story 4.2:

1. Role and task
2. The user's question
3. Retrieved passages (source material)
4. Document scope context
5. Conversation context (if any)
6. Answer instructions (5 requirements from Architecture §6)
7. Evidence quality instructions (4 categories + conservative heuristics)
8. Output format instructions (structured JSON schema)

The prompt is a pure function of RetrievedContext and GenerationConfig.
It does not call the LLM. This makes it inspectable and testable.

Separation: build_prompt produces the prompt string; the LLM call is a
separate step performed by the LLMInterface (Story 4.1).

Do NOT implement:
- Epic 4.4 (abstention logic) — deferred
- Epic 4.5 (partial/conflicting evidence handling) — deferred
- Epic 5 (conversation) — deferred
- Epic 6 (citation UI) — deferred
- Epic 7 (API integration) — deferred
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.generation.interface import Citation, EvidenceQuality, GenerationConfig

# ---------------------------------------------------------------------------
# Boundary types (retrieval → generation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversationTurn:
    """A single turn in a conversation session.

    Present in RetrievedContext.conversation_history when the query is
    part of an ongoing conversation. Persisted and loaded by
    ``app.conversation.session`` (Story 5.1).
    """

    turn_index: int
    user_query: str
    answer_text: str
    citations: list[Citation]
    evidence_quality: EvidenceQuality
    is_abstention: bool
    created_at: datetime
    evidence_quality_narrative: str = ""


@dataclass(frozen=True)
class RetrievedPassage:
    """A single retrieved passage presented to the generation layer.

    Carries everything the generation layer needs to cite the passage
    and to include it in the prompt as source material.
    """

    passage_id: str
    document_name: str
    location: str
    text: str
    score: float


@dataclass(frozen=True)
class RetrievedContext:
    """The complete retrieval result delivered to the generation layer.

    This is the boundary type between retrieval and generation (architecture
    §2.5). Generation does not call the store or retrieval directly — it
    receives this from the caller (API layer or evaluation runner).

    Attributes:
        query: The user's question text.
        scope: Which documents were searched (all or specific).
        passages: Ranked retrieved passages with metadata.
        conversation_history: Previous conversation turns, if any.
        retrieval_latency_seconds: Wall-clock time for retrieval.
    """

    query: str
    scope: dict[str, Any]
    passages: list[RetrievedPassage]
    conversation_history: list[ConversationTurn] | None = None
    retrieval_latency_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


# Answer instructions (Architecture §6, items 1–5)
_ANSWER_INSTRUCTIONS = """\
Answer instructions:
1. Answer the question using ONLY the retrieved passages as source material.
   Do not introduce outside knowledge as if it came from the documents.
2. Cite specific passages for each claim. Use the document name and location
   from the passage metadata in your citations.
3. Assess and narrate evidence quality: does the retrieved evidence fully
   support the answer, partially support it, not support it, or contain
   conflicting information?
4. If the evidence is insufficient to answer definitively, say so clearly
   rather than generating a plausible answer from your prior knowledge.
5. Stay within the uploaded documents. Do not present information as coming
   from the documents if it is not present in the retrieved passages."""

# Evidence quality instructions (PRFAQ §2, Architecture §6)
_EVIDENCE_QUALITY_INSTRUCTIONS = """\
Evidence quality assessment:
- "sufficient": The retrieved passages directly and clearly support the answer.
  All key claims are backed by specific passages.
- "partial": The retrieved passages support part of the answer but not all.
  Distinguish the part supported by the documents from any inference.
  Label inferences clearly as inferences, not as statements from the documents.
- "insufficient": The retrieved passages do not contain enough information
  to answer the question definitively. Say so. Point to the closest relevant
  passage if one exists. Do not fabricate a definitive answer.
- "conflicting": The retrieved passages from different sources disagree on
  the point. Name the conflicting sources and what each says. Do not silently
  choose one side.

When the boundary between categories is unclear, err toward "insufficient"
rather than overclaiming."""

# Output format instructions (JSON schema for structured output)
def _output_format_instructions(require_json: bool) -> str:
    if require_json:
        return """\
Output format:
Produce your response as a single JSON object with the following structure:

{
    "answer_text": "The answer to the user's question, grounded in the retrieved passages.",
    "citations": [
        {
            "passage_id": "The passage ID from the retrieved passages (use exactly as given).",
            "document_name": "The document name from the passage metadata.",
            "location": "The location from the passage metadata."
        }
    ],
    "evidence_quality": "sufficient | partial | insufficient | conflicting",
    "evidence_quality_narrative": "A brief narrative explaining the evidence quality assessment.",
    "is_abstention": false
}

Rules for citations:
- Each citation must reference a passage_id that exists in the retrieved passages above.
- Do not fabricate citation entries for passages not in the retrieved set.
- citation.excerpt is NOT your responsibility — the system populates it from the store.
- Include a citation for each claim or claim-group that is supported by a retrieved passage.
- If no claim is supported by the retrieved passages, the citations array may be empty.

Rules for evidence quality:
- Use exactly one of: "sufficient", "partial", "insufficient", "conflicting".
- is_abstention must be true when evidence_quality is "insufficient" and the answer
  does not make a definitive claim about the question."""

    return ""


def build_prompt(
    context: RetrievedContext,
    config: GenerationConfig | None = None,
) -> str:
    """Build the full prompt for the LLM from a RetrievedContext.

    Args:
        context: The retrieval result including the query, retrieved passages,
            document scope, and optional conversation history.
        config: Generation configuration. If None, a default config is used.

    Returns:
        A prompt string with all 8 sections assembled.
    """
    if config is None:
        config = GenerationConfig()

    sections: list[str] = []

    # Section 1: Role and task
    sections.append(_section_1_role_and_task())

    # Section 2: The user's question
    sections.append(_section_2_question(context.query))

    # Section 3: Retrieved passages (source material)
    sections.append(_section_3_passages(context.passages, config.max_passages_in_prompt))

    # Section 4: Document scope context
    sections.append(_section_4_scope(context.scope))

    # Section 5: Conversation context (if any)
    sections.append(_section_5_conversation(context.conversation_history))

    # Section 6: Answer instructions
    sections.append(_ANSWER_INSTRUCTIONS)

    # Section 7: Evidence quality instructions
    sections.append(_EVIDENCE_QUALITY_INSTRUCTIONS)

    # Section 8: Output format instructions
    sections.append(_output_format_instructions(config.require_json_output))

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _section_1_role_and_task() -> str:
    return """\
## Role and Task

You are a research assistant for DocuResearch. Your task is to answer the
user's question using ONLY the document passages provided below as source
material. You must:

- Ground your answer in the retrieved passages, not in your prior knowledge.
- Cite specific passages (by document name and location) for each claim.
- Assess and narrate the evidence quality.
- Abstain or qualify when evidence is insufficient.
- Stay within the uploaded documents."""


def _section_2_question(query: str) -> str:
    return f"""\
## User's Question

{query}"""


def _section_3_passages(
    passages: list[RetrievedPassage],
    max_passages: int,
) -> str:
    if not passages:
        return """\
## Retrieved Passages

No passages were retrieved for this query. The documents do not contain
relevant information for this question."""

    limited = passages[:max_passages]
    parts = [
        "## Retrieved Passages\n\n"
        "The following passages were retrieved as potentially relevant source material. "
        "Each passage includes its document name, location, and text. "
        "Use these as your ONLY source material for answering the question.",
    ]
    for i, p in enumerate(limited, 1):
        parts.append(
            f"### Passage {i} (score: {p.score:.3f})\n"
            f"- **Document:** {p.document_name}\n"
            f"- **Location:** {p.location}\n"
            f"- **Passage ID:** {p.passage_id}\n"
            f"- **Text:**\n{p.text}\n"
        )
    return "\n".join(parts)


def _section_4_scope(scope: dict[str, Any]) -> str:
    mode = scope.get("mode", "all")
    doc_id = scope.get("document_id")

    if mode == "specific" and doc_id:
        return f"""\
## Document Scope

The search was restricted to the document with ID: {doc_id}.
Only passages from this document were retrieved and are presented above."""

    return """\
## Document Scope

The search spanned all uploaded documents. Passages from any document
in the corpus may appear above."""


def _section_5_conversation(
    history: list[ConversationTurn] | None,
) -> str:
    if not history:
        return """\
## Conversation Context

This is the first turn in the conversation. No prior context is available."""

    parts = [
        "## Conversation Context\n\n"
        "This question is part of an ongoing conversation. "
        "Previous turns are provided for context:"
    ]
    for turn in history:
        sources = "; ".join(
            f"{c.document_name}, {c.location} (Passage ID: {c.passage_id})" for c in turn.citations
        )
        parts.append(
            f"\n--- Turn {turn.turn_index} ---\n"
            f"User: {turn.user_query}\n"
            f"Assistant: {turn.answer_text}\n"
            f"(Evidence quality: {turn.evidence_quality}; Abstention: {turn.is_abstention})\n"
            f"Sources cited: {sources or 'none'}\n"
        )
    parts.append(
        "\nThe user's question may refer back to these turns (for example \"it\" or "
        "\"that\"); interpret it in light of the conversation. Passages cited by earlier "
        "answers are included among the retrieved passages above when still available.\n"
        "Use the conversation history to maintain consistency with prior answers. "
        "Do not contradict earlier answers unless the retrieved passages clearly support "
        "a different conclusion.\n"
        "The conversation history is context only, not a source. Every claim in your "
        "answer must be supported by, and cite, the retrieved passages above; do not "
        "cite or repeat earlier answers as evidence."
    )
    return "\n".join(parts)
