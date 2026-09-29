"""Tests for Generation layer — Stories 4.1, 4.2, 4.3.

Tests use synthetic/constructed RetrievedContext objects rather than
requiring a live LLM or database. They validate:

Story 4.1:
- LLMInterface protocol existence and structure
- LLMResponse model fields
- GeneratedAnswer model fields
- Citation model fields (excerpt empty at generation time)
- EvidenceQuality enumeration values
- GenerationConfig validation

Story 4.2:
- build_prompt produces all 8 sections
- Prompt contains all 8 sections with correct content
- Prompt does not instruct the model to introduce outside knowledge
- Prompt works with and without conversation context
- Prompt handles empty passages
- Prompt handles specific vs all document scope

Story 4.3:
- parse_llm_output correctly parses valid JSON LLM output
- parse_llm_output correctly parses markdown-fenced JSON
- parse_llm_output handles raw (non-fenced) JSON
- parse_llm_output returns None for malformed JSON
- parse_llm_output returns None for missing fields
- parse_llm_output returns None for invalid evidence_quality
- Citation.excerpt is empty string after parsing (not populated here)
- parse_llm_output handles is_abstention correctly

Do NOT test:
- Epic 4.4 (abstention logic) — deferred
- Epic 4.5 (partial/conflicting evidence handling) — deferred
- Epic 4.6 (citation generation) — deferred
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

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
from app.generation.prompt import (
    ConversationTurn,
    RetrievedContext,
    RetrievedPassage,
    build_prompt,
)
from app.generation.schema import parse_llm_output

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_passage() -> RetrievedPassage:
    """A single sample retrieved passage."""
    return RetrievedPassage(
        passage_id="doc-abc123:page:5:chunk:0",
        document_name="Security Standards Manual",
        location="Page 5, Section 3.2",
        text="All encryption keys must be rotated every 90 days. "
        "Keys older than 90 days must be retired and replaced.",
        score=0.95,
    )


@pytest.fixture
def sample_passages(sample_passage: RetrievedPassage) -> list[RetrievedPassage]:
    """Multiple sample retrieved passages."""
    p2 = RetrievedPassage(
        passage_id="doc-abc123:page:8:chunk:1",
        document_name="Security Standards Manual",
        location="Page 8, Section 4.1",
        text="Two-factor authentication is required for all administrative access.",
        score=0.88,
    )
    p3 = RetrievedPassage(
        passage_id="doc-xyz789:page:2:chunk:0",
        document_name="Access Control Policy",
        location="Page 2, Paragraph 3",
        text="Administrative accounts must use hardware tokens for authentication.",
        score=0.82,
    )
    return [sample_passage, p2, p3]


@pytest.fixture
def all_documents_context(
    sample_passage: RetrievedPassage,
) -> RetrievedContext:
    """A RetrievedContext with all-documents scope."""
    return RetrievedContext(
        query="How often should encryption keys be rotated?",
        scope={"mode": "all", "document_id": None},
        passages=[sample_passage],
        conversation_history=None,
        retrieval_latency_seconds=0.42,
    )


@pytest.fixture
def specific_doc_context(
    sample_passage: RetrievedPassage,
) -> RetrievedContext:
    """A RetrievedContext with specific-document scope."""
    return RetrievedContext(
        query="How often should encryption keys be rotated?",
        scope={"mode": "specific", "document_id": "doc-abc123"},
        passages=[sample_passage],
        conversation_history=None,
        retrieval_latency_seconds=0.31,
    )


@pytest.fixture
def conversation_context(
    sample_passages: list[RetrievedPassage],
) -> RetrievedContext:
    """A RetrievedContext with conversation history."""
    history = [
        ConversationTurn(
            turn_index=0,
            user_query="What is the document about?",
            answer_text="This document covers security standards including encryption and access control.",
            citations=[
                {"passage_id": "doc-abc123:page:1:chunk:0", "document_name": "Security Standards Manual", "location": "Page 1"}
            ],
            evidence_quality="sufficient",
            is_abstention=False,
            created_at="2026-09-21T10:00:00",
        ),
    ]
    return RetrievedContext(
        query="How often should encryption keys be rotated?",
        scope={"mode": "all", "document_id": None},
        passages=sample_passages,
        conversation_history=history,
        retrieval_latency_seconds=0.55,
    )


@pytest.fixture
def empty_context() -> RetrievedContext:
    """A RetrievedContext with no retrieved passages."""
    return RetrievedContext(
        query="What is the capital of France?",
        scope={"mode": "all", "document_id": None},
        passages=[],
        conversation_history=None,
        retrieval_latency_seconds=0.05,
    )


@pytest.fixture
def default_config() -> GenerationConfig:
    """Default generation config."""
    return GenerationConfig()


@pytest.fixture
def custom_config() -> GenerationConfig:
    """A custom generation config."""
    return GenerationConfig(
        max_passages_in_prompt=5,
        default_evidence_quality=EvidenceQuality.INSUFFICIENT,
        require_json_output=True,
    )


# ============================================================================
# Story 4.1 — Model-agnostic LLM interface
# ============================================================================


class TestEvidenceQualityEnumeration:
    """Story 4.1 AC: EvidenceQuality is an enumeration with four values."""

    def test_has_four_values(self) -> None:
        assert EvidenceQuality.SUFFICIENT == "sufficient"
        assert EvidenceQuality.PARTIAL == "partial"
        assert EvidenceQuality.INSUFFICIENT == "insufficient"
        assert EvidenceQuality.CONFLICTING == "conflicting"

    def test_is_valid_recognises_all_four(self) -> None:
        assert EvidenceQuality.is_valid("sufficient") is True
        assert EvidenceQuality.is_valid("partial") is True
        assert EvidenceQuality.is_valid("insufficient") is True
        assert EvidenceQuality.is_valid("conflicting") is True

    def test_is_valid_rejects_unknown(self) -> None:
        assert EvidenceQuality.is_valid("unknown") is False
        assert EvidenceQuality.is_valid("") is False
        assert EvidenceQuality.is_valid("SUFFICIENT") is False  # case-sensitive

    def test_values_are_strings(self) -> None:
        assert isinstance(EvidenceQuality.SUFFICIENT, str)
        assert isinstance(EvidenceQuality.PARTIAL, str)


class TestGenerationConfig:
    """Story 4.1 AC: GenerationConfig exists with sensible defaults."""

    def test_defaults(self, default_config: GenerationConfig) -> None:
        assert default_config.max_passages_in_prompt == 10
        assert default_config.default_evidence_quality == EvidenceQuality.INSUFFICIENT
        assert default_config.require_json_output is True

    def test_custom_values(self, custom_config: GenerationConfig) -> None:
        assert custom_config.max_passages_in_prompt == 5
        assert custom_config.default_evidence_quality == EvidenceQuality.INSUFFICIENT
        assert custom_config.require_json_output is True

    def test_invalid_max_passages_raises(self) -> None:
        with pytest.raises(ValueError, match="max_passages_in_prompt"):
            GenerationConfig(max_passages_in_prompt=0)

    def test_invalid_evidence_quality_raises(self) -> None:
        with pytest.raises(ValueError, match="default_evidence_quality"):
            GenerationConfig(default_evidence_quality="invalid")


class TestLLMResponse:
    """Story 4.1 AC: LLMResponse model with structured field and parse_error."""

    def test_fields(self) -> None:
        resp = LLMResponse(
            raw_text='{"answer_text": "test", "citations": [], "evidence_quality": "sufficient", "evidence_quality_narrative": "ok", "is_abstention": false}',
            structured=None,
            parse_error="malformed",
            latency_seconds=1.5,
        )
        assert resp.raw_text.startswith('{"answer_text"')
        assert resp.structured is None
        assert resp.parse_error == "malformed"
        assert resp.latency_seconds == 1.5

    def test_structured_populated(self) -> None:
        answer = GeneratedAnswer(
            answer_text="test answer",
            citations=[],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="direct support",
            is_abstention=False,
            generation_latency_seconds=0.0,
        )
        resp = LLMResponse(raw_text="{}", structured=answer, latency_seconds=0.5)
        assert resp.structured is answer
        assert resp.parse_error is None


class TestGeneratedAnswer:
    """Story 4.1 AC: GeneratedAnswer model with all required fields."""

    def test_fields(self) -> None:
        cit = Citation(
            passage_id="doc-1:page:1:chunk:0",
            document_name="Doc A",
            location="Page 1",
        )
        answer = GeneratedAnswer(
            answer_text="The answer text.",
            citations=[cit],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="Directly supported.",
            is_abstention=False,
            generation_latency_seconds=1.23,
        )
        assert answer.answer_text == "The answer text."
        assert len(answer.citations) == 1
        assert answer.evidence_quality == EvidenceQuality.SUFFICIENT
        assert answer.evidence_quality_narrative == "Directly supported."
        assert answer.is_abstention is False
        assert answer.generation_latency_seconds == 1.23

    def test_empty_citations(self) -> None:
        answer = GeneratedAnswer(
            answer_text="No citations.",
            citations=[],
            evidence_quality=EvidenceQuality.INSUFFICIENT,
            evidence_quality_narrative="No evidence.",
            is_abstention=True,
            generation_latency_seconds=0.0,
        )
        assert answer.citations == []

    def test_immutable(self) -> None:
        answer = GeneratedAnswer(
            answer_text="x",
            citations=[],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="n",
            is_abstention=False,
            generation_latency_seconds=0.0,
        )
        with pytest.raises(Exception):
            answer.answer_text = "y"


class TestCitation:
    """Story 4.1 AC: Citation model with excerpt NOT populated at generation time."""

    def test_fields(self) -> None:
        cit = Citation(
            passage_id="doc-1:page:1:chunk:0",
            document_name="Security Manual",
            location="Page 5, Section 2",
        )
        assert cit.passage_id == "doc-1:page:1:chunk:0"
        assert cit.document_name == "Security Manual"
        assert cit.location == "Page 5, Section 2"
        assert cit.excerpt == ""  # NOT populated here — resolver does that

    def test_excerpt_default_empty(self) -> None:
        """Citation.excerpt is empty by default — the resolver populates it."""
        cit = Citation(
            passage_id="p1",
            document_name="Doc",
            location="L1",
        )
        assert cit.excerpt == ""

    def test_frozen(self) -> None:
        cit = Citation(passage_id="p1", document_name="D", location="L")
        with pytest.raises(Exception):
            cit.passage_id = "p2"


class TestLLMInterfaceProtocol:
    """Story 4.1 AC: LLMInterface protocol exists and describes the contract."""

    def test_protocol_exists(self) -> None:
        """The LLMInterface protocol must exist and have a generate method."""
        assert hasattr(LLMInterface, "generate")

    def test_protocol_is_protocol(self) -> None:
        from typing import Protocol

        assert issubclass(LLMInterface, Protocol)

    def test_concrete_implementation(self) -> None:
        """A concrete class implementing LLMInterface should satisfy isinstance checks
        when type-checked (runtime: Protocol is structural)."""

        class MockLLM:
            def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
                return LLMResponse(raw_text="{}", latency_seconds=0.1)

        mock = MockLLM()
        # Structural subtyping: a class with the right method satisfies the protocol
        # at runtime we can't check isinstance against a Protocol without @runtime_checkable,
        # but we can verify the method exists and has the right signature shape.
        assert hasattr(mock, "generate")
        resp = mock.generate("test prompt")
        assert isinstance(resp, LLMResponse)
        assert resp.raw_text == "{}"


# ============================================================================
# Story 4.2 — Prompt construction
# ============================================================================


_SECTION_MARKERS = {
    "role_and_task": r"## Role and Task|Role and Task|role and task",
    "user_question": r"## User's Question|User's Question|user's question",
    "passages": r"## Retrieved Passages|Retrieved Passages|retrieved passages",
    "scope": r"## Document Scope|Document Scope|document scope",
    "conversation": r"## Conversation Context|Conversation Context|conversation context",
    "answer_instructions": r"Answer instructions:",
    "evidence_quality": r"Evidence quality assessment:",
    "output_format": r"Output format:",
}


def _detect_section_names(prompt: str) -> set[str]:
    """Detect which sections are present in a prompt by their markers."""
    found: set[str] = set()
    for name, pattern in _SECTION_MARKERS.items():
        if re.search(pattern, prompt, re.IGNORECASE):
            found.add(name)
    return found


class TestBuildPromptSections:
    """Story 4.2 AC: prompt contains all 8 sections."""

    def test_all_eight_sections_present(
        self, all_documents_context: RetrievedContext, default_config: GenerationConfig
    ) -> None:
        prompt = build_prompt(all_documents_context, default_config)
        found = _detect_section_names(prompt)
        assert "role_and_task" in found, f"Missing role_and_task. Prompt head: {prompt[:200]}"
        assert "user_question" in found, f"Missing user_question. Prompt head: {prompt[:200]}"
        assert "passages" in found, f"Missing passages. Prompt head: {prompt[:200]}"
        assert "scope" in found, f"Missing scope. Prompt head: {prompt[:200]}"
        assert "conversation" in found, f"Missing conversation. Prompt head: {prompt[:200]}"
        assert "answer_instructions" in found, f"Missing answer_instructions. Prompt head: {prompt[:200]}"
        assert "evidence_quality" in found, f"Missing evidence_quality. Prompt head: {prompt[:200]}"
        assert "output_format" in found, f"Missing output_format. Prompt head: {prompt[:200]}"

    def test_query_in_prompt(
        self, all_documents_context: RetrievedContext
    ) -> None:
        prompt = build_prompt(all_documents_context)
        assert all_documents_context.query in prompt

    def test_passage_text_in_prompt(
        self, all_documents_context: RetrievedContext
    ) -> None:
        prompt = build_prompt(all_documents_context)
        for p in all_documents_context.passages:
            assert p.text in prompt, "Passage text not found in prompt"
            assert p.document_name in prompt
            assert p.location in prompt
            assert p.passage_id in prompt

    def test_document_scope_in_prompt(
        self, specific_doc_context: RetrievedContext
    ) -> None:
        prompt = build_prompt(specific_doc_context)
        assert specific_doc_context.scope["document_id"] in prompt
        assert "restricted to the document" in prompt.lower()

    def test_all_documents_scope_in_prompt(
        self, all_documents_context: RetrievedContext
    ) -> None:
        prompt = build_prompt(all_documents_context)
        assert "all uploaded documents" in prompt.lower()

    def test_prompt_does_not_instruct_outside_knowledge(
        self, all_documents_context: RetrievedContext
    ) -> None:
        """Story 4.2 AC: prompt does NOT instruct the model to introduce outside knowledge."""
        prompt = build_prompt(all_documents_context)
        # The prompt should explicitly say NOT to use outside knowledge
        assert "ONLY" in prompt or "only" in prompt
        assert "outside knowledge" in prompt.lower()
        assert "prior knowledge" in prompt.lower() or "your prior knowledge" in prompt.lower()
        # The prompt should NOT have instructions like "use your knowledge"
        assert "use your knowledge" not in prompt.lower()
        assert "bring in outside" not in prompt.lower()

    def test_with_conversation_context(
        self, conversation_context: RetrievedContext
    ) -> None:
        prompt = build_prompt(conversation_context)
        assert "conversation context" in prompt.lower()
        assert "ongoing conversation" in prompt.lower()
        # Previous turn content should be in the prompt
        hist = conversation_context.conversation_history
        assert hist is not None
        assert hist[0].user_query in prompt
        assert hist[0].answer_text in prompt

    def test_without_conversation_context(
        self, all_documents_context: RetrievedContext
    ) -> None:
        prompt = build_prompt(all_documents_context)
        assert "first turn" in prompt.lower()
        assert "no prior context" in prompt.lower()

    def test_empty_passages_handled(self, empty_context: RetrievedContext) -> None:
        prompt = build_prompt(empty_context)
        assert "no passages were retrieved" in prompt.lower()
        # Should still have the other sections
        found = _detect_section_names(prompt)
        assert "role_and_task" in found
        assert "user_question" in found
        assert "answer_instructions" in found


class TestBuildPromptPassageLimiting:
    """Story 4.2 AC: prompt respects max_passages_in_prompt config."""

    def test_more_passages_than_limit(self, sample_passages: list[RetrievedPassage]) -> None:
        config = GenerationConfig(max_passages_in_prompt=2)
        context = RetrievedContext(
            query="test",
            scope={"mode": "all", "document_id": None},
            passages=sample_passages,
        )
        prompt = build_prompt(context, config)
        # Count how many "Passage N" headers appear
        headers = re.findall(r"### Passage (\d+)", prompt)
        assert len(headers) <= 2, f"Expected at most 2 passages in prompt, found {len(headers)}"

    def test_exactly_limit(self, sample_passages: list[RetrievedPassage]) -> None:
        config = GenerationConfig(max_passages_in_prompt=3)
        context = RetrievedContext(
            query="test",
            scope={"mode": "all", "document_id": None},
            passages=sample_passages,  # 3 passages
        )
        prompt = build_prompt(context, config)
        headers = re.findall(r"### Passage (\d+)", prompt)
        assert len(headers) == 3


class TestBuildPromptDeterministic:
    """Story 4.2 AC: prompt construction is deterministic."""

    def test_same_context_same_prompt(
        self, all_documents_context: RetrievedContext
    ) -> None:
        p1 = build_prompt(all_documents_context)
        p2 = build_prompt(all_documents_context)
        assert p1 == p2

    def test_different_contexts_different_prompts(
        self, all_documents_context: RetrievedContext, specific_doc_context: RetrievedContext
    ) -> None:
        p1 = build_prompt(all_documents_context)
        p2 = build_prompt(specific_doc_context)
        assert p1 != p2


# ============================================================================
# Story 4.3 — Structured output schema and parsing
# ============================================================================

_VALID_JSON_ANSWER = {
    "answer_text": "Encryption keys must be rotated every 90 days.",
    "citations": [
        {
            "passage_id": "doc-abc123:page:5:chunk:0",
            "document_name": "Security Standards Manual",
            "location": "Page 5, Section 3.2",
        }
    ],
    "evidence_quality": "sufficient",
    "evidence_quality_narrative": "The passage directly states the 90-day rotation requirement.",
    "is_abstention": False,
}


class TestParseLLMOutputValid:
    """Story 4.3 AC: parse_llm_output correctly parses valid LLM output."""

    def test_parse_raw_json(self) -> None:
        raw = json.dumps(_VALID_JSON_ANSWER)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.answer_text == "Encryption keys must be rotated every 90 days."
        assert len(result.citations) == 1
        assert result.citations[0].passage_id == "doc-abc123:page:5:chunk:0"
        assert result.citations[0].document_name == "Security Standards Manual"
        assert result.citations[0].location == "Page 5, Section 3.2"
        assert result.citations[0].excerpt == ""  # not populated by parser
        assert result.evidence_quality == EvidenceQuality.SUFFICIENT
        assert result.evidence_quality_narrative == "The passage directly states the 90-day rotation requirement."
        assert result.is_abstention is False

    def test_parse_markdown_fenced_json(self) -> None:
        raw = f"```json\n{json.dumps(_VALID_JSON_ANSWER)}\n```"
        result = parse_llm_output(raw)
        assert result is not None
        assert result.answer_text == "Encryption keys must be rotated every 90 days."

    def test_parse_markdown_fenced_no_lang(self) -> None:
        raw = f"```\n{json.dumps(_VALID_JSON_ANSWER)}\n```"
        result = parse_llm_output(raw)
        assert result is not None
        assert result.answer_text == "Encryption keys must be rotated every 90 days."

    def test_parse_with_surrounding_text(self) -> None:
        raw = f"Here is my response:\n{json.dumps(_VALID_JSON_ANSWER)}\nHope this helps!"
        result = parse_llm_output(raw)
        assert result is not None
        assert result.answer_text == "Encryption keys must be rotated every 90 days."

    def test_multiple_citations(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        data["citations"] = [
            {
                "passage_id": "p1",
                "document_name": "Doc A",
                "location": "Page 1",
            },
            {
                "passage_id": "p2",
                "document_name": "Doc B",
                "location": "Page 3",
            },
        ]
        raw = json.dumps(data)
        result = parse_llm_output(raw)
        assert result is not None
        assert len(result.citations) == 2
        assert result.citations[0].passage_id == "p1"
        assert result.citations[1].passage_id == "p2"

    def test_empty_citations_array(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        data["citations"] = []
        raw = json.dumps(data)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.citations == []

    def test_is_abstention_false(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        data["is_abstention"] = False
        raw = json.dumps(data)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.is_abstention is False

    def test_is_abstention_true(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        data["is_abstention"] = True
        raw = json.dumps(data)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.is_abstention is True

    def test_evidence_quality_partial(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        data["evidence_quality"] = "partial"
        data["evidence_quality_narrative"] = "Partially supported."
        raw = json.dumps(data)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.evidence_quality == EvidenceQuality.PARTIAL

    def test_evidence_quality_insufficient(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        data["evidence_quality"] = "insufficient"
        data["evidence_quality_narrative"] = "Not enough info."
        raw = json.dumps(data)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.evidence_quality == EvidenceQuality.INSUFFICIENT

    def test_evidence_quality_conflicting(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        data["evidence_quality"] = "conflicting"
        data["evidence_quality_narrative"] = "Sources disagree."
        raw = json.dumps(data)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.evidence_quality == EvidenceQuality.CONFLICTING


class TestParseLLMOutputInvalid:
    """Story 4.3 AC: parse_llm_output returns None for invalid output."""

    def test_empty_string(self) -> None:
        assert parse_llm_output("") is None
        assert parse_llm_output("   ") is None

    def test_none_handling(self) -> None:
        # parse_llm_output receives a string; None would be a type error upstream,
        # but guard against empty-ish strings.
        assert parse_llm_output("") is None

    def test_malformed_json(self) -> None:
        assert parse_llm_output("{this is not json}") is None

    def test_json_array_not_object(self) -> None:
        assert parse_llm_output("[1, 2, 3]") is None

    def test_json_with_missing_answer_text(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        del data["answer_text"]
        raw = json.dumps(data)
        assert parse_llm_output(raw) is None

    def test_json_with_missing_evidence_quality(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        del data["evidence_quality"]
        raw = json.dumps(data)
        assert parse_llm_output(raw) is None

    def test_json_with_missing_evidence_narrative(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        del data["evidence_quality_narrative"]
        raw = json.dumps(data)
        assert parse_llm_output(raw) is None

    def test_invalid_evidence_quality_value(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        data["evidence_quality"] = "unknown"
        raw = json.dumps(data)
        assert parse_llm_output(raw) is None

    def test_citation_missing_passage_id(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        data["citations"] = [
            {
                "document_name": "Doc",
                "location": "Page 1",
            }
        ]
        raw = json.dumps(data)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.citations == []  # invalid citation skipped

    def test_citation_missing_document_name(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        data["citations"] = [
            {
                "passage_id": "p1",
                "location": "Page 1",
            }
        ]
        raw = json.dumps(data)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.citations == []

    def test_citation_missing_location(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        data["citations"] = [
            {
                "passage_id": "p1",
                "document_name": "Doc",
            }
        ]
        raw = json.dumps(data)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.citations == []

    def test_citation_with_empty_fields(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        data["citations"] = [
            {
                "passage_id": "",
                "document_name": "Doc",
                "location": "Page 1",
            }
        ]
        raw = json.dumps(data)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.citations == []

    def test_non_dict_citation_skipped(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        data["citations"] = ["not a dict", 123, None]
        raw = json.dumps(data)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.citations == []

    def test_extra_fields_ignored(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        data["extra_field"] = "ignored"
        data["another_extra"] = 123
        raw = json.dumps(data)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.answer_text == _VALID_JSON_ANSWER["answer_text"]


class TestParseLLMOutputExcerptNotPopulated:
    """Story 4.3 AC: citation excerpt is NOT populated by the parser."""

    def test_excerpt_stays_empty(self) -> None:
        raw = json.dumps(_VALID_JSON_ANSWER)
        result = parse_llm_output(raw)
        assert result is not None
        for cit in result.citations:
            assert cit.excerpt == "", "parse_llm_output must not populate excerpt"


class TestParseLLMOutputBooleanVariants:
    """Story 4.3 AC: parse_llm_output handles various boolean representations."""

    def test_is_abstention_integer_one(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        data["is_abstention"] = 1
        raw = json.dumps(data)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.is_abstention is True

    def test_is_abstention_integer_zero(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        data["is_abstention"] = 0
        raw = json.dumps(data)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.is_abstention is False

    def test_is_abstention_string_true(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        data["is_abstention"] = "true"
        raw = json.dumps(data)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.is_abstention is True

    def test_is_abstention_string_false(self) -> None:
        data = dict(_VALID_JSON_ANSWER)
        data["is_abstention"] = "false"
        raw = json.dumps(data)
        result = parse_llm_output(raw)
        assert result is not None
        assert result.is_abstention is False

    def test_is_abstention_string_yes_no(self) -> None:
        for yes_val in ("yes", "1", "True", "TRUE"):
            data = dict(_VALID_JSON_ANSWER)
            data["is_abstention"] = yes_val
            raw = json.dumps(data)
            result = parse_llm_output(raw)
            assert result is not None, f"Failed for is_abstention={yes_val!r}"
            assert result.is_abstention is True

        for no_val in ("no", "0", "False", "FALSE"):
            data = dict(_VALID_JSON_ANSWER)
            data["is_abstention"] = no_val
            raw = json.dumps(data)
            result = parse_llm_output(raw)
            assert result is not None, f"Failed for is_abstention={no_val!r}"
            assert result.is_abstention is False


class TestGeneratedAnswerConstructionFromRetrievedContext:
    """Story 4.3 AC: GeneratedAnswer can be constructed from retrieved passage data."""

    def test_citation_fields_from_passage(self, sample_passage: RetrievedPassage) -> None:
        """A Citation built from RetrievedPassage data carries the right fields."""
        cit = Citation(
            passage_id=sample_passage.passage_id,
            document_name=sample_passage.document_name,
            location=sample_passage.location,
        )
        assert cit.passage_id == sample_passage.passage_id
        assert cit.document_name == sample_passage.document_name
        assert cit.location == sample_passage.location
        assert cit.excerpt == ""  # not populated here


# ============================================================================
# GenerationConfig validation tests
# ============================================================================

class TestGenerationConfigValidation:
    """Story 4.1 AC: GenerationConfig validates its inputs."""

    def test_default_is_valid(self) -> None:
        cfg = GenerationConfig()
        assert cfg.max_passages_in_prompt >= 1
        assert EvidenceQuality.is_valid(cfg.default_evidence_quality)

    def test_all_evidence_quality_values_accepted(self) -> None:
        for eq in ("sufficient", "partial", "insufficient", "conflicting"):
            cfg = GenerationConfig(default_evidence_quality=eq)
            assert cfg.default_evidence_quality == eq

    def test_negative_max_passages_rejected(self) -> None:
        with pytest.raises(ValueError):
            GenerationConfig(max_passages_in_prompt=-1)

    def test_zero_max_passages_rejected(self) -> None:
        with pytest.raises(ValueError):
            GenerationConfig(max_passages_in_prompt=0)

    def test_invalid_evidence_quality_rejected(self) -> None:
        with pytest.raises(ValueError):
            GenerationConfig(default_evidence_quality="bogus")

    def test_case_sensitive_evidence_quality(self) -> None:
        with pytest.raises(ValueError):
            GenerationConfig(default_evidence_quality="SUFFICIENT")


# ============================================================================
# Integration-style tests (synthetic, no LLM required)
# ============================================================================

class TestPromptToLLMContract:
    """Story 4.2 + 4.3: prompt → LLM → parse chain with synthetic output."""

    def test_build_prompt_then_parse(
        self, all_documents_context: RetrievedContext
    ) -> None:
        """The prompt instructs JSON output; a valid JSON response parses correctly."""
        config = GenerationConfig(require_json_output=True)
        prompt = build_prompt(all_documents_context, config)

        # Verify the prompt asks for JSON
        assert "JSON" in prompt
        assert "answer_text" in prompt

        # Simulate an LLM that follows the JSON format
        llm_output = {
            "answer_text": "Based on the Security Standards Manual, encryption keys must be rotated every 90 days.",
            "citations": [
                {
                    "passage_id": all_documents_context.passages[0].passage_id,
                    "document_name": all_documents_context.passages[0].document_name,
                    "location": all_documents_context.passages[0].location,
                }
            ],
            "evidence_quality": "sufficient",
            "evidence_quality_narrative": "Directly supported by the retrieved passage.",
            "is_abstention": False,
        }
        raw = json.dumps(llm_output)
        result = parse_llm_output(raw)
        assert result is not None
        assert "90 days" in result.answer_text
        assert len(result.citations) == 1

    def test_prompt_without_json_requirement(
        self, all_documents_context: RetrievedContext
    ) -> None:
        """When require_json_output=False, the output format section is omitted."""
        config = GenerationConfig(require_json_output=False)
        prompt = build_prompt(all_documents_context, config)
        assert "JSON" not in prompt
        assert "answer_text" not in prompt


# ============================================================================
# Story 4.4 — Abstention logic
# ============================================================================


class TestFinalizeGenerationAbstention:
    """Story 4.4 AC: finalize_generation enforces abstention rules.

    Tests use synthetic GeneratedAnswer objects (no LLM required).
    """

    def test_insufficient_evidence_sets_abstention(self) -> None:
        """INSUFFICIENT evidence → is_abstention=True."""
        answer = GeneratedAnswer(
            answer_text="The documents do not contain enough information to answer this question definitively.",
            citations=[],
            evidence_quality=EvidenceQuality.INSUFFICIENT,
            evidence_quality_narrative="No relevant passages were found for this query.",
            is_abstention=False,  # LLM may not set this correctly; we enforce it
            generation_latency_seconds=0.5,
        )
        result = finalize_generation(answer)
        assert result.is_abstention is True
        assert result.evidence_quality == EvidenceQuality.INSUFFICIENT

    def test_sufficient_evidence_clears_abstention(self) -> None:
        """SUFFICIENT evidence → is_abstention=False."""
        answer = GeneratedAnswer(
            answer_text="Encryption keys must be rotated every 90 days.",
            citations=[
                Citation(
                    passage_id="doc-1:page:5:chunk:0",
                    document_name="Security Manual",
                    location="Page 5, Section 3.2",
                )
            ],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="Directly supported by the retrieved passage.",
            is_abstention=True,  # LLM may incorrectly set this; we enforce it
            generation_latency_seconds=0.3,
        )
        result = finalize_generation(answer)
        assert result.is_abstention is False
        assert result.evidence_quality == EvidenceQuality.SUFFICIENT

    def test_partial_evidence_preserves_abstention_flag(self) -> None:
        """PARTIAL evidence → is_abstention left as LLM set it (not enforced)."""
        answer = GeneratedAnswer(
            answer_text="The documents support the rotation requirement, "
            "but do not specify the exact period. This is a reasonable inference.",
            citations=[
                Citation(
                    passage_id="doc-1:page:5:chunk:0",
                    document_name="Security Manual",
                    location="Page 5, Section 3.2",
                )
            ],
            evidence_quality=EvidenceQuality.PARTIAL,
            evidence_quality_narrative="The documents support part of the answer but not all.",
            is_abstention=False,
            generation_latency_seconds=0.4,
        )
        result = finalize_generation(answer)
        assert result.is_abstention is False  # left as-is
        assert result.evidence_quality == EvidenceQuality.PARTIAL

    def test_conflicting_evidence_preserves_abstention_flag(self) -> None:
        """CONFLICTING evidence → is_abstention left as LLM set it."""
        answer = GeneratedAnswer(
            answer_text="Document A says encryption keys must be rotated every 90 days. "
            "Document B says they must be rotated every 180 days. The documents disagree.",
            citations=[
                Citation(
                    passage_id="doc-a:page:5:chunk:0",
                    document_name="Document A",
                    location="Page 5, Section 3.2",
                ),
                Citation(
                    passage_id="doc-b:page:3:chunk:1",
                    document_name="Document B",
                    location="Page 3, Paragraph 2",
                ),
            ],
            evidence_quality=EvidenceQuality.CONFLICTING,
            evidence_quality_narrative="The documents disagree on the rotation period.",
            is_abstention=False,
            generation_latency_seconds=0.6,
        )
        result = finalize_generation(answer)
        assert result.is_abstention is False  # left as-is
        assert result.evidence_quality == EvidenceQuality.CONFLICTING

    def test_abstention_answer_identifies_closest_passage(
        self, sample_passage: RetrievedPassage
    ) -> None:
        """Abstention with context: closest passage is available for reference."""
        context = RetrievedContext(
            query="What is the capital of France?",
            scope={"mode": "all", "document_id": None},
            passages=[sample_passage],
            conversation_history=None,
            retrieval_latency_seconds=0.2,
        )
        answer = GeneratedAnswer(
            answer_text="The documents do not contain information about the capital of France. "
            "The closest relevant passage discusses encryption key rotation (Security Standards Manual, Page 5).",
            citations=[],
            evidence_quality=EvidenceQuality.INSUFFICIENT,
            evidence_quality_narrative="No relevant passages for this question; closest passage is about encryption.",
            is_abstention=False,
            generation_latency_seconds=0.3,
        )
        result = finalize_generation(answer, context)
        assert result.is_abstention is True
        assert result.evidence_quality == EvidenceQuality.INSUFFICIENT
        # The answer text mentions the closest passage (Security Standards Manual)
        assert "Security Standards Manual" in result.answer_text

    def test_abstention_without_context(self) -> None:
        """Abstention with no retrieved passages: no closest passage available."""
        context = RetrievedContext(
            query="What is the capital of France?",
            scope={"mode": "all", "document_id": None},
            passages=[],
            conversation_history=None,
            retrieval_latency_seconds=0.1,
        )
        answer = GeneratedAnswer(
            answer_text="The documents do not contain enough information to answer this question.",
            citations=[],
            evidence_quality=EvidenceQuality.INSUFFICIENT,
            evidence_quality_narrative="No passages were retrieved for this query.",
            is_abstention=False,
            generation_latency_seconds=0.2,
        )
        result = finalize_generation(answer, context)
        assert result.is_abstention is True
        assert result.evidence_quality == EvidenceQuality.INSUFFICIENT
        assert result.citations == []

    def test_abstention_does_not_fabricate(self) -> None:
        """Abstention answer must not fabricate a definitive answer."""
        answer = GeneratedAnswer(
            answer_text="The documents do not contain enough information to answer "
            "this question definitively. No relevant passages were found.",
            citations=[],
            evidence_quality=EvidenceQuality.INSUFFICIENT,
            evidence_quality_narrative="No relevant passages for this query.",
            is_abstention=False,
            generation_latency_seconds=0.3,
        )
        result = finalize_generation(answer)
        assert result.is_abstention is True
        # The answer text does not make a definitive claim about the question
        assert "do not contain enough information" in result.answer_text.lower()
        # It should not contain a fabricated definitive answer
        # (e.g., it should not say "The answer is X" when evidence is insufficient)
        assert_result_is_not_fabricated(result)


def assert_result_is_not_fabricated(result: GeneratedAnswer) -> None:
    """Assert that a GeneratedAnswer does not fabricate a definitive claim.

    This is a test helper. A fabricated answer makes a definitive claim
    about the question when evidence is insufficient. We check that the
    answer text uses abstention language rather than definitive assertion
    language.

    This is a heuristic check for testing — the real enforcement is the
    prompt instructions to the LLM.
    """
    text = result.answer_text.lower()
    # Abstention answers should use qualifying language
    abstention_phrases = [
        "do not contain", "does not contain", "not enough",
        "cannot answer", "no information", "not available",
        "does not support", "do not support",
    ]
    has_abstention_language = any(phrase in text for phrase in abstention_phrases)
    assert has_abstention_language, (
        f"Answer appears to fabricate a definitive claim: {result.answer_text!r}. "
        f"Expected abstention language (e.g., 'do not contain enough information')."
    )


class TestIsAbstentionAnswer:
    """Story 4.4 AC: is_abstention_answer helper correctly identifies abstentions."""

    def test_abstention_identified(self) -> None:
        answer = GeneratedAnswer(
            answer_text="The documents do not contain enough information.",
            citations=[],
            evidence_quality=EvidenceQuality.INSUFFICIENT,
            evidence_quality_narrative="No relevant passages.",
            is_abstention=True,
            generation_latency_seconds=0.1,
        )
        assert is_abstention_answer(answer) is True

    def test_non_abstention_sufficient(self) -> None:
        answer = GeneratedAnswer(
            answer_text="Encryption keys must be rotated every 90 days.",
            citations=[],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="Directly supported.",
            is_abstention=False,
            generation_latency_seconds=0.1,
        )
        assert is_abstention_answer(answer) is False

    def test_non_abstention_partial(self) -> None:
        answer = GeneratedAnswer(
            answer_text="Partially supported.",
            citations=[],
            evidence_quality=EvidenceQuality.PARTIAL,
            evidence_quality_narrative="Partially supported.",
            is_abstention=False,
            generation_latency_seconds=0.1,
        )
        assert is_abstention_answer(answer) is False

    def test_non_abstention_conflicting(self) -> None:
        answer = GeneratedAnswer(
            answer_text="Sources disagree.",
            citations=[],
            evidence_quality=EvidenceQuality.CONFLICTING,
            evidence_quality_narrative="Sources disagree.",
            is_abstention=False,
            generation_latency_seconds=0.1,
        )
        assert is_abstention_answer(answer) is False


class TestFinalizeGenerationIdempotent:
    """Story 4.4 AC: finalize_generation is idempotent for already-correct answers."""

    def test_sufficient_already_correct(self) -> None:
        answer = GeneratedAnswer(
            answer_text="The answer.",
            citations=[],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="Supported.",
            is_abstention=False,
            generation_latency_seconds=0.1,
        )
        result = finalize_generation(answer)
        assert result.is_abstention is False
        assert result.evidence_quality == EvidenceQuality.SUFFICIENT
        assert result.answer_text == answer.answer_text

    def test_insufficient_already_correct(self) -> None:
        answer = GeneratedAnswer(
            answer_text="The documents do not contain enough information.",
            citations=[],
            evidence_quality=EvidenceQuality.INSUFFICIENT,
            evidence_quality_narrative="No relevant passages.",
            is_abstention=True,
            generation_latency_seconds=0.1,
        )
        result = finalize_generation(answer)
        assert result.is_abstention is True
        assert result.evidence_quality == EvidenceQuality.INSUFFICIENT
        assert result.answer_text == answer.answer_text


# ============================================================================
# Story 4.5 — Partial and conflicting evidence handling
# ============================================================================


class TestPartialEvidenceHandling:
    """Story 4.5 AC: partial evidence answers distinguish supported from inference."""

    def test_partial_evidence_distinguishes_supported_from_inference(self) -> None:
        """Answer with PARTIAL evidence distinguishes supported claims from inference."""
        answer = GeneratedAnswer(
            answer_text="The documents state that encryption keys must be rotated periodically. "
            "The exact rotation period (90 days) is not stated in the documents, but is a "
            "reasonable inference based on industry standards. [Supported: keys must be rotated; "
            "Inference: 90-day period]",
            citations=[
                Citation(
                    passage_id="doc-1:page:5:chunk:0",
                    document_name="Security Manual",
                    location="Page 5, Section 3.2",
                )
            ],
            evidence_quality=EvidenceQuality.PARTIAL,
            evidence_quality_narrative="The documents support the rotation requirement but not the specific period.",
            is_abstention=False,
            generation_latency_seconds=0.5,
        )
        result = finalize_generation(answer)
        assert result.evidence_quality == EvidenceQuality.PARTIAL
        assert result.is_abstention is False
        # Answer distinguishes supported from inference
        assert "Supported" in result.answer_text or "inference" in result.answer_text.lower()
        # Answer does not present unsupported info as fact
        assert "not stated" in result.answer_text.lower() or "inference" in result.answer_text.lower()

    def test_partial_evidence_narrative_uses_partial_language(self) -> None:
        """Partial evidence narrative uses the partial language from the architecture."""
        answer = GeneratedAnswer(
            answer_text="The documents support part of this answer.",
            citations=[],
            evidence_quality=EvidenceQuality.PARTIAL,
            evidence_quality_narrative="The documents support part of this. The rotation requirement "
            "appears in the Security Manual; the specific period is a reasonable inference but not "
            "stated explicitly.",
            is_abstention=False,
            generation_latency_seconds=0.3,
        )
        result = finalize_generation(answer)
        assert result.evidence_quality == EvidenceQuality.PARTIAL
        assert "part of this" in result.evidence_quality_narrative.lower()

    def test_partial_evidence_cites_supported_claims(self) -> None:
        """Partial evidence answer cites passages for supported claims."""
        answer = GeneratedAnswer(
            answer_text="Based on the Security Standards Manual (Page 5, Section 3.2), "
            "encryption keys must be rotated. The specific period is not stated.",
            citations=[
                Citation(
                    passage_id="doc-1:page:5:chunk:0",
                    document_name="Security Standards Manual",
                    location="Page 5, Section 3.2",
                )
            ],
            evidence_quality=EvidenceQuality.PARTIAL,
            evidence_quality_narrative="Supported: keys must be rotated. Not supported: specific period.",
            is_abstention=False,
            generation_latency_seconds=0.4,
        )
        result = finalize_generation(answer)
        assert result.evidence_quality == EvidenceQuality.PARTIAL
        assert len(result.citations) == 1
        assert result.citations[0].passage_id == "doc-1:page:5:chunk:0"

    def test_partial_evidence_does_not_invent_missing_info(self) -> None:
        """Partial evidence answer does not invent missing information."""
        answer = GeneratedAnswer(
            answer_text="The documents state that encryption keys must be rotated periodically. "
            "The exact rotation period is not specified in the documents.",
            citations=[
                Citation(
                    passage_id="doc-1:page:5:chunk:0",
                    document_name="Security Manual",
                    location="Page 5, Section 3.2",
                )
            ],
            evidence_quality=EvidenceQuality.PARTIAL,
            evidence_quality_narrative="Rotation requirement is supported; specific period is not stated.",
            is_abstention=False,
            generation_latency_seconds=0.4,
        )
        result = finalize_generation(answer)
        assert result.evidence_quality == EvidenceQuality.PARTIAL
        # Answer explicitly states the period is not specified (does not invent)
        assert "not specified" in result.answer_text.lower() or "not stated" in result.answer_text.lower()


class TestConflictingEvidenceHandling:
    """Story 4.5 AC: conflicting evidence answers name both sources and don't choose."""

    def test_conflicting_evidence_names_both_sources(self) -> None:
        """Conflicting evidence answer names both conflicting sources."""
        answer = GeneratedAnswer(
            answer_text="The documents disagree on the encryption key rotation period. "
            "The Security Standards Manual (Document A, Page 5, Section 3.2) states that keys "
            "must be rotated every 90 days. The Access Control Policy (Document B, Page 3, "
            "Paragraph 2) states that keys must be rotated every 180 days.",
            citations=[
                Citation(
                    passage_id="doc-a:page:5:chunk:0",
                    document_name="Security Standards Manual",
                    location="Page 5, Section 3.2",
                ),
                Citation(
                    passage_id="doc-b:page:3:chunk:1",
                    document_name="Access Control Policy",
                    location="Page 3, Paragraph 2",
                ),
            ],
            evidence_quality=EvidenceQuality.CONFLICTING,
            evidence_quality_narrative="Document A says 90 days; Document B says 180 days.",
            is_abstention=False,
            generation_latency_seconds=0.6,
        )
        result = finalize_generation(answer)
        assert result.evidence_quality == EvidenceQuality.CONFLICTING
        assert result.is_abstention is False
        # Answer names both sources
        assert "Security Standards Manual" in result.answer_text
        assert "Access Control Policy" in result.answer_text
        # Answer presents what each says
        assert "90 days" in result.answer_text
        assert "180 days" in result.answer_text

    def test_conflicting_evidence_does_not_silently_choose(self) -> None:
        """Conflicting evidence answer does not silently choose one side."""
        answer = GeneratedAnswer(
            answer_text="The documents disagree. Document A (Security Standards Manual, "
            "Page 5) says keys must be rotated every 90 days. Document B (Access Control Policy, "
            "Page 3) says keys must be rotated every 180 days. The documents provide conflicting "
            "information on this point.",
            citations=[
                Citation(
                    passage_id="doc-a:page:5:chunk:0",
                    document_name="Security Standards Manual",
                    location="Page 5",
                ),
                Citation(
                    passage_id="doc-b:page:3:chunk:1",
                    document_name="Access Control Policy",
                    location="Page 3",
                ),
            ],
            evidence_quality=EvidenceQuality.CONFLICTING,
            evidence_quality_narrative="Both sources are presented; no resolution is claimed.",
            is_abstention=False,
            generation_latency_seconds=0.5,
        )
        result = finalize_generation(answer)
        assert result.evidence_quality == EvidenceQuality.CONFLICTING
        # Answer does not pick a winner
        assert "disagree" in result.answer_text.lower()
        assert "conflicting" in result.answer_text.lower()
        # Both sources are named
        assert len(result.citations) == 2

    def test_conflicting_evidence_narrative_uses_conflicting_language(self) -> None:
        """Conflicting evidence narrative uses the conflicting language from the architecture."""
        answer = GeneratedAnswer(
            answer_text="The documents disagree on this point.",
            citations=[],
            evidence_quality=EvidenceQuality.CONFLICTING,
            evidence_quality_narrative="The documents disagree. Source A says X; Source B says Y.",
            is_abstention=False,
            generation_latency_seconds=0.3,
        )
        result = finalize_generation(answer)
        assert result.evidence_quality == EvidenceQuality.CONFLICTING
        assert "disagree" in result.evidence_quality_narrative.lower()

    def test_conflicting_evidence_with_multiple_sources(self) -> None:
        """Conflicting evidence with more than two sources preserves all sources."""
        answer = GeneratedAnswer(
            answer_text="Three documents provide conflicting information on the rotation period. "
            "Document A says 90 days; Document B says 180 days; Document C says 365 days.",
            citations=[
                Citation(passage_id="a:1", document_name="Document A", location="Page 1"),
                Citation(passage_id="b:1", document_name="Document B", location="Page 1"),
                Citation(passage_id="c:1", document_name="Document C", location="Page 1"),
            ],
            evidence_quality=EvidenceQuality.CONFLICTING,
            evidence_quality_narrative="Three sources disagree: 90 days, 180 days, 365 days.",
            is_abstention=False,
            generation_latency_seconds=0.7,
        )
        result = finalize_generation(answer)
        assert result.evidence_quality == EvidenceQuality.CONFLICTING
        assert len(result.citations) == 3
        assert "90 days" in result.answer_text
        assert "180 days" in result.answer_text
        assert "365 days" in result.answer_text


class TestEvidenceQualityClassificationHelper:
    """Test the classify_evidence_quality helper (test utility, not runtime)."""

    def test_no_passages_returns_insufficient(self) -> None:
        ctx = RetrievedContext(
            query="test",
            scope={"mode": "all", "document_id": None},
            passages=[],
        )
        result = classify_evidence_quality("The answer is X.", ctx)
        assert result == EvidenceQuality.INSUFFICIENT

    def test_none_context_returns_insufficient(self) -> None:
        result = classify_evidence_quality("The answer is X.", None)
        assert result == EvidenceQuality.INSUFFICIENT

    def test_conflict_indicators_detected(self) -> None:
        ctx = RetrievedContext(
            query="test",
            scope={"mode": "all", "document_id": None},
            passages=[RetrievedPassage("p1", "Doc A", "Page 1", "text", 0.9)],
        )
        result = classify_evidence_quality(
            "Document A says X, but Document B says Y. The documents disagree.",
            ctx,
        )
        assert result == EvidenceQuality.CONFLICTING

    def test_partial_indicators_detected(self) -> None:
        ctx = RetrievedContext(
            query="test",
            scope={"mode": "all", "document_id": None},
            passages=[RetrievedPassage("p1", "Doc A", "Page 1", "text", 0.9)],
        )
        result = classify_evidence_quality(
            "The documents partially support this. The rotation requirement is stated, "
            "but the specific period is a reasonable inference, not stated explicitly.",
            ctx,
        )
        assert result == EvidenceQuality.PARTIAL

    def test_insufficient_indicators_detected(self) -> None:
        ctx = RetrievedContext(
            query="test",
            scope={"mode": "all", "document_id": None},
            passages=[RetrievedPassage("p1", "Doc A", "Page 1", "text", 0.9)],
        )
        result = classify_evidence_quality(
            "The documents do not contain enough information to answer this question.",
            ctx,
        )
        assert result == EvidenceQuality.INSUFFICIENT

    def test_default_sufficient_when_undetermined(self) -> None:
        ctx = RetrievedContext(
            query="test",
            scope={"mode": "all", "document_id": None},
            passages=[RetrievedPassage("p1", "Doc A", "Page 1", "text", 0.9)],
        )
        result = classify_evidence_quality(
            "Encryption keys must be rotated every 90 days.",
            ctx,
        )
        assert result == EvidenceQuality.SUFFICIENT


# ============================================================================
# Cross-story regression tests
# ============================================================================


class TestStories44And45DoNotBreak41To43:
    """Verify that Stories 4.4 and 4.5 changes do not break 4.1–4.3 behavior."""

    def test_evidence_quality_enum_unchanged(self) -> None:
        """EvidenceQuality still has the same four values."""
        assert EvidenceQuality.SUFFICIENT == "sufficient"
        assert EvidenceQuality.PARTIAL == "partial"
        assert EvidenceQuality.INSUFFICIENT == "insufficient"
        assert EvidenceQuality.CONFLICTING == "conflicting"

    def test_generated_answer_fields_unchanged(self) -> None:
        """GeneratedAnswer still has all required fields."""
        answer = GeneratedAnswer(
            answer_text="test",
            citations=[],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="test",
            is_abstention=False,
            generation_latency_seconds=0.0,
        )
        assert answer.answer_text == "test"
        assert answer.citations == []
        assert answer.evidence_quality == EvidenceQuality.SUFFICIENT

    def test_citation_excerpt_still_empty(self) -> None:
        """Citation.excerpt is still not populated by generation."""
        cit = Citation(
            passage_id="p1",
            document_name="Doc",
            location="Page 1",
        )
        assert cit.excerpt == ""

    def test_parse_llm_output_still_works(self) -> None:
        """parse_llm_output still parses valid JSON (4.3 not broken)."""
        raw = json.dumps({
            "answer_text": "test answer",
            "citations": [],
            "evidence_quality": "sufficient",
            "evidence_quality_narrative": "test",
            "is_abstention": False,
        })
        result = parse_llm_output(raw)
        assert result is not None
        assert result.answer_text == "test answer"

    def test_build_prompt_still_produces_all_sections(self) -> None:
        """build_prompt still produces all 8 sections (4.2 not broken)."""
        ctx = RetrievedContext(
            query="test question",
            scope={"mode": "all", "document_id": None},
            passages=[RetrievedPassage("p1", "Doc", "Page 1", "passage text", 0.9)],
        )
        prompt = build_prompt(ctx)
        assert "## Role and Task" in prompt
        assert "## User's Question" in prompt
        assert "## Retrieved Passages" in prompt
        assert "## Document Scope" in prompt
        assert "## Conversation Context" in prompt
        assert "Answer instructions:" in prompt
        assert "Evidence quality assessment:" in prompt
        assert "Output format:" in prompt

    def test_finalize_generation_preserves_answer_text(self) -> None:
        """finalize_generation does not modify answer_text (only flags)."""
        original_text = "The documents do not contain enough information."
        answer = GeneratedAnswer(
            answer_text=original_text,
            citations=[],
            evidence_quality=EvidenceQuality.INSUFFICIENT,
            evidence_quality_narrative="No relevant passages.",
            is_abstention=False,
            generation_latency_seconds=0.1,
        )
        result = finalize_generation(answer)
        assert result.answer_text == original_text

    def test_finalize_generation_preserves_citations(self) -> None:
        """finalize_generation does not modify citations."""
        cit = Citation(
            passage_id="p1",
            document_name="Doc",
            location="Page 1",
        )
        answer = GeneratedAnswer(
            answer_text="test",
            citations=[cit],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="test",
            is_abstention=False,
            generation_latency_seconds=0.1,
        )
        result = finalize_generation(answer)
        assert len(result.citations) == 1
        assert result.citations[0].passage_id == "p1"
        assert result.citations[0].document_name == "Doc"
        assert result.citations[0].location == "Page 1"
        assert result.citations[0].excerpt == ""  # still not populated


# ============================================================================
# Story 4.6 — Citation generation and resolution
# ============================================================================


class TestResolveCitationsBasic:
    """Story 4.6 AC: resolve_citations validates and resolves citations."""

    def test_resolve_valid_citations(self, sample_passages: list[RetrievedPassage]) -> None:
        """Valid citations are resolved from trusted RetrievedPassage data."""
        ctx = RetrievedContext(
            query="test",
            scope={"mode": "all", "document_id": None},
            passages=sample_passages,
        )
        # Create citations with WRONG document_name/location (LLM-supplied, untrusted)
        answer = GeneratedAnswer(
            answer_text="The answer.",
            citations=[
                Citation(
                    passage_id=sample_passages[0].passage_id,
                    document_name="WRONG_DOC_NAME",
                    location="WRONG_LOCATION",
                ),
                Citation(
                    passage_id=sample_passages[1].passage_id,
                    document_name="ALSO_WRONG",
                    location="ALSO_WRONG_LOC",
                ),
            ],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="test",
            is_abstention=False,
            generation_latency_seconds=0.1,
        )
        result = resolve_citations(answer, ctx)
        assert len(result.citations) == 2
        # document_name and location should come from trusted RetrievedPassage, NOT LLM-supplied
        assert result.citations[0].document_name == sample_passages[0].document_name
        assert result.citations[0].location == sample_passages[0].location
        assert result.citations[1].document_name == sample_passages[1].document_name
        assert result.citations[1].location == sample_passages[1].location
        # LLM-supplied values are NOT preserved
        assert result.citations[0].document_name != "WRONG_DOC_NAME"
        assert result.citations[0].location != "WRONG_LOCATION"
        # excerpt is the actual passage text from the trusted RetrievedPassage
        assert result.citations[0].excerpt == sample_passages[0].text
        assert result.citations[1].excerpt == sample_passages[1].text

    def test_unknown_passage_id_dropped(self, sample_passages: list[RetrievedPassage]) -> None:
        """Citations with passage_id not in retrieved context are dropped."""
        ctx = RetrievedContext(
            query="test",
            scope={"mode": "all", "document_id": None},
            passages=sample_passages,
        )
        answer = GeneratedAnswer(
            answer_text="The answer.",
            citations=[
                Citation(
                    passage_id=sample_passages[0].passage_id,
                    document_name="Doc A",
                    location="Page 1",
                ),
                Citation(
                    passage_id="NONEXISTENT_PASAGE_ID",
                    document_name="Unknown Doc",
                    location="Unknown Loc",
                ),
            ],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="test",
            is_abstention=False,
            generation_latency_seconds=0.1,
        )
        result = resolve_citations(answer, ctx)
        assert len(result.citations) == 1
        assert result.citations[0].passage_id == sample_passages[0].passage_id
        assert result.citations[0].document_name == sample_passages[0].document_name

    def test_all_citations_unknown_dropped(self, sample_passages: list[RetrievedPassage]) -> None:
        """When all citations reference unknown passage_ids, result has no citations."""
        ctx = RetrievedContext(
            query="test",
            scope={"mode": "all", "document_id": None},
            passages=sample_passages,
        )
        answer = GeneratedAnswer(
            answer_text="The answer.",
            citations=[
                Citation(passage_id="p999", document_name="Unknown", location="Unknown"),
                Citation(passage_id="p888", document_name="Also Unknown", location="Also Unknown"),
            ],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="test",
            is_abstention=False,
            generation_latency_seconds=0.1,
        )
        result = resolve_citations(answer, ctx)
        assert len(result.citations) == 0
        assert result.answer_text == "The answer."
        assert result.evidence_quality == EvidenceQuality.SUFFICIENT

    def test_answer_text_and_evidence_preserved(self, sample_passages: list[RetrievedPassage]) -> None:
        """resolve_citations does not modify answer_text or evidence_quality."""
        ctx = RetrievedContext(
            query="test",
            scope={"mode": "all", "document_id": None},
            passages=sample_passages,
        )
        original_text = "This is the original answer text with specific details."
        original_narrative = "This is the evidence quality narrative."
        answer = GeneratedAnswer(
            answer_text=original_text,
            citations=[
                Citation(
                    passage_id=sample_passages[0].passage_id,
                    document_name=sample_passages[0].document_name,
                    location=sample_passages[0].location,
                ),
            ],
            evidence_quality=EvidenceQuality.PARTIAL,
            evidence_quality_narrative=original_narrative,
            is_abstention=True,
            generation_latency_seconds=0.5,
        )
        result = resolve_citations(answer, ctx)
        assert result.answer_text == original_text
        assert result.evidence_quality == EvidenceQuality.PARTIAL
        assert result.evidence_quality_narrative == original_narrative
        assert result.is_abstention is True
        assert result.generation_latency_seconds == 0.5
        # excerpt is resolved from the trusted RetrievedPassage
        assert result.citations[0].excerpt == sample_passages[0].text


class TestResolveCitationsEdgeCases:
    """Story 4.6 AC: edge cases for citation resolution."""

    def test_empty_citations_preserved(self, sample_passages: list[RetrievedPassage]) -> None:
        """Answer with empty citations list is preserved as-is."""
        ctx = RetrievedContext(
            query="test",
            scope={"mode": "all", "document_id": None},
            passages=sample_passages,
        )
        answer = GeneratedAnswer(
            answer_text="No citations needed.",
            citations=[],
            evidence_quality=EvidenceQuality.INSUFFICIENT,
            evidence_quality_narrative="No relevant passages.",
            is_abstention=True,
            generation_latency_seconds=0.2,
        )
        result = resolve_citations(answer, ctx)
        assert result.citations == []
        assert result.answer_text == "No citations needed."
        assert result.evidence_quality == EvidenceQuality.INSUFFICIENT

    def test_none_context_drops_all_citations(self) -> None:
        """When context is None, all citations are dropped (no trusted source)."""
        answer = GeneratedAnswer(
            answer_text="The answer.",
            citations=[
                Citation(passage_id="p1", document_name="Doc A", location="Page 1"),
                Citation(passage_id="p2", document_name="Doc B", location="Page 2"),
            ],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="test",
            is_abstention=False,
            generation_latency_seconds=0.1,
        )
        result = resolve_citations(answer, None)
        assert len(result.citations) == 0
        assert result.answer_text == "The answer."

    def test_empty_passages_drops_all_citations(self, sample_passages: list[RetrievedPassage]) -> None:
        """When context has no passages, all citations are dropped."""
        ctx = RetrievedContext(
            query="test",
            scope={"mode": "all", "document_id": None},
            passages=[],  # empty
        )
        answer = GeneratedAnswer(
            answer_text="The answer.",
            citations=[
                Citation(passage_id="p1", document_name="Doc A", location="Page 1"),
            ],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="test",
            is_abstention=False,
            generation_latency_seconds=0.1,
        )
        result = resolve_citations(answer, ctx)
        assert len(result.citations) == 0

    def test_malformed_citation_skipped(self, sample_passages: list[RetrievedPassage]) -> None:
        """Citations with missing fields (malformed) are skipped, valid ones kept."""
        ctx = RetrievedContext(
            query="test",
            scope={"mode": "all", "document_id": None},
            passages=sample_passages,
        )
        # Note: Citation is a frozen dataclass so all fields are always present.
        # The "malformed" case here is a citation with an unknown passage_id.
        # We also test that a valid citation alongside an invalid one works.
        answer = GeneratedAnswer(
            answer_text="The answer.",
            citations=[
                Citation(
                    passage_id=sample_passages[0].passage_id,
                    document_name=sample_passages[0].document_name,
                    location=sample_passages[0].location,
                ),
                Citation(
                    passage_id="UNKNOWN",
                    document_name="Unknown",
                    location="Unknown",
                ),
                Citation(
                    passage_id=sample_passages[1].passage_id,
                    document_name=sample_passages[1].document_name,
                    location=sample_passages[1].location,
                ),
            ],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="test",
            is_abstention=False,
            generation_latency_seconds=0.1,
        )
        result = resolve_citations(answer, ctx)
        assert len(result.citations) == 2  # unknown dropped, two valid kept
        assert result.citations[0].passage_id == sample_passages[0].passage_id
        assert result.citations[1].passage_id == sample_passages[1].passage_id

    def test_duplicate_passage_ids_deduped(self, sample_passages: list[RetrievedPassage]) -> None:
        """Duplicate citations for the same passage_id are deduplicated (first kept)."""
        ctx = RetrievedContext(
            query="test",
            scope={"mode": "all", "document_id": None},
            passages=sample_passages,
        )
        answer = GeneratedAnswer(
            answer_text="The answer.",
            citations=[
                Citation(
                    passage_id=sample_passages[0].passage_id,
                    document_name=sample_passages[0].document_name,
                    location=sample_passages[0].location,
                ),
                Citation(
                    passage_id=sample_passages[0].passage_id,
                    document_name="DUPLICATE",
                    location="DUPLICATE_LOC",
                ),
                Citation(
                    passage_id=sample_passages[1].passage_id,
                    document_name=sample_passages[1].document_name,
                    location=sample_passages[1].location,
                ),
            ],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="test",
            is_abstention=False,
            generation_latency_seconds=0.1,
        )
        result = resolve_citations(answer, ctx)
        assert len(result.citations) == 2  # deduped: p1 once, p2 once
        assert result.citations[0].passage_id == sample_passages[0].passage_id
        assert result.citations[0].document_name == sample_passages[0].document_name
        assert result.citations[0].location == sample_passages[0].location
        # excerpt is the actual passage text from the trusted RetrievedPassage
        assert result.citations[0].excerpt == sample_passages[0].text
        # Duplicate was dropped (not the one with DUPLICATE name)
        assert result.citations[0].document_name != "DUPLICATE"
        assert result.citations[1].passage_id == sample_passages[1].passage_id

    def test_excerpt_resolved_from_passage_text(
        self, sample_passages: list[RetrievedPassage]
    ) -> None:
        """Excerpt is resolved from the RetrievedPassage text (not LLM-supplied)."""
        ctx = RetrievedContext(
            query="test",
            scope={"mode": "all", "document_id": None},
            passages=sample_passages,
        )
        answer = GeneratedAnswer(
            answer_text="The answer.",
            citations=[
                Citation(
                    passage_id=sample_passages[0].passage_id,
                    document_name=sample_passages[0].document_name,
                    location=sample_passages[0].location,
                    excerpt="LLM_LIES_ABOUT_TEXT",  # LLM-supplied, untrusted
                ),
            ],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="test",
            is_abstention=False,
            generation_latency_seconds=0.1,
        )
        result = resolve_citations(answer, ctx)
        # excerpt must be the actual passage text, NOT the LLM-supplied value
        assert result.citations[0].excerpt == sample_passages[0].text
        assert result.citations[0].excerpt != "LLM_LIES_ABOUT_TEXT"
        assert result.citations[0].document_name == sample_passages[0].document_name
        assert result.citations[0].location == sample_passages[0].location

    def test_resolve_with_specific_scope(self, sample_passages: list[RetrievedPassage]) -> None:
        """Citation resolution works with specific-document scope."""
        ctx = RetrievedContext(
            query="test",
            scope={"mode": "specific", "document_id": sample_passages[0].passage_id.split(":")[0]},
            passages=sample_passages,
        )
        answer = GeneratedAnswer(
            answer_text="The answer.",
            citations=[
                Citation(
                    passage_id=sample_passages[0].passage_id,
                    document_name="Wrong",
                    location="Wrong",
                ),
            ],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="test",
            is_abstention=False,
            generation_latency_seconds=0.1,
        )
        result = resolve_citations(answer, ctx)
        assert len(result.citations) == 1
        assert result.citations[0].document_name == sample_passages[0].document_name
        assert result.citations[0].location == sample_passages[0].location


class TestCitationHelperFunctions:
    """Story 4.6 AC: helper functions for citation inspection."""

    def test_citation_passage_ids(self, sample_passages: list[RetrievedPassage]) -> None:
        """citation_passage_ids returns ordered list of passage_ids."""
        answer = GeneratedAnswer(
            answer_text="The answer.",
            citations=[
                Citation(
                    passage_id=sample_passages[0].passage_id,
                    document_name=sample_passages[0].document_name,
                    location=sample_passages[0].location,
                ),
                Citation(
                    passage_id=sample_passages[1].passage_id,
                    document_name=sample_passages[1].document_name,
                    location=sample_passages[1].location,
                ),
            ],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="test",
            is_abstention=False,
            generation_latency_seconds=0.1,
        )
        ids = citation_passage_ids(answer)
        assert len(ids) == 2
        assert ids[0] == sample_passages[0].passage_id
        assert ids[1] == sample_passages[1].passage_id

    def test_has_valid_citations_true(self, sample_passages: list[RetrievedPassage]) -> None:
        """has_valid_citations returns True when citations exist."""
        answer = GeneratedAnswer(
            answer_text="The answer.",
            citations=[
                Citation(
                    passage_id=sample_passages[0].passage_id,
                    document_name=sample_passages[0].document_name,
                    location=sample_passages[0].location,
                ),
            ],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="test",
            is_abstention=False,
            generation_latency_seconds=0.1,
        )
        assert has_valid_citations(answer) is True

    def test_has_valid_citations_false(self) -> None:
        """has_valid_citations returns False when no citations."""
        answer = GeneratedAnswer(
            answer_text="No citations.",
            citations=[],
            evidence_quality=EvidenceQuality.INSUFFICIENT,
            evidence_quality_narrative="No passages.",
            is_abstention=True,
            generation_latency_seconds=0.1,
        )
        assert has_valid_citations(answer) is False


class TestResolveCitationsIntegrationWithFinalize:
    """Story 4.6 AC: resolve_citations composes with finalize_generation."""

    def test_finalize_then_resolve(self, sample_passages: list[RetrievedPassage]) -> None:
        """finalize_generation followed by resolve_citations produces correct output."""
        ctx = RetrievedContext(
            query="test",
            scope={"mode": "all", "document_id": None},
            passages=sample_passages,
        )
        # Start with LLM-output-style answer (wrong metadata, correct flags)
        answer = GeneratedAnswer(
            answer_text="The answer.",
            citations=[
                Citation(
                    passage_id=sample_passages[0].passage_id,
                    document_name="WRONG_NAME",
                    location="WRONG_LOC",
                ),
                Citation(
                    passage_id="UNKNOWN",
                    document_name="UNKNOWN",
                    location="UNKNOWN",
                ),
            ],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="test",
            is_abstention=True,  # LLM incorrectly set this
            generation_latency_seconds=0.3,
        )
        # Step 1: finalize (fixes abstention flag)
        finalized = finalize_generation(answer, ctx)
        assert finalized.is_abstention is False  # SUFFICIENT → False
        assert len(finalized.citations) == 2  # both still there

        # Step 2: resolve (drops unknown, fixes metadata)
        resolved = resolve_citations(finalized, ctx)
        assert len(resolved.citations) == 1  # unknown dropped
        assert resolved.citations[0].passage_id == sample_passages[0].passage_id
        assert resolved.citations[0].document_name == sample_passages[0].document_name
        assert resolved.citations[0].location == sample_passages[0].location
        # excerpt is resolved from the trusted RetrievedPassage text
        assert resolved.citations[0].excerpt == sample_passages[0].text
        assert resolved.answer_text == "The answer."
        assert resolved.evidence_quality == EvidenceQuality.SUFFICIENT

    def test_resolve_then_finalize(self, sample_passages: list[RetrievedPassage]) -> None:
        """resolve_citations followed by finalize_generation produces correct output."""
        ctx = RetrievedContext(
            query="test",
            scope={"mode": "all", "document_id": None},
            passages=sample_passages,
        )
        answer = GeneratedAnswer(
            answer_text="The answer.",
            citations=[
                Citation(
                    passage_id=sample_passages[0].passage_id,
                    document_name=sample_passages[0].document_name,
                    location=sample_passages[0].location,
                ),
            ],
            evidence_quality=EvidenceQuality.INSUFFICIENT,
            evidence_quality_narrative="No relevant passages.",
            is_abstention=False,  # needs to be set to True
            generation_latency_seconds=0.2,
        )
        # Step 1: resolve (validates citations against context)
        resolved = resolve_citations(answer, ctx)
        assert len(resolved.citations) == 1
        assert resolved.citations[0].passage_id == sample_passages[0].passage_id

        # Step 2: finalize (sets abstention flag)
        finalized = finalize_generation(resolved, ctx)
        assert finalized.is_abstention is True  # INSUFFICIENT → True
        assert finalized.citations[0].passage_id == sample_passages[0].passage_id
        assert finalized.answer_text == "The answer."
        assert finalized.evidence_quality == EvidenceQuality.INSUFFICIENT


class TestResolveCitationsDoesNotBreakExisting:
    """Verify Story 4.6 does not break Stories 4.1–4.5 behavior."""

    def test_evidence_quality_enum_unchanged(self) -> None:
        assert EvidenceQuality.SUFFICIENT == "sufficient"
        assert EvidenceQuality.PARTIAL == "partial"
        assert EvidenceQuality.INSUFFICIENT == "insufficient"
        assert EvidenceQuality.CONFLICTING == "conflicting"

    def test_generated_answer_fields_unchanged(self) -> None:
        answer = GeneratedAnswer(
            answer_text="test",
            citations=[],
            evidence_quality=EvidenceQuality.SUFFICIENT,
            evidence_quality_narrative="test",
            is_abstention=False,
            generation_latency_seconds=0.0,
        )
        assert answer.answer_text == "test"
        assert answer.citations == []

    def test_citation_excerpt_still_empty(self) -> None:
        cit = Citation(
            passage_id="p1",
            document_name="Doc",
            location="Page 1",
        )
        assert cit.excerpt == ""

    def test_parse_llm_output_still_works(self) -> None:
        import json

        from app.generation.schema import parse_llm_output
        raw = json.dumps({
            "answer_text": "test answer",
            "citations": [],
            "evidence_quality": "sufficient",
            "evidence_quality_narrative": "test",
            "is_abstention": False,
        })
        result = parse_llm_output(raw)
        assert result is not None
        assert result.answer_text == "test answer"

    def test_build_prompt_still_produces_all_sections(self) -> None:
        from app.generation.prompt import RetrievedContext, RetrievedPassage, build_prompt
        ctx = RetrievedContext(
            query="test question",
            scope={"mode": "all", "document_id": None},
            passages=[RetrievedPassage("p1", "Doc", "Page 1", "passage text", 0.9)],
        )
        prompt = build_prompt(ctx)
        assert "## Role and Task" in prompt
        assert "## User's Question" in prompt
        assert "## Retrieved Passages" in prompt
        assert "## Document Scope" in prompt
        assert "## Conversation Context" in prompt
        assert "Answer instructions:" in prompt
        assert "Evidence quality assessment:" in prompt
        assert "Output format:" in prompt

    def test_finalize_generation_preserves_answer_text(self) -> None:
        from app.generation.post_process import finalize_generation
        original_text = "The documents do not contain enough information."
        answer = GeneratedAnswer(
            answer_text=original_text,
            citations=[],
            evidence_quality=EvidenceQuality.INSUFFICIENT,
            evidence_quality_narrative="No relevant passages.",
            is_abstention=False,
            generation_latency_seconds=0.1,
        )
        result = finalize_generation(answer)
        assert result.answer_text == original_text

