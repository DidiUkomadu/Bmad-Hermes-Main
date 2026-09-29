"""Tests for the evaluation runner (Story 0.4).

Tests:
- Question loading from JSON
- Runner execution with StoreBackedSystem
- Metric computation (precision, recall, citation correctness, faithfulness,
  abstention, conflict handling)
- Result serialization
- Mock system for isolated runner testing
"""

from __future__ import annotations

import json

import pytest

from evaluation.runner import (
    EvaluationRunner,
    StoreBackedSystem,
    SystemInterface,
)
from evaluation.schema import (
    EvaluationQuestion,
    EvaluationResult,
    EvidenceQuality,
    GoldPassage,
    QuestionType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_questions():
    """A small set of EvaluationQuestions for testing."""
    return [
        EvaluationQuestion(
            id="q-test-001",
            text="What is AES-256?",
            type=QuestionType.SINGLE_SOURCE,
            gold_answer="AES-256 is an encryption algorithm.",
            gold_passages=[
                GoldPassage(
                    passage_id="doc-001:page:1:chunk:0",
                    document_id="doc-001",
                    document_name="test.pdf",
                    location="page:1",
                    claim="AES-256 is used for encryption.",
                )
            ],
        ),
        EvaluationQuestion(
            id="q-test-002",
            text="What is the key rotation period?",
            type=QuestionType.INSUFFICIENT_EVIDENCE,
            gold_answer="The documents do not contain information about the key rotation period.",
            gold_passages=[],
        ),
        EvaluationQuestion(
            id="q-test-003",
            text="Do protocols conflict?",
            type=QuestionType.CONFLICTING_EVIDENCE,
            gold_answer="Protocol A says X, Protocol B says Y.",
            gold_passages=[
                GoldPassage(
                    passage_id="doc-001:page:1:chunk:0",
                    document_id="doc-001",
                    document_name="test.pdf",
                    location="page:1",
                    claim="Protocol A says X.",
                ),
                GoldPassage(
                    passage_id="doc-002:page:1:chunk:0",
                    document_id="doc-002",
                    document_name="test2.pdf",
                    location="page:1",
                    claim="Protocol B says Y.",
                ),
            ],
            conflicting_sources=[
                {"source": "Protocol A Spec", "claim": "X"},
                {"source": "Protocol B Spec", "claim": "Y"},
            ],
        ),
    ]


class MockSystem(SystemInterface):
    """A mock system for testing the runner independently of the store."""

    def __init__(self, retrieval_results=None, generation_output=None):
        self.retrieval_results = retrieval_results or {}
        self.generation_output = generation_output or {}
        self.resolve_calls: list[str] = []

    def retrieve(self, query: str, scope_doc_id: str | None = None) -> list[dict]:
        return self.retrieval_results.get(query, [])

    def generate(
        self, query: str, passages: list[dict], scope_doc_id: str | None = None,
        conversation_history: list[dict] | None = None,
    ) -> dict:
        return self.generation_output.get(query, {
            "answer": "Mock answer.",
            "citations": [],
            "evidence_quality": "sufficient",
            "evidence_narrative": "Mock narrative.",
            "abstention": False,
        })

    def resolve_passage(self, passage_id: str) -> dict | None:
        self.resolve_calls.append(passage_id)
        return {
            "passage_id": passage_id,
            "document_id": "doc-001",
            "document_name": "test.pdf",
            "location": "page:1",
            "text": "Passage text for " + passage_id,
        }


# ---------------------------------------------------------------------------
# Tests: Question loading
# ---------------------------------------------------------------------------

class TestQuestionLoading:
    """Story 0.4: Dataset loading tests."""

    def test_loads_questions_from_json(self, tmp_path):
        questions_data = {
            "questions": [
                {
                    "id": "q-load-001",
                    "text": "Test question?",
                    "type": "single_source",
                    "gold_answer": "Test answer.",
                    "gold_passages": [],
                }
            ]
        }
        qs_file = tmp_path / "questions.json"
        qs_file.write_text(json.dumps(questions_data))

        runner = EvaluationRunner(questions_path=qs_file)
        questions = runner.load_questions()

        assert len(questions) == 1
        assert questions[0].id == "q-load-001"
        assert questions[0].text == "Test question?"
        assert questions[0].type.value == "single_source"

    def test_loads_all_question_types(self, tmp_path):
        questions_data = {
            "questions": [
                {
                    "id": f"q-type-{i}",
                    "text": f"Question {i}?",
                    "type": qtype,
                    "gold_answer": f"Answer {i}.",
                    "gold_passages": [],
                }
                for i, qtype in enumerate([
                    "single_source",
                    "multi_passage",
                    "partial_evidence",
                    "insufficient_evidence",
                    "conflicting_evidence",
                ])
            ]
        }
        qs_file = tmp_path / "questions.json"
        qs_file.write_text(json.dumps(questions_data))

        runner = EvaluationRunner(questions_path=qs_file)
        questions = runner.load_questions()

        assert len(questions) == 5
        types = {q.type for q in questions}
        assert len(types) == 5  # All 5 types represented

    def test_gold_passages_loaded(self, tmp_path):
        questions_data = {
            "questions": [
                {
                    "id": "q-gold-001",
                    "text": "Test?",
                    "type": "single_source",
                    "gold_answer": "Answer.",
                    "gold_passages": [
                        {
                            "passage_id": "passage-001",
                            "document_id": "doc-001",
                            "document_name": "test.pdf",
                            "location": "page:1",
                            "claim": "Claim text.",
                        }
                    ],
                }
            ]
        }
        qs_file = tmp_path / "questions.json"
        qs_file.write_text(json.dumps(questions_data))

        runner = EvaluationRunner(questions_path=qs_file)
        questions = runner.load_questions()

        assert len(questions[0].gold_passages) == 1
        gp = questions[0].gold_passages[0]
        assert gp.passage_id == "passage-001"
        assert gp.claim == "Claim text."


# ---------------------------------------------------------------------------
# Tests: Metric computation
# ---------------------------------------------------------------------------

class TestRetrievalMetrics:
    """Story 0.4: Retrieval precision and recall computation."""

    def test_precision_with_perfect_retrieval(self, sample_questions):
        """When all retrieved passages are relevant, precision = 1.0."""
        q = sample_questions[0]
        mock = MockSystem(
            retrieval_results={
                q.text: [
                    {"passage_id": "doc-001:page:1:chunk:0", "score": 0.9, "document_id": "doc-001"},
                ]
            },
            generation_output={
                q.text: {
                    "answer": "Answer.",
                    "citations": [],
                    "evidence_quality": "sufficient",
                    "evidence_narrative": "",
                    "abstention": False,
                }
            },
        )

        runner = EvaluationRunner()
        result = runner._run_one(q, mock)

        assert result.retrieval_precision == 1.0
        assert result.retrieval_recall == 1.0

    def test_precision_with_some_irrelevant(self, sample_questions):
        """When some retrieved passages are not relevant, precision < 1.0."""
        q = sample_questions[0]
        mock = MockSystem(
            retrieval_results={
                q.text: [
                    {"passage_id": "doc-001:page:1:chunk:0", "score": 0.9, "document_id": "doc-001"},
                    {"passage_id": "doc-001:page:2:chunk:0", "score": 0.3, "document_id": "doc-001"},  # not relevant
                ]
            },
            generation_output={
                q.text: {
                    "answer": "Answer.",
                    "citations": [],
                    "evidence_quality": "sufficient",
                    "evidence_narrative": "",
                    "abstention": False,
                }
            },
        )

        runner = EvaluationRunner()
        result = runner._run_one(q, mock)

        # 1 relevant out of 2 retrieved = 0.5 precision
        assert result.retrieval_precision == 0.5

    def test_recall_missing_relevant_passage(self, sample_questions):
        """When a gold passage is not retrieved, recall < 1.0."""
        q = sample_questions[0]
        mock = MockSystem(
            retrieval_results={
                q.text: []  # Nothing retrieved
            },
            generation_output={
                q.text: {
                    "answer": "Answer.",
                    "citations": [],
                    "evidence_quality": "sufficient",
                    "evidence_narrative": "",
                    "abstention": False,
                }
            },
        )

        runner = EvaluationRunner()
        result = runner._run_one(q, mock)

        assert result.retrieval_recall == 0.0

    def test_recall_insufficient_evidence_no_gold(self, sample_questions):
        """For insufficient-evidence questions with no gold passages,
        recall is N/A (None) regardless of what was retrieved."""
        q = sample_questions[1]  # insufficient_evidence
        mock = MockSystem(
            retrieval_results={
                q.text: []  # Nothing retrieved -- correct
            },
            generation_output={
                q.text: {
                    "answer": "I don't know.",
                    "citations": [],
                    "evidence_quality": "insufficient",
                    "evidence_narrative": "",
                    "abstention": True,
                }
            },
        )

        runner = EvaluationRunner()
        result = runner._run_one(q, mock)

        assert result.retrieval_recall is None  # N/A: no gold passages

    def test_precision_none_for_no_gold_passages(self, sample_questions):
        """For insufficient-evidence questions, precision is None (N/A)."""
        q = sample_questions[1]
        mock = MockSystem(
            retrieval_results={
                q.text: [{"passage_id": "doc-001:page:1:chunk:0", "score": 0.5, "document_id": "doc-001"}]
            },
            generation_output={
                q.text: {
                    "answer": "Answer.",
                    "citations": [],
                    "evidence_quality": "insufficient",
                    "evidence_narrative": "",
                    "abstention": True,
                }
            },
        )

        runner = EvaluationRunner()
        result = runner._run_one(q, mock)

        assert result.retrieval_precision is None

    def test_empty_retrieval_zero_precision(self, sample_questions):
        """When nothing is retrieved and there ARE gold passages,
        precision is 0.0."""
        q = sample_questions[0]
        mock = MockSystem(
            retrieval_results={q.text: []},
            generation_output={
                q.text: {
                    "answer": "No answer.",
                    "citations": [],
                    "evidence_quality": "insufficient",
                    "evidence_narrative": "",
                    "abstention": True,
                }
            },
        )

        runner = EvaluationRunner()
        result = runner._run_one(q, mock)

        assert result.retrieval_precision == 0.0


class TestCitationCorrectness:
    """Story 0.4: Citation correctness computation."""

    def test_all_citations_correct(self, sample_questions):
        q = sample_questions[0]
        mock = MockSystem(
            retrieval_results={q.text: []},
            generation_output={
                q.text: {
                    "answer": "Answer.",
                    "citations": [
                        {
                            "passage_id": "doc-001:page:1:chunk:0",
                            "document_name": "test.pdf",
                            "location": "page:1",
                            "excerpt": "AES-256 text.",
                        }
                    ],
                    "evidence_quality": "sufficient",
                    "evidence_narrative": "",
                    "abstention": False,
                }
            },
        )

        runner = EvaluationRunner()
        result = runner._run_one(q, mock)

        assert result.citation_correctness == 1.0

    def test_no_citations_na(self, sample_questions):
        q = sample_questions[0]
        mock = MockSystem(
            retrieval_results={q.text: []},
            generation_output={
                q.text: {
                    "answer": "Answer without citations.",
                    "citations": [],
                    "evidence_quality": "sufficient",
                    "evidence_narrative": "",
                    "abstention": False,
                }
            },
        )

        runner = EvaluationRunner()
        result = runner._run_one(q, mock)

        assert result.citation_correctness is None

    def test_partial_citations(self, sample_questions):
        q = sample_questions[0]
        mock = MockSystem(
            retrieval_results={q.text: []},
            generation_output={
                q.text: {
                    "answer": "Answer.",
                    "citations": [
                        {"passage_id": "doc-001:page:1:chunk:0", "document_name": "test.pdf", "location": "page:1", "excerpt": ""},
                        {"passage_id": "wrong-passage-id", "document_name": "test.pdf", "location": "page:2", "excerpt": ""},
                    ],
                    "evidence_quality": "sufficient",
                    "evidence_narrative": "",
                    "abstention": False,
                }
            },
        )

        runner = EvaluationRunner()
        result = runner._run_one(q, mock)

        assert result.citation_correctness == 0.5


class TestAbstentionAccuracy:
    """Story 0.4: Abstention accuracy computation."""

    def test_abstains_on_insufficient_evidence(self, sample_questions):
        q = sample_questions[1]  # insufficient_evidence
        mock = MockSystem(
            retrieval_results={q.text: []},
            generation_output={
                q.text: {
                    "answer": "I don't know.",
                    "citations": [],
                    "evidence_quality": "insufficient",
                    "evidence_narrative": "",
                    "abstention": True,
                }
            },
        )

        runner = EvaluationRunner()
        result = runner._run_one(q, mock)

        assert result.abstention_correct is True

    def test_false_abstention_on_sufficient_evidence(self, sample_questions):
        q = sample_questions[0]  # single_source (has sufficient evidence)
        mock = MockSystem(
            retrieval_results={q.text: []},
            generation_output={
                q.text: {
                    "answer": "I don't know.",
                    "citations": [],
                    "evidence_quality": "insufficient",
                    "evidence_narrative": "",
                    "abstention": True,
                }
            },
        )

        runner = EvaluationRunner()
        result = runner._run_one(q, mock)

        assert result.abstention_correct is False

    def test_correct_non_abstention(self, sample_questions):
        q = sample_questions[0]  # single_source
        mock = MockSystem(
            retrieval_results={q.text: []},
            generation_output={
                q.text: {
                    "answer": "The answer is AES-256.",
                    "citations": [],
                    "evidence_quality": "sufficient",
                    "evidence_narrative": "",
                    "abstention": False,
                }
            },
        )

        runner = EvaluationRunner()
        result = runner._run_one(q, mock)

        assert result.abstention_correct is True


class TestConflictHandling:
    """Story 0.4: Conflict handling computation."""

    def test_conflict_correctly_surfaced(self, sample_questions):
        q = sample_questions[2]  # conflicting_evidence
        mock = MockSystem(
            retrieval_results={q.text: []},
            generation_output={
                q.text: {
                    "answer": "Protocol A Spec says X. Protocol B Spec says Y.",
                    "citations": [],
                    "evidence_quality": "conflicting",
                    "evidence_narrative": "The documents disagree.",
                    "abstention": False,
                }
            },
        )

        runner = EvaluationRunner()
        result = runner._run_one(q, mock)

        # Both sources mentioned, evidence quality is conflicting
        assert result.conflict_handled is True

    def test_conflict_not_surfaced(self, sample_questions):
        q = sample_questions[2]
        mock = MockSystem(
            retrieval_results={q.text: []},
            generation_output={
                q.text: {
                    "answer": "The answer is X, as per Protocol A.",
                    "citations": [],
                    "evidence_quality": "sufficient",
                    "evidence_narrative": "",
                    "abstention": False,
                }
            },
        )

        runner = EvaluationRunner()
        result = runner._run_one(q, mock)

        assert result.conflict_handled is False

    def test_na_for_non_conflicting_questions(self, sample_questions):
        q = sample_questions[0]  # single_source
        mock = MockSystem(
            retrieval_results={q.text: []},
            generation_output={
                q.text: {
                    "answer": "Answer.",
                    "citations": [],
                    "evidence_quality": "sufficient",
                    "evidence_narrative": "",
                    "abstention": False,
                }
            },
        )

        runner = EvaluationRunner()
        result = runner._run_one(q, mock)

        assert result.conflict_handled is None


# ---------------------------------------------------------------------------
# Tests: Result serialization
# ---------------------------------------------------------------------------

class TestResultSerialization:
    """Story 0.4: Result serialization tests."""

    def test_result_to_dict(self):
        result = EvaluationResult(
            question_id="q-001",
            system_answer="The answer is X.",
            system_citations=[
                {"passage_id": "p-001", "document_name": "doc.pdf", "location": "page:1", "excerpt": "text"}
            ],
            system_evidence_quality=EvidenceQuality.SUFFICIENT,
            system_evidence_narrative="Supported directly.",
            system_abstention=False,
            retrieval_precision=0.8,
            retrieval_recall=1.0,
            citation_correctness=1.0,
            answer_faithful=True,
            abstention_correct=True,
            conflict_handled=None,
            retrieval_latency_s=0.001,
            generation_latency_s=0.05,
            total_latency_s=0.051,
            run_timestamp="2026-09-21T12:00:00",
        )

        d = result.to_dict()
        assert d["question_id"] == "q-001"
        assert d["system_answer"] == "The answer is X."
        assert d["system_evidence_quality"] == "sufficient"
        assert d["system_abstention"] is False
        assert d["retrieval_precision"] == 0.8
        assert d["citation_correctness"] == 1.0

    def test_result_summary_line(self):
        result = EvaluationResult(
            question_id="q-summary-001",
            system_answer="Answer.",
            system_citations=[],
            system_evidence_quality=EvidenceQuality.SUFFICIENT,
            system_evidence_narrative="",
            system_abstention=False,
            retrieval_precision=0.75,
            retrieval_recall=1.0,
            citation_correctness=0.5,
            answer_faithful=True,
            abstention_correct=True,
            total_latency_s=0.042,
        )

        line = result.summary_line()
        assert "q-summary-001" in line
        assert "faith=True" in line
        assert "abs=True" in line
        assert "prec=0.75" in line
        assert "rec=1.00" in line


class TestStoreBackedSystemFixes:
    """Regression tests for the document name and location fixes.

    Issue B: StoreBackedSystem.generate() was displaying 'unknown' for
    locations because retrieve() did not include location in the result
    dict, and sqlite3.Row does not support dict.get() in Python 3.12.
    """

    @pytest.fixture
    def db_with_docs(self, tmp_path):
        """Set up a SQLite DB with the three evaluation documents."""
        from pathlib import Path

        from app.ingestion.markdown import parse_markdown
        from app.ingestion.pdf import extract_text_from_pdf
        from app.ingestion.text import parse_plain_text
        from app.models import DocumentFormat, DocumentMeta, Passage, generate_passage_id
        from app.store.schema import (
            create_document,
            create_passage,
            create_schema,
            get_connection,
            set_db_path,
        )

        db_path = str(tmp_path / "test.db")
        set_db_path(db_path)
        conn = get_connection()
        create_schema(conn)

        docs_dir = Path(__file__).resolve().parent.parent.parent / "evaluation" / "dataset" / "documents"

        def store_raw(raw, doc_name, fmt, count=None):
            doc = DocumentMeta(
                id=raw.document_id, name=doc_name, format=fmt,
                uploaded_at=raw.uploaded_at,
                page_count=count if fmt == DocumentFormat.PDF else None,
                section_count=count if fmt == DocumentFormat.MARKDOWN else None,
            )
            create_document(conn, doc)
            for i, u in enumerate(raw.units):
                if not u["text"]:
                    continue
                pid = generate_passage_id(raw.document_id, u["location"], i)
                passage = Passage(
                    id=pid, document_id=raw.document_id,
                    text=u["text"], location=u["location"],
                    start_offset=u["start_offset"], end_offset=u["end_offset"],
                )
                create_passage(conn, passage)

        store_raw(
            extract_text_from_pdf((docs_dir / "sample_spec.pdf").read_bytes(), "sample_spec.pdf"),
            "sample_spec.pdf", DocumentFormat.PDF, 3,
        )
        store_raw(
            parse_markdown((docs_dir / "sample_spec.md").read_bytes(), "sample_spec.md"),
            "sample_spec.md", DocumentFormat.MARKDOWN, 7,
        )
        store_raw(
            parse_plain_text((docs_dir / "sample_document.txt").read_bytes(), "sample_document.txt"),
            "sample_document.txt", DocumentFormat.TXT, 9,
        )
        return db_path

    def test_retrieve_returns_location(self, db_with_docs):
        """retrieve() must include location in each result dict."""
        system = StoreBackedSystem(db_with_docs)
        results = system.retrieve("Alpha Protocol encryption")
        assert len(results) > 0
        for r in results:
            assert "location" in r, f"Passage {r.get('passage_id')} missing location key"
            assert isinstance(r["location"], str)
            assert len(r["location"]) > 0

    def test_retrieve_returns_document_name_via_get_document_name(self, db_with_docs):
        """_get_document_name() must resolve document names correctly."""
        system = StoreBackedSystem(db_with_docs)
        assert system._get_document_name("sample-spec-pdf-d58d38e8") == "sample_spec.pdf"
        assert system._get_document_name("sample-spec-md-b0f9ddb6") == "sample_spec.md"
        assert system._get_document_name("sample-document-txt-27c030ba") == "sample_document.txt"

    def test_generate_citations_include_location(self, db_with_docs):
        """generate() citations must include a non-empty location string."""
        system = StoreBackedSystem(db_with_docs)
        passages = system.retrieve("Alpha Protocol")
        output = system.generate(query="What is the Alpha Protocol?", passages=passages)
        citations = output["citations"]
        assert len(citations) > 0
        for cit in citations:
            loc = cit.get("location", "")
            assert loc and loc != "unknown", (
                f"Citation for {cit.get('passage_id')} has location={loc!r} — "
                "location must not be 'unknown'. Fix: retrieve() must include "
                "location in result dicts, and sqlite3.Row does not support dict.get()."
            )

    def test_generate_citations_include_document_name(self, db_with_docs):
        """generate() citations must include the actual document name, not a UUID."""
        system = StoreBackedSystem(db_with_docs)
        passages = system.retrieve("Alpha Protocol")
        output = system.generate(query="What is the Alpha Protocol?", passages=passages)
        citations = output["citations"]
        assert len(citations) > 0
        for cit in citations:
            doc_name = cit.get("document_name", "")
            assert doc_name and doc_name != "unknown", (
                f"Citation for {cit.get('passage_id')} has document_name={doc_name!r} — "
                "document name must not be 'unknown'."
            )
            # Should not be a raw UUID
            assert "-" not in doc_name or ".pdf" in doc_name or ".md" in doc_name or ".txt" in doc_name, (
                f"Citation document_name={doc_name!r} looks like a raw ID, not a document name."
            )



class TestEquivalentPassagesAndConflictCitations:
    """Equivalent gold passages and citation-based conflict surfacing."""

    @staticmethod
    def _question_with_equivalent() -> EvaluationQuestion:
        return EvaluationQuestion(
            id="q-eq",
            text="What encryption?",
            type=QuestionType.SINGLE_SOURCE,
            gold_answer="AES-256.",
            gold_passages=[
                GoldPassage(
                    passage_id="spec-pdf:page:1:chunk:0",
                    document_id="spec-pdf",
                    document_name="spec.pdf",
                    location="page:1",
                    claim="AES-256.",
                    equivalent_passage_ids=["spec-md:Security:chunk:1"],
                )
            ],
        )

    def test_equivalent_passage_counts_for_retrieval_and_citation(self):
        q = self._question_with_equivalent()
        mock = MockSystem(
            retrieval_results={q.text: [{"passage_id": "spec-md:Security:chunk:1", "score": 1.0}]},
            generation_output={
                q.text: {
                    "answer": "AES-256.",
                    "citations": [{"passage_id": "spec-md:Security:chunk:1"}],
                    "evidence_quality": "sufficient",
                    "evidence_narrative": "",
                    "abstention": False,
                }
            },
        )
        result = EvaluationRunner()._run_one(q, mock)
        assert result.retrieval_precision == 1.0
        assert result.retrieval_recall == 1.0
        assert result.citation_correctness == 1.0

    def test_retrieving_both_copies_counts_gold_once(self):
        q = self._question_with_equivalent()
        mock = MockSystem(retrieval_results={q.text: [
            {"passage_id": "spec-pdf:page:1:chunk:0", "score": 1.0},
            {"passage_id": "spec-md:Security:chunk:1", "score": 0.9},
        ]})
        result = EvaluationRunner()._run_one(q, mock)
        assert result.retrieval_recall == 1.0
        assert result.retrieval_precision == 1.0

    def test_conflict_surfaced_via_citations_to_both_documents(self, sample_questions):
        q = sample_questions[2]
        q.conflicting_sources = [
            {"source": "Protocol A Spec", "document_name": "test.pdf", "claim": "X"},
            {"source": "Protocol B Spec", "document_name": "test2.pdf", "claim": "Y"},
        ]
        mock = MockSystem(
            retrieval_results={q.text: []},
            generation_output={
                q.text: {
                    "answer": "One document says X; the other says Y.",
                    "citations": [
                        {"passage_id": "doc-001:page:1:chunk:0", "document_name": "test.pdf"},
                        {"passage_id": "doc-002:page:1:chunk:0", "document_name": "test2.pdf"},
                    ],
                    "evidence_quality": "conflicting",
                    "evidence_narrative": "The documents disagree.",
                    "abstention": False,
                }
            },
        )
        assert EvaluationRunner()._run_one(q, mock).conflict_handled is True

    def test_conflict_not_surfaced_when_one_document_missing(self, sample_questions):
        q = sample_questions[2]
        q.conflicting_sources = [
            {"source": "Protocol A Spec", "document_name": "test.pdf", "claim": "X"},
            {"source": "Protocol B Spec", "document_name": "test2.pdf", "claim": "Y"},
        ]
        mock = MockSystem(
            retrieval_results={q.text: []},
            generation_output={
                q.text: {
                    "answer": "It is X.",
                    "citations": [{"passage_id": "doc-001:page:1:chunk:0", "document_name": "test.pdf"}],
                    "evidence_quality": "conflicting",
                    "evidence_narrative": "",
                    "abstention": False,
                }
            },
        )
        assert EvaluationRunner()._run_one(q, mock).conflict_handled is False
