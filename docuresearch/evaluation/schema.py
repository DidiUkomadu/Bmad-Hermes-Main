"""Evaluation data models (Stories 0.1, 0.3, 0.4).

Defines:
- EvaluationQuestion: a curated question with gold answer and source references.
- EvaluationResult: the output of running one question through the system.
- Metric names and types used across the evaluation framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class QuestionType(StrEnum):
    """The five question types from the evaluation methodology (§3)."""

    SINGLE_SOURCE = "single_source"
    MULTI_PASSAGE = "multi_passage"
    PARTIAL_EVIDENCE = "partial_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


class EvidenceQuality(StrEnum):
    """Evidence quality categories from the PRFAQ/architecture."""

    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"


@dataclass
class GoldPassage:
    """A passage that supports the gold answer for a question.

    Attributes:
        passage_id: The stable passage ID (as stored in the content store).
        document_id: The document this passage belongs to.
        document_name: Human-readable document name.
        location: Page, section, or paragraph locator.
        claim: What this passage supports (specific claim text).
        equivalent_passage_ids: Passages in other documents carrying the same
            content (e.g. the Markdown copy of a PDF spec). Retrieving or
            citing any of them counts as hitting this gold passage.
    """
    passage_id: str
    document_id: str
    document_name: str
    location: str
    claim: str
    equivalent_passage_ids: list[str] = field(default_factory=list)

    def accepted_ids(self) -> set[str]:
        """The gold passage ID plus all equivalent passage IDs."""
        return {self.passage_id, *self.equivalent_passage_ids}


@dataclass
class EvaluationQuestion:
    """A curated evaluation question with gold answer and source references.

    Attributes:
        id: Unique question identifier.
        text: The question text.
        type: One of the five QuestionType values.
        gold_answer: The expected fully-correct grounded answer.
        gold_passages: Passages that support the gold answer.
        closest_passage: For insufficient-evidence questions, the closest
            relevant passage (if one exists).
        conflicting_sources: For conflicting-evidence questions, the sources
            that conflict and what each says.
    """
    id: str
    text: str
    type: QuestionType
    gold_answer: str
    gold_passages: list[GoldPassage] = field(default_factory=list)
    closest_passage: GoldPassage | None = None
    conflicting_sources: list[dict[str, str]] = field(default_factory=list)
    # Metadata for dataset management
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "type": self.type.value,
            "gold_answer": self.gold_answer,
            "gold_passages": [
                {
                    "passage_id": gp.passage_id,
                    "document_id": gp.document_id,
                    "document_name": gp.document_name,
                    "location": gp.location,
                    "claim": gp.claim,
                    "equivalent_passage_ids": gp.equivalent_passage_ids,
                }
                for gp in self.gold_passages
            ],
            "closest_passage": (
                {
                    "passage_id": self.closest_passage.passage_id,
                    "document_id": self.closest_passage.document_id,
                    "document_name": self.closest_passage.document_name,
                    "location": self.closest_passage.location,
                    "claim": self.closest_passage.claim,
                }
                if self.closest_passage else None
            ),
            "conflicting_sources": self.conflicting_sources,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationQuestion:
        return cls(
            id=data["id"],
            text=data["text"],
            type=QuestionType(data["type"]),
            gold_answer=data["gold_answer"],
            gold_passages=[
                GoldPassage(**gp) for gp in data.get("gold_passages", [])
            ],
            closest_passage=(
                GoldPassage(**data["closest_passage"])
                if data.get("closest_passage") else None
            ),
            conflicting_sources=data.get("conflicting_sources", []),
            tags=data.get("tags", []),
        )


@dataclass
class EvaluationResult:
    """The result of running one evaluation question through the system.

    Attributes:
        question_id: The ID of the question that was run.
        system_answer: The answer text produced by the system.
        system_citations: List of {passage_id, document_name, location, excerpt}
            from the system's answer.
        system_evidence_quality: The evidence quality category the system reported.
        system_evidence_narrative: The system's evidence quality narrative text.
        system_abstention: Whether the system abstained (True = abstained).
        retrieval_precision: Computed retrieval precision for this question.
        retrieval_recall: Computed retrieval recall for this question.
        citation_correctness: Fraction of citations that are correct (0–1).
        answer_faithful: Boolean — was the answer faithful to retrieved passages?
        abstention_correct: Boolean — was the abstention decision correct?
        conflict_handled: Boolean — was the conflict correctly surfaced?
        retrieval_latency_s: Retrieval latency in seconds.
        generation_latency_s: Generation latency in seconds.
        total_latency_s: Total end-to-end latency in seconds.
        run_timestamp: When the evaluation run was performed.
        notes: Any additional notes (e.g. failure reasons).
    """
    question_id: str
    system_answer: str
    system_citations: list[dict[str, Any]] = field(default_factory=list)
    system_evidence_quality: EvidenceQuality | None = None
    system_evidence_narrative: str = ""
    system_abstention: bool = False
    retrieval_precision: float | None = None
    retrieval_recall: float | None = None
    citation_correctness: float | None = None
    answer_faithful: bool | None = None
    abstention_correct: bool | None = None
    conflict_handled: bool | None = None
    retrieval_latency_s: float = 0.0
    generation_latency_s: float = 0.0
    total_latency_s: float = 0.0
    run_timestamp: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "system_answer": self.system_answer,
            "system_citations": self.system_citations,
            "system_evidence_quality": (
                self.system_evidence_quality.value
                if self.system_evidence_quality else None
            ),
            "system_evidence_narrative": self.system_evidence_narrative,
            "system_abstention": self.system_abstention,
            "retrieval_precision": self.retrieval_precision,
            "retrieval_recall": self.retrieval_recall,
            "citation_correctness": self.citation_correctness,
            "answer_faithful": self.answer_faithful,
            "abstention_correct": self.abstention_correct,
            "conflict_handled": self.conflict_handled,
            "retrieval_latency_s": self.retrieval_latency_s,
            "generation_latency_s": self.generation_latency_s,
            "total_latency_s": self.total_latency_s,
            "run_timestamp": self.run_timestamp,
            "notes": self.notes,
        }

    def summary_line(self) -> str:
        """One-line summary for console output."""
        parts = [
            f"Q={self.question_id}",
            f"faith={self.answer_faithful}",
            f"abs={self.abstention_correct}",
            _fmt("prec", self.retrieval_precision),
            _fmt("rec", self.retrieval_recall),
            _fmt("cit", self.citation_correctness),
            f"lat={self.total_latency_s:.3f}s",
        ]
        return "  ".join(parts)


# Metric names for reporting
METRIC_NAMES = {
    "retrieval_precision": "Retrieval Precision",
    "retrieval_recall": "Retrieval Recall",
    "citation_correctness": "Citation Correctness",
    "answer_faithfulness": "Answer Faithfulness",
    "abstention_accuracy": "Abstention Accuracy",
    "conflict_handling": "Conflict Handling",
    "latency": "Response Latency",
}


def _fmt(label: str, value: float | None) -> str:
    """Format an optional 0–1 metric for summary lines."""
    return f"{label}={value:.2f}" if value is not None else f"{label}=N/A"
