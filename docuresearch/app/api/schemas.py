"""API request/response models — Story 7.1 (implementation plan §5).

The HTTP contract for DocuResearch. Invariants (§5.8):
- No confidence scores anywhere.
- No authentication fields.
- The ``answer`` object mirrors the GeneratedAnswer domain model.
- Errors are ``{"error": <summary>, "detail": <explanation>}``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.retrieval.interface import DocumentScope

EvidenceQualityValue = Literal["sufficient", "partial", "insufficient", "conflicting"]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


class DocumentScopeModel(BaseModel):
    mode: Literal["all", "specific"] = "all"
    document_id: str | None = None

    @model_validator(mode="after")
    def _document_id_matches_mode(self) -> DocumentScopeModel:
        if self.mode == "specific" and not self.document_id:
            raise ValueError("document_id is required when mode is 'specific'")
        if self.mode == "all" and self.document_id:
            raise ValueError("document_id must be omitted when mode is 'all'")
        return self

    def to_domain(self) -> DocumentScope:
        return DocumentScope(mode=self.mode, document_id=self.document_id)


# ---------------------------------------------------------------------------
# Documents (§5.1–5.3)
# ---------------------------------------------------------------------------


class DocumentResponse(BaseModel):
    document_id: str
    name: str
    format: Literal["pdf", "markdown", "txt"]
    status: Literal["available"] = "available"
    uploaded_at: datetime
    page_count: int | None = None
    section_count: int | None = None


class UploadResponse(DocumentResponse):
    passage_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class RemovalResponse(BaseModel):
    document_id: str
    status: Literal["removed"] = "removed"


# ---------------------------------------------------------------------------
# Answers (§5.4, §5.6)
# ---------------------------------------------------------------------------


class CitationModel(BaseModel):
    passage_id: str
    document_name: str
    location: str
    excerpt: str


class AnswerModel(BaseModel):
    answer_text: str
    citations: list[CitationModel]
    evidence_quality: EvidenceQualityValue
    evidence_quality_narrative: str
    is_abstention: bool
    retrieval_latency_seconds: float
    generation_latency_seconds: float
    total_latency_seconds: float


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    document_scope: DocumentScopeModel | None = None
    session_id: str | None = None


class QueryResponse(BaseModel):
    answer: AnswerModel
    session_id: str | None = None


class FollowUpRequest(BaseModel):
    question: str = Field(min_length=1)
    document_scope: DocumentScopeModel | None = None


class FollowUpResponse(BaseModel):
    answer: AnswerModel
    session_id: str


class SessionCreateRequest(BaseModel):
    initial_document_scope: DocumentScopeModel | None = None


class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Citations (§5.5)
# ---------------------------------------------------------------------------


class PassageResponse(BaseModel):
    passage_id: str
    document_id: str
    document_name: str
    location: str
    text: str
    start_offset: int
    end_offset: int
