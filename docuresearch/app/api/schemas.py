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
    location_label: str  # human-readable location, e.g. "Page 2"
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


class SessionSummaryModel(BaseModel):
    session_id: str
    title: str  # the conversation's first question
    turn_count: int
    created_at: datetime
    last_activity: datetime


class SessionListResponse(BaseModel):
    sessions: list[SessionSummaryModel]


class StoredAnswerModel(BaseModel):
    """An answer as recorded in a conversation (latencies are not stored)."""

    answer_text: str
    citations: list[CitationModel]
    evidence_quality: EvidenceQualityValue
    evidence_quality_narrative: str
    is_abstention: bool


class TurnModel(BaseModel):
    turn_index: int
    question: str
    answer: StoredAnswerModel
    created_at: datetime


class SessionDetailResponse(BaseModel):
    session_id: str
    created_at: datetime
    document_scope: DocumentScopeModel | None
    turns: list[TurnModel]


class SessionRemovalResponse(BaseModel):
    session_id: str
    status: Literal["removed"] = "removed"


# ---------------------------------------------------------------------------
# Citations (§5.5)
# ---------------------------------------------------------------------------


class PassageResponse(BaseModel):
    passage_id: str
    document_id: str
    document_name: str
    location: str
    location_label: str
    text: str
    start_offset: int
    end_offset: int


# ---------------------------------------------------------------------------
# Accounts (Epic 8)
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=1024)
    display_name: str | None = Field(default=None, max_length=200)


class LoginRequest(BaseModel):
    email: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=1024)


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    created_at: datetime
