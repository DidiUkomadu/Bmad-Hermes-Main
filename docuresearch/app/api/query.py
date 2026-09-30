"""Query and conversation endpoints — Stories 7.3 and 5.2 (implementation plan §5.4, 5.6, 5.7).

Handlers are thin: they validate input, call the pipeline or the conversation
service, and shape the response. Grounding, citation validation, and
abstention all happen in the domain layer.
"""

from __future__ import annotations

import sqlite3
import time

from fastapi import APIRouter, Depends

from app.api.deps import AppState, get_conn, get_state, require_document, require_llm
from app.api.errors import APIError
from app.api.schemas import (
    AnswerModel,
    CitationModel,
    DocumentScopeModel,
    FollowUpRequest,
    FollowUpResponse,
    QueryRequest,
    QueryResponse,
    SessionCreateRequest,
    SessionCreateResponse,
)
from app.citation import format_location
from app.conversation import SessionNotFoundError, ask, create_session, get_session
from app.pipeline import GenerationError, PipelineResult

router = APIRouter(tags=["query"])


def _answer_model(result: PipelineResult, total_seconds: float) -> AnswerModel:
    answer = result.answer
    return AnswerModel(
        answer_text=answer.answer_text,
        citations=[
            CitationModel(
                passage_id=c.passage_id,
                document_name=c.document_name,
                location=c.location,
                location_label=format_location(c.location),
                excerpt=c.excerpt,
            )
            for c in answer.citations
        ],
        evidence_quality=answer.evidence_quality.value,
        evidence_quality_narrative=answer.evidence_quality_narrative,
        is_abstention=answer.is_abstention,
        retrieval_latency_seconds=result.context.retrieval_latency_seconds,
        generation_latency_seconds=answer.generation_latency_seconds,
        total_latency_seconds=total_seconds,
    )


def _in_session(
    state: AppState,
    conn: sqlite3.Connection,
    question: str,
    session_id: str,
    scope: DocumentScopeModel | None,
) -> tuple[PipelineResult, str]:
    require_llm(state)
    require_document(conn, scope.document_id if scope else None)
    try:
        turn = ask(
            state.pipeline,
            question,
            session_id=session_id,
            scope=scope.to_domain() if scope else None,
            max_turns=state.settings.max_history_turns,
        )
    except SessionNotFoundError as exc:
        raise APIError(404, "Session not found", session_id) from exc
    except GenerationError as exc:
        raise APIError(502, "Generation failed", str(exc)) from exc
    return turn.result, turn.session_id


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    state: AppState = Depends(get_state),
    conn: sqlite3.Connection = Depends(get_conn),
) -> QueryResponse:
    """Answer a question. With ``session_id`` the turn is recorded in that session."""
    start = time.monotonic()
    if req.session_id:
        result, session_id = _in_session(
            state, conn, req.question, req.session_id, req.document_scope
        )
        return QueryResponse(
            answer=_answer_model(result, time.monotonic() - start), session_id=session_id
        )

    require_llm(state)
    scope = req.document_scope or DocumentScopeModel()
    require_document(conn, scope.document_id)
    try:
        result = state.pipeline.answer(req.question, document_id=scope.document_id)
    except GenerationError as exc:
        raise APIError(502, "Generation failed", str(exc)) from exc
    return QueryResponse(answer=_answer_model(result, time.monotonic() - start))


@router.post("/conversation", response_model=SessionCreateResponse, status_code=201)
def start_conversation(
    req: SessionCreateRequest | None = None,
    conn: sqlite3.Connection = Depends(get_conn),
) -> SessionCreateResponse:
    """Create a conversation session, optionally with an initial document scope."""
    scope = req.initial_document_scope if req else None
    require_document(conn, scope.document_id if scope else None)
    session = create_session(conn, scope=scope.to_domain() if scope else None)
    return SessionCreateResponse(session_id=session.session_id, created_at=session.created_at)


@router.post("/conversation/{session_id}/follow-up", response_model=FollowUpResponse)
def follow_up(
    session_id: str,
    req: FollowUpRequest,
    state: AppState = Depends(get_state),
    conn: sqlite3.Connection = Depends(get_conn),
) -> FollowUpResponse:
    """Ask a follow-up in a session; history and scope carry over."""
    start = time.monotonic()
    if get_session(conn, session_id) is None:
        raise APIError(404, "Session not found", session_id)
    result, _ = _in_session(state, conn, req.question, session_id, req.document_scope)
    return FollowUpResponse(
        answer=_answer_model(result, time.monotonic() - start), session_id=session_id
    )
