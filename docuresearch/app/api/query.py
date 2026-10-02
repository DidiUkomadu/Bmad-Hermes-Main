"""Query and conversation endpoints — Stories 7.3 and 5.2 (implementation plan §5.4, 5.6, 5.7).

Handlers are thin: they validate input, call the pipeline or the conversation
service, and shape the response. Grounding, citation validation, and
abstention all happen in the domain layer.
"""

from __future__ import annotations

import sqlite3
import time

from fastapi import APIRouter, Depends

from app.api.deps import (
    AppState,
    current_user,
    get_conn,
    get_state,
    require_document,
    require_llm,
)
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
    SessionDetailResponse,
    SessionListResponse,
    SessionRemovalResponse,
    SessionSummaryModel,
    StoredAnswerModel,
    TurnModel,
)
from app.auth import User
from app.citation import format_location
from app.conversation import (
    SessionNotFoundError,
    ask,
    create_session,
    delete_session,
    get_history,
    get_session,
    list_sessions,
)
from app.generation.interface import Citation
from app.pipeline import GenerationError, PipelineResult

router = APIRouter(tags=["query"])


def _citation_models(citations: list[Citation]) -> list[CitationModel]:
    return [
        CitationModel(
            passage_id=c.passage_id,
            document_name=c.document_name,
            location=c.location,
            location_label=format_location(c.location),
            excerpt=c.excerpt,
        )
        for c in citations
    ]


def _answer_model(result: PipelineResult, total_seconds: float) -> AnswerModel:
    answer = result.answer
    return AnswerModel(
        answer_text=answer.answer_text,
        citations=_citation_models(answer.citations),
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
    user: User,
    question: str,
    session_id: str,
    scope: DocumentScopeModel | None,
) -> tuple[PipelineResult, str]:
    require_llm(state)
    require_document(conn, scope.document_id if scope else None, owner_id=user.id)
    try:
        turn = ask(
            state.pipeline,
            question,
            session_id=session_id,
            scope=scope.to_domain() if scope else None,
            max_turns=state.settings.max_history_turns,
            owner_id=user.id,
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
    user: User = Depends(current_user),
) -> QueryResponse:
    """Answer a question from the user's documents.

    With ``session_id`` the turn is recorded in that (user's) session.
    """
    start = time.monotonic()
    if req.session_id:
        result, session_id = _in_session(
            state, conn, user, req.question, req.session_id, req.document_scope
        )
        return QueryResponse(
            answer=_answer_model(result, time.monotonic() - start), session_id=session_id
        )

    require_llm(state)
    scope = req.document_scope or DocumentScopeModel()
    require_document(conn, scope.document_id, owner_id=user.id)
    try:
        result = state.pipeline.answer(
            req.question, document_id=scope.document_id, owner_id=user.id
        )
    except GenerationError as exc:
        raise APIError(502, "Generation failed", str(exc)) from exc
    return QueryResponse(answer=_answer_model(result, time.monotonic() - start))


@router.post("/conversation", response_model=SessionCreateResponse, status_code=201)
def start_conversation(
    req: SessionCreateRequest | None = None,
    conn: sqlite3.Connection = Depends(get_conn),
    user: User = Depends(current_user),
) -> SessionCreateResponse:
    """Create a conversation session, optionally with an initial document scope."""
    scope = req.initial_document_scope if req else None
    require_document(conn, scope.document_id if scope else None, owner_id=user.id)
    session = create_session(
        conn, scope=scope.to_domain() if scope else None, owner_id=user.id
    )
    return SessionCreateResponse(session_id=session.session_id, created_at=session.created_at)


@router.get("/conversation", response_model=SessionListResponse)
def list_conversations(
    conn: sqlite3.Connection = Depends(get_conn),
    user: User = Depends(current_user),
) -> SessionListResponse:
    """The user's conversations with at least one question, most recently active first."""
    return SessionListResponse(
        sessions=[
            SessionSummaryModel(
                session_id=s.session_id,
                title=s.title,
                turn_count=s.turn_count,
                created_at=s.created_at,
                last_activity=s.last_activity,
            )
            for s in list_sessions(conn, owner_id=user.id)
        ]
    )


@router.get("/conversation/{session_id}", response_model=SessionDetailResponse)
def get_conversation(
    session_id: str,
    conn: sqlite3.Connection = Depends(get_conn),
    user: User = Depends(current_user),
) -> SessionDetailResponse:
    """One of the user's conversations with every recorded turn, oldest first."""
    session = get_session(conn, session_id, owner_id=user.id)
    if session is None:
        raise APIError(404, "Session not found", session_id)
    scope = session.current_document_scope
    return SessionDetailResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        document_scope=(
            DocumentScopeModel(mode=scope.mode, document_id=scope.document_id) if scope else None
        ),
        turns=[
            TurnModel(
                turn_index=t.turn_index,
                question=t.user_query,
                created_at=t.created_at,
                answer=StoredAnswerModel(
                    answer_text=t.answer_text,
                    citations=_citation_models(t.citations),
                    evidence_quality=t.evidence_quality.value,
                    evidence_quality_narrative=t.evidence_quality_narrative,
                    is_abstention=t.is_abstention,
                ),
            )
            for t in get_history(conn, session_id, max_turns=None)
        ],
    )


@router.delete("/conversation/{session_id}", response_model=SessionRemovalResponse)
def delete_conversation(
    session_id: str,
    conn: sqlite3.Connection = Depends(get_conn),
    user: User = Depends(current_user),
) -> SessionRemovalResponse:
    """Delete one of the user's conversations and all its turns."""
    if not delete_session(conn, session_id, owner_id=user.id):
        raise APIError(404, "Session not found", session_id)
    return SessionRemovalResponse(session_id=session_id)


@router.post("/conversation/{session_id}/follow-up", response_model=FollowUpResponse)
def follow_up(
    session_id: str,
    req: FollowUpRequest,
    state: AppState = Depends(get_state),
    conn: sqlite3.Connection = Depends(get_conn),
    user: User = Depends(current_user),
) -> FollowUpResponse:
    """Ask a follow-up in one of the user's sessions; history and scope carry over."""
    start = time.monotonic()
    if get_session(conn, session_id, owner_id=user.id) is None:
        raise APIError(404, "Session not found", session_id)
    result, _ = _in_session(state, conn, user, req.question, session_id, req.document_scope)
    return FollowUpResponse(
        answer=_answer_model(result, time.monotonic() - start), session_id=session_id
    )
