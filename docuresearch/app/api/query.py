"""Query and conversation endpoints — Stories 7.3 and 5.2 (implementation plan §5.4, 5.6, 5.7).

Handlers are thin: they validate input, call the pipeline or the conversation
service, and shape the response. Grounding, citation validation, and
abstention all happen in the domain layer.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime

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
    UsageCountModel,
    UsageResponse,
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
from app.limits import (
    DailyLimits,
    QuestionLimitReached,
    Reservation,
    UsageCount,
    refund_user_question,
    reserve_question,
    usage_summary,
)
from app.pipeline import GenerationError, PipelineResult, ProviderAllowanceExhausted
from app.store.schema import get_connection, set_db_path

router = APIRouter(tags=["query"])

LIMIT_REACHED = "Daily question limit reached"
ALLOWANCE_EXHAUSTED = "The demo has used today's free AI allowance"


def _limits(state: AppState) -> DailyLimits:
    return DailyLimits(
        per_user=state.settings.questions_per_user_per_day,
        per_site=state.settings.questions_per_site_per_day,
    )


class _QuestionCounter:
    """The ``before_model_call`` hook that counts one question (Story 9.1).

    It opens its own short-lived connection, so ``BEGIN IMMEDIATE`` never
    collides with a transaction open on the request's connection. With limits
    off it does nothing (nothing is counted).
    """

    def __init__(self, state: AppState, user: User) -> None:
        self._db_path = state.pipeline.db_path
        self._user_id = user.id
        self._limits = _limits(state)
        self.reservation: Reservation | None = None

    def _connect(self) -> sqlite3.Connection:
        set_db_path(self._db_path)
        return get_connection()

    def __call__(self) -> None:
        if not self._limits.enabled:
            return
        conn = self._connect()
        try:
            self.reservation = reserve_question(conn, self._user_id, self._limits)
        finally:
            conn.close()

    def refund(self) -> None:
        """Refund the user after a failed model call (capped per day; site count kept)."""
        if self.reservation is None:
            return
        conn = self._connect()
        try:
            refund_user_question(conn, self.reservation.user_id, self.reservation.day)
        finally:
            conn.close()
        self.reservation = None


def _allowance_detail(resets_at: datetime) -> str:
    when = resets_at.strftime("%Y-%m-%d %H:%M UTC")
    return f"AI questions are paused until the provider's allowance resets at {when}."


def require_allowance(state: AppState) -> None:
    """503 while the LLM provider's daily allowance is exhausted (Story 9.2).

    Checked before retrieval and before counting, so nothing is called or counted.
    """
    allowance = getattr(state.pipeline, "llm_allowance", None)
    resets_at = allowance.exhausted_until() if allowance is not None else None
    if resets_at is not None:
        raise APIError(503, ALLOWANCE_EXHAUSTED, _allowance_detail(resets_at))


def _generation_failed(counter: _QuestionCounter, exc: GenerationError) -> APIError:
    """Refund the user (Story 9.1 failed-call rule) and map the failure to an API error.

    The request that discovers the provider's daily cap gets 503; others get 502.
    """
    counter.refund()
    if isinstance(exc, ProviderAllowanceExhausted):
        return APIError(503, ALLOWANCE_EXHAUSTED, _allowance_detail(exc.resets_at))
    return APIError(502, "Generation failed", str(exc))


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
    """Answer in a session. ``QuestionLimitReached`` raised by the counter
    hook propagates out of ``ask`` unchanged (no turn recorded) and becomes 429.
    An exhausted provider allowance gives 503 (Story 9.2).
    """
    require_llm(state)
    require_document(conn, scope.document_id if scope else None, owner_id=user.id)
    require_allowance(state)
    counter = _QuestionCounter(state, user)
    try:
        turn = ask(
            state.pipeline,
            question,
            session_id=session_id,
            scope=scope.to_domain() if scope else None,
            max_turns=state.settings.max_history_turns,
            owner_id=user.id,
            before_model_call=counter,
        )
    except SessionNotFoundError as exc:
        raise APIError(404, "Session not found", session_id) from exc
    except QuestionLimitReached as exc:
        raise APIError(429, LIMIT_REACHED, str(exc)) from exc
    except GenerationError as exc:
        raise _generation_failed(counter, exc) from exc
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
    429 when a daily question limit is reached (no model call, no turn
    recorded); 503 while the provider's daily AI allowance is used up (no
    model call, not counted); 502 when generation fails.
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
    require_allowance(state)
    counter = _QuestionCounter(state, user)
    try:
        result = state.pipeline.answer(
            req.question,
            document_id=scope.document_id,
            owner_id=user.id,
            before_model_call=counter,
        )
    except QuestionLimitReached as exc:
        raise APIError(429, LIMIT_REACHED, str(exc)) from exc
    except GenerationError as exc:
        raise _generation_failed(counter, exc) from exc
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
    """Ask a follow-up in one of the user's sessions; history and scope carry over.

    429 when a daily question limit is reached (no model call, no turn
    recorded); 503 while the provider's daily AI allowance is used up (no
    model call, not counted); 502 when generation fails.
    """
    start = time.monotonic()
    if get_session(conn, session_id, owner_id=user.id) is None:
        raise APIError(404, "Session not found", session_id)
    result, _ = _in_session(state, conn, user, req.question, session_id, req.document_scope)
    return FollowUpResponse(
        answer=_answer_model(result, time.monotonic() - start), session_id=session_id
    )


def _usage_count(count: UsageCount) -> UsageCountModel:
    return UsageCountModel(used=count.used, limit=count.limit, remaining=count.remaining)


@router.get("/usage", response_model=UsageResponse)
def usage(
    state: AppState = Depends(get_state),
    conn: sqlite3.Connection = Depends(get_conn),
    user: User = Depends(current_user),
) -> UsageResponse:
    """Today's question counts: the signed-in user's own, plus the site total."""
    summary = usage_summary(conn, user.id, _limits(state))
    return UsageResponse(
        day=summary.day,
        resets_at=summary.resets_at,
        user=_usage_count(summary.user),
        site=_usage_count(summary.site),
    )
