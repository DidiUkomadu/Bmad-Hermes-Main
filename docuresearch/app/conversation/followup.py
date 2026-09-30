"""Follow-up queries with conversation context — Story 5.2.

``ask`` runs one conversational turn: it loads the session's history and
document scope, answers through the normal ResearchPipeline (a full
retrieval + generation cycle with the same grounding, citation, and
abstention rules as a first question), then records the turn.

Conversation history is context, never a source (Architecture §7):
- It is shown to the model in the prompt, but citations are only accepted
  if they resolve to passages retrieved for *this* question, so a follow-up
  cannot be "supported" by an earlier answer alone.
- For retrieval, the previous user question is prepended to the follow-up so
  elliptical questions ("who receives it?") still find the right passages.
  The prompt carries the user's question verbatim.

Scope: a follow-up inherits the session's current document scope unless a
new scope is passed; the scope used is saved as the session's current scope.

The HTTP routes for this (implementation plan §5.4 and §5.6) belong to Epic 7
and call ``ask``.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.conversation.session import (
    DEFAULT_MAX_TURNS,
    SessionNotFoundError,
    add_turn,
    create_session,
    get_history,
    get_session,
    set_document_scope,
)
from app.generation.prompt import ConversationTurn
from app.pipeline import PipelineResult, ResearchPipeline
from app.retrieval.interface import DocumentScope
from app.store.schema import get_connection, set_db_path


@dataclass(frozen=True)
class ConversationAnswer:
    """The outcome of one conversational turn."""

    session_id: str
    turn: ConversationTurn
    result: PipelineResult
    scope: DocumentScope


def ask(
    pipeline: ResearchPipeline,
    query: str,
    session_id: str | None = None,
    scope: DocumentScope | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> ConversationAnswer:
    """Answer *query* within a conversation and record the turn.

    Args:
        pipeline: The research pipeline to answer with.
        query: The user's (follow-up) question.
        session_id: Existing session, or None to start a new one.
        scope: New document scope for this turn, or None to keep the
            session's current scope (all documents for a new session).
        max_turns: History limit used as prompt context and for pruning.

    Raises:
        SessionNotFoundError: if *session_id* does not exist.
        ValueError: if *scope* is malformed.
        app.pipeline.GenerationError: if generation fails; no turn is recorded.
    """
    if scope is not None:
        _validate_scope(scope)

    set_db_path(pipeline.db_path)
    conn = get_connection()
    try:
        if session_id is None:
            session = create_session(conn, scope=scope)
        else:
            session = get_session(conn, session_id)
            if session is None:
                raise SessionNotFoundError(session_id)

        effective = scope or session.current_document_scope or DocumentScope(mode="all")
        history = get_history(conn, session.session_id, max_turns=max_turns)

        result = pipeline.answer(
            query,
            document_id=effective.document_id if effective.mode == "specific" else None,
            conversation_history=history or None,
            retrieval_query=_retrieval_query(query, history),
        )

        turn = add_turn(conn, session.session_id, query, result.answer, max_turns=max_turns)
        set_document_scope(conn, session.session_id, effective)
    finally:
        conn.close()

    return ConversationAnswer(
        session_id=session.session_id, turn=turn, result=result, scope=effective
    )


def _retrieval_query(query: str, history: list[ConversationTurn]) -> str:
    """Expand a follow-up with the previous user question for retrieval."""
    if not history:
        return query
    return f"{history[-1].user_query} {query}"


def _validate_scope(scope: DocumentScope) -> None:
    if scope.mode == "all":
        return
    if scope.mode == "specific" and scope.document_id:
        return
    raise ValueError(
        f"Invalid document scope {scope!r}: mode must be 'all', or 'specific' with a document_id"
    )
