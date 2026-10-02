"""Follow-up queries with conversation context — Story 5.2.

``ask`` runs one conversational turn: it loads the session's history and
document scope, answers through the normal ResearchPipeline (a full
retrieval + generation cycle with the same grounding, citation, and
abstention rules as a first question), then records the turn.

Follow-ups build on what the conversation established, while staying grounded
in the documents (Architecture §7: history is context, never a source):
- Earlier questions and answers, with the sources each answer cited, are shown
  to the model in the prompt.
- Retrieval is expanded with the previous question and answer, so elliptical
  follow-ups ("who receives it?") find the right passages. The prompt carries
  the user's question verbatim.
- Passages cited by recent answers are carried forward into this turn's
  retrieved context (up to ``MAX_CARRIED_PASSAGES``, newest turn first, within
  the current document scope), so the model sees the same evidence the earlier
  answer relied on. They are re-read from the content store; passages of
  removed documents are skipped.
- Citations are still only accepted if they resolve to passages in *this*
  turn's context, so an earlier answer's text alone can never support a claim.

Scope: a follow-up inherits the session's current document scope unless a
new scope is passed; the scope used is saved as the session's current scope.

The HTTP routes for this (implementation plan §5.4 and §5.6) belong to Epic 7
and call ``ask``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.citation import resolve_passage
from app.conversation.session import (
    DEFAULT_MAX_TURNS,
    SessionNotFoundError,
    add_turn,
    create_session,
    get_history,
    get_session,
    set_document_scope,
)
from app.generation.interface import EvidenceQuality
from app.generation.prompt import ConversationTurn, RetrievedContext, RetrievedPassage
from app.pipeline import PipelineResult, ResearchPipeline
from app.retrieval.interface import DocumentScope
from app.store.schema import get_connection, set_db_path

MAX_CARRIED_PASSAGES = 3
"""Passages cited by earlier answers that are re-supplied to a follow-up."""

_ANSWER_CHARS_FOR_RETRIEVAL = 300


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
    owner_id: str | None = None,
) -> ConversationAnswer:
    """Answer *query* within a conversation and record the turn.

    Args:
        pipeline: The research pipeline to answer with.
        query: The user's (follow-up) question.
        session_id: Existing session, or None to start a new one.
        scope: New document scope for this turn, or None to keep the
            session's current scope (all documents for a new session).
        max_turns: How many recent turns are given to the model as context.
        owner_id: The signed-in user. The session must be theirs, and retrieval
            and carried-forward passages are limited to their documents.

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
            session = create_session(conn, scope=scope, owner_id=owner_id)
        else:
            session = get_session(conn, session_id, owner_id=owner_id)
            if session is None:
                raise SessionNotFoundError(session_id)

        effective = scope or session.current_document_scope or DocumentScope(mode="all")
        history = get_history(conn, session.session_id, max_turns=max_turns)

        document_id = effective.document_id if effective.mode == "specific" else None
        context = pipeline.retrieve(
            query,
            document_id=document_id,
            conversation_history=history or None,
            retrieval_query=_retrieval_query(query, history),
            owner_id=owner_id,
        )
        context = _carry_forward_cited_passages(conn, context, history, document_id, owner_id)
        result = PipelineResult(answer=pipeline.generate(context), context=context)

        turn = add_turn(conn, session.session_id, query, result.answer)
        set_document_scope(conn, session.session_id, effective)
    finally:
        conn.close()

    return ConversationAnswer(
        session_id=session.session_id, turn=turn, result=result, scope=effective
    )


def _retrieval_query(query: str, history: list[ConversationTurn]) -> str:
    """Expand a follow-up with the previous question (and answer) for retrieval.

    The previous answer is included only when it was a real answer; an
    abstention's boilerplate would add noise, not context.
    """
    if not history:
        return query
    last = history[-1]
    parts = [last.user_query]
    if last.evidence_quality != EvidenceQuality.INSUFFICIENT:
        parts.append(last.answer_text[:_ANSWER_CHARS_FOR_RETRIEVAL])
    parts.append(query)
    return " ".join(parts)


def _carry_forward_cited_passages(
    conn,
    context: RetrievedContext,
    history: list[ConversationTurn],
    document_id: str | None,
    owner_id: str | None = None,
) -> RetrievedContext:
    """Put passages cited by recent answers at the front of the context."""
    already = {p.passage_id for p in context.passages}
    top_score = max((p.score for p in context.passages), default=1.0)
    carried: list[RetrievedPassage] = []
    for turn in reversed(history):
        for citation in turn.citations:
            if len(carried) >= MAX_CARRIED_PASSAGES:
                break
            if citation.passage_id in already:
                continue
            stored = resolve_passage(conn, citation.passage_id, owner_id=owner_id)
            if stored is None:  # document removed since (or not this user's)
                continue
            if document_id and stored.document_id != document_id:
                continue  # outside the scope chosen for this turn
            already.add(stored.passage_id)
            carried.append(
                RetrievedPassage(
                    passage_id=stored.passage_id,
                    document_name=stored.document_name,
                    location=stored.location,
                    text=stored.text,
                    score=top_score,
                )
            )
    if not carried:
        return context
    return replace(context, passages=carried + list(context.passages))


def _validate_scope(scope: DocumentScope) -> None:
    if scope.mode == "all":
        return
    if scope.mode == "specific" and scope.document_id:
        return
    raise ValueError(
        f"Invalid document scope {scope!r}: mode must be 'all', or 'specific' with a document_id"
    )
