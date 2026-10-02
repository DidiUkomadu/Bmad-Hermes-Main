"""Conversation history storage — Story 5.1.

SQLite-backed sessions and turns so follow-up questions (Story 5.2) can be
answered in context. Tables are defined in ``app.store.schema`` (implementation
plan §6.3–6.4); this module owns all reads and writes to them.

Design:
- A session has a stable UUID4 ``session_id`` and an optional current
  document scope (the scope of its most recent query).
- Each turn stores the user query and the full answer: text, citations
  (JSON, including the trusted passage excerpt), evidence quality, and the
  abstention flag.
- Every turn is kept, so a conversation can be reopened in full. The context
  given to the model is bounded instead: ``get_history`` returns the newest
  ``max_turns`` turns (default ``DEFAULT_MAX_TURNS`` = 5, Epics & Stories
  story 5.1 decision), so prompt context never grows unbounded.
  ``prune_history`` deletes old turns explicitly if storage must be bounded.
- ``turn_index`` is 0-based and keeps increasing after pruning, so indexes are
  never reused within a session.

Conversation history is context for the next query, never a fact source:
answers must still be grounded in retrieved passages (Architecture §7).

Like ``app.store.schema``, functions take an open connection and commit
their own writes.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.generation.interface import Citation, EvidenceQuality, GeneratedAnswer
from app.generation.prompt import ConversationTurn
from app.retrieval.interface import DocumentScope

DEFAULT_MAX_TURNS = 5
"""Turns (query + answer pairs) given to the model as conversation context."""


class SessionNotFoundError(KeyError):
    """Raised when an operation references a session that does not exist."""


@dataclass(frozen=True)
class Session:
    """A conversation session (implementation plan §3.7)."""

    session_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    current_document_scope: DocumentScope | None = None
    owner_id: str | None = None  # the user it belongs to (Epic 8); None for unowned


@dataclass(frozen=True)
class SessionSummary:
    """A session as listed for reopening: titled by its first question."""

    session_id: str
    created_at: datetime
    last_activity: datetime
    title: str
    turn_count: int


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def _owner_clause(owner_id: str | None) -> tuple[str, tuple[str, ...]]:
    """SQL fragment restricting conversations to an owner (no restriction for None)."""
    return ("", ()) if owner_id is None else (" AND owner_id = ?", (owner_id,))


def create_session(
    conn: sqlite3.Connection,
    scope: DocumentScope | None = None,
    owner_id: str | None = None,
) -> Session:
    """Create and persist a new session with a fresh UUID4 identifier."""
    session = Session(
        session_id=str(uuid.uuid4()), current_document_scope=scope, owner_id=owner_id
    )
    conn.execute(
        "INSERT INTO conversations (session_id, created_at, current_document_scope, owner_id) "
        "VALUES (?, ?, ?, ?)",
        (session.session_id, session.created_at.isoformat(), _scope_to_json(scope), owner_id),
    )
    conn.commit()
    return session


def get_session(
    conn: sqlite3.Connection, session_id: str, owner_id: str | None = None
) -> Session | None:
    """Return the session, or None if it does not exist (or is not *owner_id*'s)."""
    clause, args = _owner_clause(owner_id)
    row = conn.execute(
        "SELECT session_id, created_at, current_document_scope, owner_id "
        f"FROM conversations WHERE session_id = ?{clause}",
        (session_id, *args),
    ).fetchone()
    if row is None:
        return None
    return Session(
        session_id=row["session_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        current_document_scope=_scope_from_json(row["current_document_scope"]),
        owner_id=row["owner_id"],
    )


def set_document_scope(
    conn: sqlite3.Connection, session_id: str, scope: DocumentScope | None
) -> None:
    """Record the document scope of the session's most recent query."""
    cur = conn.execute(
        "UPDATE conversations SET current_document_scope = ? WHERE session_id = ?",
        (_scope_to_json(scope), session_id),
    )
    if cur.rowcount == 0:
        raise SessionNotFoundError(session_id)
    conn.commit()


def delete_session(
    conn: sqlite3.Connection, session_id: str, owner_id: str | None = None
) -> bool:
    """Delete a session and (via CASCADE) all its turns.

    Returns True if a session was deleted, False if it did not exist (or is
    not *owner_id*'s).
    """
    clause, args = _owner_clause(owner_id)
    cur = conn.execute(
        f"DELETE FROM conversations WHERE session_id = ?{clause}", (session_id, *args)
    )
    conn.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Turns
# ---------------------------------------------------------------------------


def list_sessions(
    conn: sqlite3.Connection, owner_id: str | None = None
) -> list[SessionSummary]:
    """Sessions with at least one turn (only *owner_id*'s when given), most recent first."""
    clause, args = ("", ()) if owner_id is None else (" WHERE c.owner_id = ?", (owner_id,))
    rows = conn.execute(
        "SELECT c.session_id, c.created_at, COUNT(t.id) AS turn_count, "
        "MAX(t.created_at) AS last_activity, "
        "(SELECT user_query FROM conversation_turns f WHERE f.session_id = c.session_id "
        " ORDER BY f.turn_index ASC LIMIT 1) AS title "
        "FROM conversations c JOIN conversation_turns t ON t.session_id = c.session_id"
        f"{clause} GROUP BY c.session_id ORDER BY last_activity DESC",
        args,
    ).fetchall()
    return [
        SessionSummary(
            session_id=r["session_id"],
            created_at=datetime.fromisoformat(r["created_at"]),
            last_activity=datetime.fromisoformat(r["last_activity"]),
            title=r["title"],
            turn_count=r["turn_count"],
        )
        for r in rows
    ]


def add_turn(
    conn: sqlite3.Connection,
    session_id: str,
    user_query: str,
    answer: GeneratedAnswer,
) -> ConversationTurn:
    """Append a turn to the session. All turns are kept (see module docstring).

    Args:
        conn: Open store connection.
        session_id: Existing session ID.
        user_query: The user's question for this turn.
        answer: The final answer returned to the user (after citation
            resolution), whose citations carry the trusted excerpts.

    Raises:
        SessionNotFoundError: if the session does not exist.
    """
    if get_session(conn, session_id) is None:
        raise SessionNotFoundError(session_id)

    row = conn.execute(
        "SELECT MAX(turn_index) AS last FROM conversation_turns WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    turn = ConversationTurn(
        turn_index=0 if row["last"] is None else row["last"] + 1,
        user_query=user_query,
        answer_text=answer.answer_text,
        citations=list(answer.citations),
        evidence_quality=answer.evidence_quality,
        is_abstention=answer.is_abstention,
        created_at=datetime.now(UTC),
        evidence_quality_narrative=answer.evidence_quality_narrative,
    )
    conn.execute(
        "INSERT INTO conversation_turns (session_id, turn_index, user_query, answer_text, "
        "citations, evidence_quality, is_abstention, created_at, evidence_quality_narrative) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session_id,
            turn.turn_index,
            turn.user_query,
            turn.answer_text,
            _citations_to_json(turn.citations),
            turn.evidence_quality.value,
            int(turn.is_abstention),
            turn.created_at.isoformat(),
            turn.evidence_quality_narrative,
        ),
    )
    conn.commit()
    return turn


def get_history(
    conn: sqlite3.Connection,
    session_id: str,
    max_turns: int | None = DEFAULT_MAX_TURNS,
) -> list[ConversationTurn]:
    """Return the newest *max_turns* turns (all turns if None), oldest first.

    Raises:
        SessionNotFoundError: if the session does not exist.
    """
    if max_turns is not None:
        _check_max_turns(max_turns)
    if get_session(conn, session_id) is None:
        raise SessionNotFoundError(session_id)

    rows = conn.execute(
        "SELECT * FROM ("
        "  SELECT * FROM conversation_turns WHERE session_id = ?"
        "  ORDER BY turn_index DESC LIMIT ?"
        ") ORDER BY turn_index ASC",
        (session_id, -1 if max_turns is None else max_turns),
    ).fetchall()
    return [_row_to_turn(r) for r in rows]


def prune_history(
    conn: sqlite3.Connection,
    session_id: str,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> int:
    """Delete all but the newest *max_turns* turns. Returns the number deleted."""
    _check_max_turns(max_turns)
    deleted = _prune(conn, session_id, max_turns)
    conn.commit()
    return deleted


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prune(conn: sqlite3.Connection, session_id: str, max_turns: int) -> int:
    cur = conn.execute(
        "DELETE FROM conversation_turns WHERE session_id = ? AND turn_index NOT IN ("
        "  SELECT turn_index FROM conversation_turns WHERE session_id = ?"
        "  ORDER BY turn_index DESC LIMIT ?"
        ")",
        (session_id, session_id, max_turns),
    )
    return cur.rowcount


def _check_max_turns(max_turns: int) -> None:
    if max_turns < 1:
        raise ValueError(f"max_turns must be >= 1, got {max_turns}")


def _row_to_turn(row: sqlite3.Row) -> ConversationTurn:
    return ConversationTurn(
        turn_index=row["turn_index"],
        user_query=row["user_query"],
        answer_text=row["answer_text"],
        citations=_citations_from_json(row["citations"]),
        evidence_quality=EvidenceQuality(row["evidence_quality"]),
        is_abstention=bool(row["is_abstention"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        evidence_quality_narrative=row["evidence_quality_narrative"],
    )


def _citations_to_json(citations: list[Citation]) -> str:
    return json.dumps(
        [
            {
                "passage_id": c.passage_id,
                "document_name": c.document_name,
                "location": c.location,
                "excerpt": c.excerpt,
            }
            for c in citations
        ]
    )


def _citations_from_json(raw: str) -> list[Citation]:
    return [Citation(**c) for c in json.loads(raw)]


def _scope_to_json(scope: DocumentScope | None) -> str | None:
    if scope is None:
        return None
    return json.dumps({"mode": scope.mode, "document_id": scope.document_id})


def _scope_from_json(raw: str | None) -> DocumentScope | None:
    if raw is None:
        return None
    data = json.loads(raw)
    return DocumentScope(mode=data["mode"], document_id=data.get("document_id"))
