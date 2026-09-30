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
- History is bounded: after each ``add_turn`` the session is pruned to the
  newest ``max_turns`` turns (default ``DEFAULT_MAX_TURNS`` = 5, oldest-first
  pruning, Epics & Stories story 5.1 decision). Pruned turns are deleted.
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
"""Maximum turns (query + answer pairs) kept per session."""


class SessionNotFoundError(KeyError):
    """Raised when an operation references a session that does not exist."""


@dataclass(frozen=True)
class Session:
    """A conversation session (implementation plan §3.7)."""

    session_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    current_document_scope: DocumentScope | None = None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def create_session(
    conn: sqlite3.Connection, scope: DocumentScope | None = None
) -> Session:
    """Create and persist a new session with a fresh UUID4 identifier."""
    session = Session(session_id=str(uuid.uuid4()), current_document_scope=scope)
    conn.execute(
        "INSERT INTO conversations (session_id, created_at, current_document_scope) "
        "VALUES (?, ?, ?)",
        (session.session_id, session.created_at.isoformat(), _scope_to_json(scope)),
    )
    conn.commit()
    return session


def get_session(conn: sqlite3.Connection, session_id: str) -> Session | None:
    """Return the session, or None if it does not exist."""
    row = conn.execute(
        "SELECT session_id, created_at, current_document_scope "
        "FROM conversations WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return Session(
        session_id=row["session_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        current_document_scope=_scope_from_json(row["current_document_scope"]),
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


def delete_session(conn: sqlite3.Connection, session_id: str) -> bool:
    """Delete a session and (via CASCADE) all its turns.

    Returns True if a session was deleted, False if it did not exist.
    """
    cur = conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    conn.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Turns
# ---------------------------------------------------------------------------


def add_turn(
    conn: sqlite3.Connection,
    session_id: str,
    user_query: str,
    answer: GeneratedAnswer,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> ConversationTurn:
    """Append a turn to the session, then prune it to the newest *max_turns*.

    Args:
        conn: Open store connection.
        session_id: Existing session ID.
        user_query: The user's question for this turn.
        answer: The final answer returned to the user (after citation
            resolution), whose citations carry the trusted excerpts.
        max_turns: History limit applied after inserting.

    Raises:
        SessionNotFoundError: if the session does not exist.
    """
    _check_max_turns(max_turns)
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
    )
    conn.execute(
        "INSERT INTO conversation_turns (session_id, turn_index, user_query, answer_text, "
        "citations, evidence_quality, is_abstention, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session_id,
            turn.turn_index,
            turn.user_query,
            turn.answer_text,
            _citations_to_json(turn.citations),
            turn.evidence_quality.value,
            int(turn.is_abstention),
            turn.created_at.isoformat(),
        ),
    )
    _prune(conn, session_id, max_turns)
    conn.commit()
    return turn


def get_history(
    conn: sqlite3.Connection,
    session_id: str,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> list[ConversationTurn]:
    """Return up to the newest *max_turns* turns, oldest first.

    Raises:
        SessionNotFoundError: if the session does not exist.
    """
    _check_max_turns(max_turns)
    if get_session(conn, session_id) is None:
        raise SessionNotFoundError(session_id)

    rows = conn.execute(
        "SELECT * FROM ("
        "  SELECT * FROM conversation_turns WHERE session_id = ?"
        "  ORDER BY turn_index DESC LIMIT ?"
        ") ORDER BY turn_index ASC",
        (session_id, max_turns),
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
