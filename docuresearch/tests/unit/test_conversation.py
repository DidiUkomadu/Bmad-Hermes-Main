"""Tests for conversation history storage — Story 5.1."""

from __future__ import annotations

import uuid

import pytest

from app.conversation import (
    DEFAULT_MAX_TURNS,
    SessionNotFoundError,
    add_turn,
    create_session,
    delete_session,
    get_history,
    get_session,
    prune_history,
    set_document_scope,
)
from app.generation.interface import Citation, EvidenceQuality, GeneratedAnswer
from app.generation.prompt import RetrievedContext, build_prompt
from app.retrieval.interface import DocumentScope
from app.store.schema import create_schema, get_connection, set_db_path


@pytest.fixture
def conn(tmp_path):
    set_db_path(str(tmp_path / "conv.db"))
    c = get_connection()
    create_schema(c)
    yield c
    c.close()


def _answer(
    text: str = "Keys rotate every 90 days.",
    quality: EvidenceQuality = EvidenceQuality.SUFFICIENT,
    abstention: bool = False,
    citations: list[Citation] | None = None,
) -> GeneratedAnswer:
    if citations is None:
        citations = [
            Citation(
                passage_id="spec-pdf:page:1:chunk:0",
                document_name="sample_spec.pdf",
                location="page:1",
                excerpt="The encryption key must be rotated every 90 days.",
            )
        ]
    return GeneratedAnswer(
        answer_text=text,
        citations=citations,
        evidence_quality=quality,
        evidence_quality_narrative="narrative",
        is_abstention=abstention,
        generation_latency_seconds=0.1,
    )


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def test_session_has_stable_uuid_identifier(conn):
    session = create_session(conn)
    assert uuid.UUID(session.session_id).version == 4
    loaded = get_session(conn, session.session_id)
    assert loaded is not None
    assert loaded.session_id == session.session_id
    assert loaded.created_at == session.created_at


def test_unknown_session_returns_none(conn):
    assert get_session(conn, "does-not-exist") is None


def test_document_scope_round_trips(conn):
    scope = DocumentScope(mode="specific", document_id="doc-1")
    session = create_session(conn, scope=scope)
    assert get_session(conn, session.session_id).current_document_scope == scope

    set_document_scope(conn, session.session_id, DocumentScope(mode="all"))
    assert get_session(conn, session.session_id).current_document_scope == DocumentScope("all")

    set_document_scope(conn, session.session_id, None)
    assert get_session(conn, session.session_id).current_document_scope is None


def test_set_scope_on_unknown_session_raises(conn):
    with pytest.raises(SessionNotFoundError):
        set_document_scope(conn, "missing", DocumentScope(mode="all"))


# ---------------------------------------------------------------------------
# Turns
# ---------------------------------------------------------------------------


def test_add_and_retrieve_turn_preserves_full_answer(conn):
    session = create_session(conn)
    answer = _answer()
    stored = add_turn(conn, session.session_id, "How often are keys rotated?", answer)

    [turn] = get_history(conn, session.session_id)
    assert turn == stored
    assert turn.turn_index == 0
    assert turn.user_query == "How often are keys rotated?"
    assert turn.answer_text == answer.answer_text
    assert turn.citations == answer.citations  # includes the trusted excerpt
    assert turn.evidence_quality is EvidenceQuality.SUFFICIENT
    assert turn.is_abstention is False


def test_turns_returned_in_chronological_order(conn):
    session = create_session(conn)
    for i in range(3):
        add_turn(conn, session.session_id, f"q{i}", _answer(text=f"a{i}"))
    history = get_history(conn, session.session_id)
    assert [t.user_query for t in history] == ["q0", "q1", "q2"]
    assert [t.turn_index for t in history] == [0, 1, 2]
    assert history[0].created_at <= history[1].created_at <= history[2].created_at


def test_history_limited_to_newest_five_and_oldest_pruned(conn):
    session = create_session(conn)
    for i in range(DEFAULT_MAX_TURNS + 2):
        add_turn(conn, session.session_id, f"q{i}", _answer())

    history = get_history(conn, session.session_id, max_turns=100)
    assert DEFAULT_MAX_TURNS == 5
    assert [t.user_query for t in history] == ["q2", "q3", "q4", "q5", "q6"]
    stored_rows = conn.execute(
        "SELECT COUNT(*) FROM conversation_turns WHERE session_id = ?",
        (session.session_id,),
    ).fetchone()[0]
    assert stored_rows == 5  # pruned from storage, not just hidden


def test_turn_index_is_not_reused_after_pruning(conn):
    session = create_session(conn)
    for i in range(7):
        turn = add_turn(conn, session.session_id, f"q{i}", _answer(), max_turns=2)
    assert turn.turn_index == 6
    assert [t.turn_index for t in get_history(conn, session.session_id)] == [5, 6]


def test_get_history_max_turns_returns_newest_subset(conn):
    session = create_session(conn)
    for i in range(4):
        add_turn(conn, session.session_id, f"q{i}", _answer())
    assert [t.user_query for t in get_history(conn, session.session_id, max_turns=2)] == [
        "q2",
        "q3",
    ]


def test_prune_history_with_smaller_limit(conn):
    session = create_session(conn)
    for i in range(4):
        add_turn(conn, session.session_id, f"q{i}", _answer())
    assert prune_history(conn, session.session_id, max_turns=1) == 3
    assert [t.user_query for t in get_history(conn, session.session_id)] == ["q3"]


def test_invalid_max_turns_rejected(conn):
    session = create_session(conn)
    with pytest.raises(ValueError):
        add_turn(conn, session.session_id, "q", _answer(), max_turns=0)
    with pytest.raises(ValueError):
        get_history(conn, session.session_id, max_turns=0)


@pytest.mark.parametrize(
    ("quality", "abstention"),
    [
        (EvidenceQuality.SUFFICIENT, False),
        (EvidenceQuality.PARTIAL, False),
        (EvidenceQuality.INSUFFICIENT, True),
        (EvidenceQuality.CONFLICTING, False),
    ],
)
def test_evidence_quality_and_abstention_persist(conn, quality, abstention):
    session = create_session(conn)
    add_turn(conn, session.session_id, "q", _answer(quality=quality, abstention=abstention))
    [turn] = get_history(conn, session.session_id)
    assert turn.evidence_quality is quality
    assert turn.is_abstention is abstention


def test_abstention_turn_without_citations(conn):
    session = create_session(conn)
    add_turn(
        conn,
        session.session_id,
        "q",
        _answer(quality=EvidenceQuality.INSUFFICIENT, abstention=True, citations=[]),
    )
    assert get_history(conn, session.session_id)[0].citations == []


def test_multiple_citations_round_trip_in_order(conn):
    citations = [
        Citation("p-2", "b.md", "Intro", "second “quoted” excerpt — with unicode"),
        Citation("p-1", "a.pdf", "page:3", "first excerpt"),
    ]
    session = create_session(conn)
    add_turn(conn, session.session_id, "q", _answer(citations=citations))
    assert get_history(conn, session.session_id)[0].citations == citations


def test_sessions_are_isolated(conn):
    a = create_session(conn)
    b = create_session(conn)
    for i in range(6):
        add_turn(conn, a.session_id, f"a{i}", _answer())
    add_turn(conn, b.session_id, "b0", _answer())

    assert [t.user_query for t in get_history(conn, b.session_id)] == ["b0"]
    assert get_history(conn, b.session_id)[0].turn_index == 0
    assert len(get_history(conn, a.session_id)) == 5


def test_turn_operations_on_unknown_session_raise(conn):
    with pytest.raises(SessionNotFoundError):
        add_turn(conn, "missing", "q", _answer())
    with pytest.raises(SessionNotFoundError):
        get_history(conn, "missing")


def test_delete_session_cascades_to_turns(conn):
    session = create_session(conn)
    add_turn(conn, session.session_id, "q", _answer())
    assert delete_session(conn, session.session_id) is True
    assert get_session(conn, session.session_id) is None
    remaining = conn.execute("SELECT COUNT(*) FROM conversation_turns").fetchone()[0]
    assert remaining == 0
    assert delete_session(conn, session.session_id) is False


def test_history_persists_across_connections(conn):
    session = create_session(conn)
    add_turn(conn, session.session_id, "q", _answer())
    conn.close()

    fresh = get_connection()
    try:
        assert [t.user_query for t in get_history(fresh, session.session_id)] == ["q"]
    finally:
        fresh.close()


def test_stored_history_feeds_prompt_construction(conn):
    session = create_session(conn)
    add_turn(conn, session.session_id, "How often are keys rotated?", _answer())
    history = get_history(conn, session.session_id)

    prompt = build_prompt(
        RetrievedContext(
            query="And who receives the new public key?",
            scope={"mode": "all", "document_id": None},
            passages=[],
            conversation_history=history,
        )
    )
    assert "User: How often are keys rotated?" in prompt
    assert "Evidence quality: sufficient" in prompt
