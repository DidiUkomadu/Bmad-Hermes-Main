"""Follow-up query tests — Story 5.2.

Multi-turn conversations over the evaluation documents, with a scripted LLM
that cites whichever retrieved passage contains a target phrase.
"""

from __future__ import annotations

import pytest

from app.conversation import SessionNotFoundError, ask, get_history, get_session
from app.generation.interface import EvidenceQuality
from app.generation.post_process import UNGROUNDED_ANSWER_TEXT
from app.pipeline import GenerationError, ResearchPipeline
from app.retrieval.interface import DocumentScope
from app.store.schema import get_connection
from tests.integration.fakes import (
    DOCS,
    CiteLLM,
    HashEmbedding,
    ScriptedLLM,
    answer_json,
    passage_ids_in,
)

HISTORY_IS_CONTEXT_ONLY = "The conversation history is context only, not a source."


@pytest.fixture
def docs(tmp_path):
    """Pipeline factory over the evaluation corpus; returns (pipeline, doc_ids)."""

    def make(llm):
        pipeline = ResearchPipeline(str(tmp_path / "f.db"), llm, embedding_model=HashEmbedding())
        ids = {
            name: pipeline.ingest(DOCS / name)["document_id"]
            for name in ("sample_spec.pdf", "sample_spec.md", "sample_document.txt")
        }
        return pipeline, ids

    return make


def _doc_names(answer) -> set[str]:
    return {p.document_name for p in answer.result.context.passages}


def test_multi_turn_follow_up_is_grounded_and_uses_history(docs):
    llm = CiteLLM(["rotated every 90 days", "public key"])
    pipeline, _ = docs(llm)

    first = ask(pipeline, "How often must the encryption key be rotated in the Alpha Protocol?")
    follow = ask(pipeline, "What must be distributed after each one?", session_id=first.session_id)

    # Both turns are full retrieval + generation cycles with the same output standards.
    for turn in (first, follow):
        assert turn.result.answer.evidence_quality is EvidenceQuality.SUFFICIENT
        assert turn.result.answer.citations
        assert turn.result.answer.citations[0].excerpt  # trusted passage text
    assert "public key" in follow.result.answer.citations[0].excerpt

    # The follow-up prompt carries the earlier turn as context, verbatim question,
    # and the context-only rule.
    follow_prompt = llm.prompts[1]
    assert "User: How often must the encryption key be rotated" in follow_prompt
    assert "## User's Question\n\nWhat must be distributed after each one?" in follow_prompt
    assert HISTORY_IS_CONTEXT_ONLY in follow_prompt
    assert HISTORY_IS_CONTEXT_ONLY not in llm.prompts[0]

    conn = get_connection()
    try:
        history = get_history(conn, first.session_id)
    finally:
        conn.close()
    assert [t.turn_index for t in history] == [0, 1]
    assert history[1].citations == follow.result.answer.citations


def test_follow_up_retrieval_is_expanded_with_previous_question(docs):
    llm = CiteLLM(["rotated", "rotated"])
    pipeline, _ = docs(llm)
    first = ask(pipeline, "How often must the encryption key be rotated?")
    follow = ask(pipeline, "Why?", session_id=first.session_id)

    # "Why?" alone has no searchable terms; the expanded query still ranks the
    # key-rotation passage first.
    assert "rotated" in follow.result.context.passages[0].text
    assert follow.result.context.query == "Why?"


def test_follow_up_inherits_document_scope(docs):
    llm = CiteLLM(["AES-256", "AES-256"])
    pipeline, ids = docs(llm)

    md_only = DocumentScope(mode="specific", document_id=ids["sample_spec.md"])
    first = ask(pipeline, "What encryption is required?", scope=md_only)
    follow = ask(pipeline, "Is that for all transmissions?", session_id=first.session_id)

    assert _doc_names(first) == {"sample_spec.md"}
    assert _doc_names(follow) == {"sample_spec.md"}
    assert follow.scope == md_only


def test_changed_scope_is_respected_and_saved(docs):
    llm = CiteLLM(["AES-256", "AES-256", "AES-256"])
    pipeline, ids = docs(llm)

    first = ask(
        pipeline,
        "What encryption is required?",
        scope=DocumentScope(mode="specific", document_id=ids["sample_spec.md"]),
    )
    widened = ask(
        pipeline, "What do all documents say?", session_id=first.session_id,
        scope=DocumentScope(mode="all"),
    )
    assert len(_doc_names(widened)) > 1

    conn = get_connection()
    try:
        assert get_session(conn, first.session_id).current_document_scope == DocumentScope("all")
    finally:
        conn.close()

    later = ask(pipeline, "And encryption again?", session_id=first.session_id)
    assert len(_doc_names(later)) > 1  # the widened scope is now the session's scope


def test_answer_supported_only_by_history_is_withheld(docs):
    """A follow-up cannot cite a passage it did not retrieve, even one from history."""
    pipeline, ids = docs(ScriptedLLM(lambda p: ""))
    txt_scope = DocumentScope(mode="specific", document_id=ids["sample_document.txt"])
    md_scope = DocumentScope(mode="specific", document_id=ids["sample_spec.md"])

    remembered: list[str] = []

    def respond(prompt: str) -> str:
        if not remembered:
            remembered.append(passage_ids_in(prompt)[0])
            return answer_json(remembered, text="Omega uses garbage collection.")
        # Second turn: re-cite the first turn's passage, which is out of scope now.
        return answer_json(remembered, text="As established, Omega uses garbage collection.")

    pipeline._llm = ScriptedLLM(respond)
    first = ask(pipeline, "How is memory managed?", scope=txt_scope)
    follow = ask(pipeline, "Does it free memory manually?", session_id=first.session_id,
                 scope=md_scope)

    assert first.result.answer.citations
    assert remembered[0] not in passage_ids_in(pipeline._llm.prompts[1])
    assert follow.result.answer.answer_text == UNGROUNDED_ANSWER_TEXT
    assert follow.result.answer.is_abstention is True
    assert follow.turn.evidence_quality is EvidenceQuality.INSUFFICIENT


def test_generation_failure_records_no_turn(docs):
    pipeline, _ = docs(CiteLLM(["AES-256"]))
    first = ask(pipeline, "What encryption is required?")

    pipeline._llm = ScriptedLLM(lambda p: "not json")
    with pytest.raises(GenerationError):
        ask(pipeline, "Anything else?", session_id=first.session_id)

    conn = get_connection()
    try:
        assert len(get_history(conn, first.session_id)) == 1
    finally:
        conn.close()


def test_history_in_prompt_respects_turn_limit(docs):
    llm = CiteLLM(["AES-256"] * 4)
    pipeline, _ = docs(llm)
    session_id = None
    for i in range(4):
        session_id = ask(pipeline, f"question {i} about encryption", session_id=session_id,
                         max_turns=2).session_id

    # Question 3 is asked with the newest two stored turns (1 and 2) as context.
    last_prompt = llm.prompts[-1]
    assert "User: question 1 about encryption" in last_prompt
    assert "User: question 2 about encryption" in last_prompt
    assert "User: question 0 about encryption" not in last_prompt


def test_unknown_session_raises(docs):
    pipeline, _ = docs(CiteLLM([]))
    with pytest.raises(SessionNotFoundError):
        ask(pipeline, "q", session_id="missing")


@pytest.mark.parametrize(
    "scope",
    [DocumentScope(mode="specific"), DocumentScope(mode="some", document_id="x")],
)
def test_invalid_scope_rejected(docs, scope):
    pipeline, _ = docs(CiteLLM([]))
    with pytest.raises(ValueError):
        ask(pipeline, "q", scope=scope)


# ---------------------------------------------------------------------------
# Follow-ups build on the earlier answer
# ---------------------------------------------------------------------------


def _small_candidate_pipeline(tmp_path, llm, max_candidates=1):
    pipeline = ResearchPipeline(str(tmp_path / "c.db"), llm, embedding_model=HashEmbedding(),
                                max_candidates=max_candidates)
    ids = {name: pipeline.ingest(DOCS / name)["document_id"]
           for name in ("sample_spec.md", "sample_document.txt")}
    return pipeline, ids


def test_follow_up_carries_forward_passage_cited_by_earlier_answer(tmp_path, monkeypatch):
    """The follow-up's own search misses the earlier evidence; it is carried forward.

    Query expansion alone would usually recover it, so it is switched off here to
    isolate the carry-forward mechanism.
    """
    monkeypatch.setattr("app.conversation.followup._retrieval_query", lambda q, h: q)
    llm = CiteLLM(["rotated every 90 days", "rotated every 90 days"])
    pipeline, _ = _small_candidate_pipeline(tmp_path, llm, max_candidates=1)

    first = ask(pipeline, "How often must the encryption key be rotated?")
    [cited] = first.result.answer.citations
    assert "rotated every 90 days" in cited.excerpt

    follow = ask(pipeline, "Tell me about garbage collection threads", session_id=first.session_id)
    ids = [p.passage_id for p in follow.result.context.passages]
    assert ids[0] == cited.passage_id  # carried forward, placed first
    assert len(ids) == 2 and ids[1] != cited.passage_id  # plus the follow-up's own result
    # The model could cite the earlier evidence, and the citation is accepted.
    assert follow.result.answer.citations[0].passage_id == cited.passage_id
    assert follow.result.answer.evidence_quality is EvidenceQuality.SUFFICIENT


def test_carried_passages_respect_new_scope_and_removed_documents(tmp_path):
    llm = CiteLLM(["rotated every 90 days", "x", "x"])
    pipeline, ids = _small_candidate_pipeline(tmp_path, llm, max_candidates=1)
    first = ask(pipeline, "How often must the encryption key be rotated?")
    cited = first.result.answer.citations[0].passage_id

    txt_only = DocumentScope(mode="specific", document_id=ids["sample_document.txt"])
    scoped = ask(pipeline, "And memory?", session_id=first.session_id, scope=txt_only)
    assert cited not in [p.passage_id for p in scoped.result.context.passages]

    conn = get_connection()
    try:
        from app.store.schema import delete_document
        delete_document(conn, ids["sample_spec.md"])
    finally:
        conn.close()
    after_removal = ask(pipeline, "Again?", session_id=first.session_id,
                        scope=DocumentScope(mode="all"))
    assert cited not in [p.passage_id for p in after_removal.result.context.passages]


def test_at_most_three_passages_are_carried_forward(tmp_path, monkeypatch):
    import app.conversation.followup as followup

    pipeline, _ = _small_candidate_pipeline(tmp_path, ScriptedLLM(lambda p: ""), max_candidates=20)
    # Every earlier answer cited many passages.
    many = passage_ids_in_store(pipeline)

    def cite_all(prompt):
        return answer_json(many)

    pipeline._llm = ScriptedLLM(cite_all)
    first = ask(pipeline, "Tell me everything")
    assert len(first.result.answer.citations) > followup.MAX_CARRIED_PASSAGES

    pipeline._ranking.configure(1)
    follow = ask(pipeline, "More?", session_id=first.session_id)
    assert len(follow.result.context.passages) <= followup.MAX_CARRIED_PASSAGES + 1


def passage_ids_in_store(pipeline):
    conn = get_connection()
    try:
        return [r["id"] for r in conn.execute("SELECT id FROM passages ORDER BY id")]
    finally:
        conn.close()


def test_follow_up_prompt_lists_sources_cited_by_earlier_answers(docs):
    llm = CiteLLM(["rotated every 90 days", "public key"])
    pipeline, _ = docs(llm)
    first = ask(pipeline, "How often must the encryption key be rotated?")
    ask(pipeline, "What must be distributed after each one?", session_id=first.session_id)

    cited = first.result.answer.citations[0]
    assert f"Sources cited: {cited.document_name}, {cited.location} " \
           f"(Passage ID: {cited.passage_id})" in llm.prompts[1]


def test_retrieval_query_uses_previous_question_and_answer():
    from datetime import UTC, datetime

    from app.conversation.followup import _retrieval_query
    from app.generation.prompt import ConversationTurn

    def turn(answer, quality):
        return ConversationTurn(0, "How often is the key rotated?", answer, [], quality, False,
                                datetime.now(UTC))

    real = [turn("Every 90 days, with a new key pair.", EvidenceQuality.SUFFICIENT)]
    assert _retrieval_query("Who receives it?", real) == (
        "How often is the key rotated? Every 90 days, with a new key pair. Who receives it?"
    )
    abstained = [turn("The documents do not contain enough information.",
                      EvidenceQuality.INSUFFICIENT)]
    assert _retrieval_query("Who receives it?", abstained) == (
        "How often is the key rotated? Who receives it?"
    )
    assert _retrieval_query("First question", []) == "First question"


def test_expanded_follow_up_search_finds_earlier_evidence_by_itself(tmp_path):
    """With a single candidate slot, the expanded query still ranks the earlier evidence."""
    llm = CiteLLM(["rotated every 90 days", "rotated every 90 days"])
    pipeline, _ = _small_candidate_pipeline(tmp_path, llm, max_candidates=1)
    first = ask(pipeline, "How often must the encryption key be rotated?")
    follow = ask(pipeline, "Tell me about garbage collection threads", session_id=first.session_id)
    assert [p.passage_id for p in follow.result.context.passages] == [
        first.result.answer.citations[0].passage_id
    ]
