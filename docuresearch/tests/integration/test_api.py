"""API tests — Epic 7 (implementation plan §11.8).

The app runs with a real store, ingestion, retrieval, and generation
post-processing; only the LLM and embedding model are deterministic fakes.
"""

from __future__ import annotations

import io
import json
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.config import LLMSettings, Settings
from app.main import create_app
from app.pipeline import ResearchPipeline
from tests.integration.fakes import DOCS, CiteLLM, HashEmbedding, ScriptedLLM, sign_up

API = "/api/v1"
Q001_LOCATION = "page:1:chunk:0"  # where evaluation question q-001's answer lives
MD_SECURITY = "Sample Technical Document / Security Requirements:chunk:1"


def _pid(document_id: str, location: str) -> str:
    """Passage ID for a location in an uploaded document (IDs are owner-scoped)."""
    return f"{document_id}:{location}"


@pytest.fixture
def make_client(tmp_path):
    """Client factory: make_client(llm) -> TestClient over a fresh store."""
    clients = []

    def make(llm=None):
        pipeline = ResearchPipeline(
            str(tmp_path / "api.db"), llm or CiteLLM([]), embedding_model=HashEmbedding()
        )
        client = TestClient(create_app(Settings(db_path=tmp_path / "api.db"), pipeline))
        client.__enter__()
        clients.append(client)
        sign_up(client)
        return client

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def _upload(client, filename, data=None, name=None):
    data = (DOCS / filename).read_bytes() if data is None else data
    form = {"name": name} if name else None
    return client.post(
        f"{API}/documents/upload", files={"file": (filename, data)}, data=form
    )


def _assert_error(resp, status, error):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body["error"] == error
    assert "detail" in body


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


# ---------------------------------------------------------------------------
# 7.2 — Upload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "fmt", "page_count"),
    [("sample_spec.pdf", "pdf", 3), ("sample_spec.md", "markdown", None),
     ("sample_document.txt", "txt", None)],
)
def test_upload_each_format_stores_document_and_passages(make_client, filename, fmt,
                                                         page_count):
    client = make_client()
    resp = _upload(client, filename)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["format"] == fmt
    assert body["status"] == "available"
    assert body["name"] == filename
    assert body["page_count"] == page_count
    assert body["passage_count"] > 0

    listed = client.get(f"{API}/documents").json()["documents"]
    assert [d["document_id"] for d in listed] == [body["document_id"]]


def test_upload_with_display_name(make_client):
    client = make_client()
    body = _upload(client, "sample_spec.pdf", name="Alpha Protocol Spec").json()
    assert body["name"] == "Alpha Protocol Spec"
    assert body["document_id"].startswith("sample-spec-pdf-")  # IDs still from filename
    pid = _pid(body["document_id"], Q001_LOCATION)
    passage = client.get(f"{API}/citations/{quote(pid)}").json()
    assert passage["document_name"] == "Alpha Protocol Spec"


def test_upload_rejects_unsupported_format(make_client):
    client = make_client()
    _assert_error(_upload(client, "notes.docx", data=b"PK\x03\x04"), 400, "Invalid file format")
    assert client.get(f"{API}/documents").json()["documents"] == []


def test_upload_rejects_empty_file(make_client):
    _assert_error(_upload(make_client(), "empty.txt", data=b""), 400, "Empty file")


def test_upload_corrupt_pdf_fails_cleanly(make_client):
    client = make_client()
    _assert_error(_upload(client, "broken.pdf", data=b"%PDF-1.4 garbage"), 422,
                  "Extraction failed")
    assert client.get(f"{API}/documents").json()["documents"] == []


def test_upload_pdf_without_text_fails_cleanly(make_client):
    buf = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(buf)
    client = make_client()
    resp = _upload(client, "scan.pdf", data=buf.getvalue())
    _assert_error(resp, 422, "Extraction failed")
    assert "No extractable text" in resp.json()["detail"]
    assert client.get(f"{API}/documents").json()["documents"] == []


def test_duplicate_upload_conflicts(make_client):
    client = make_client()
    assert _upload(client, "sample_document.txt").status_code == 200
    _assert_error(_upload(client, "sample_document.txt"), 409, "Document already exists")
    assert len(client.get(f"{API}/documents").json()["documents"]) == 1


# ---------------------------------------------------------------------------
# 7.4 — Citation lookup
# ---------------------------------------------------------------------------


def test_citation_lookup_returns_stored_text(make_client):
    client = make_client()
    doc_id = _upload(client, "sample_spec.pdf").json()["document_id"]
    pid = _pid(doc_id, Q001_LOCATION)
    resp = client.get(f"{API}/citations/{quote(pid)}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["passage_id"] == pid
    assert body["document_id"] == doc_id
    assert body["document_name"] == "sample_spec.pdf"
    assert body["location"].startswith("page:1:")  # location carries the chunk suffix
    assert "AES-256" in body["text"]
    assert body["end_offset"] > body["start_offset"]


def test_citation_lookup_handles_ids_with_slashes_and_spaces(make_client):
    client = make_client()
    doc_id = _upload(client, "sample_spec.md").json()["document_id"]
    pid = _pid(doc_id, MD_SECURITY)
    resp = client.get(f"{API}/citations/{quote(pid)}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["passage_id"] == pid


def test_citation_lookup_unknown_passage(make_client):
    _assert_error(make_client().get(f"{API}/citations/nope:chunk:0"), 404, "Passage not found")


# ---------------------------------------------------------------------------
# 7.5 — List and remove
# ---------------------------------------------------------------------------


def test_remove_document_leaves_others(make_client):
    client = make_client()
    pdf = _upload(client, "sample_spec.pdf").json()
    txt = _upload(client, "sample_document.txt").json()

    resp = client.delete(f"{API}/documents/{pdf['document_id']}")
    assert resp.json() == {"document_id": pdf["document_id"], "status": "removed"}

    listed = client.get(f"{API}/documents").json()["documents"]
    assert [d["document_id"] for d in listed] == [txt["document_id"]]
    removed = _pid(pdf["document_id"], Q001_LOCATION)
    _assert_error(client.get(f"{API}/citations/{quote(removed)}"), 404, "Passage not found")
    other = _pid(txt["document_id"], "paragraph:5:chunk:4")
    assert client.get(f"{API}/citations/{quote(other)}").status_code == 200


def test_remove_unknown_document(make_client):
    _assert_error(make_client().delete(f"{API}/documents/missing"), 404, "Document not found")


# ---------------------------------------------------------------------------
# 7.3 — Query
# ---------------------------------------------------------------------------


def test_query_single_source_answer_matches_gold_and_resolves(make_client):
    """Evaluation question q-001, end to end: upload, query, look up the citation."""
    client = make_client(CiteLLM(["AES-256"]))
    doc_id = _upload(client, "sample_spec.pdf").json()["document_id"]

    resp = client.post(f"{API}/query", json={
        "question": "What encryption algorithm must be used for data transmissions "
                    "in the Alpha Protocol?",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    answer = body["answer"]
    assert body["session_id"] is None
    assert answer["evidence_quality"] == "sufficient"
    assert answer["is_abstention"] is False
    [citation] = answer["citations"]
    assert citation["passage_id"] == _pid(doc_id, Q001_LOCATION)
    assert citation["location"].startswith("page:1:")
    for key in ("retrieval_latency_seconds", "generation_latency_seconds",
                "total_latency_seconds"):
        assert answer[key] >= 0
    assert answer["total_latency_seconds"] >= answer["retrieval_latency_seconds"]

    lookup = client.get(f"{API}/citations/{quote(citation['passage_id'])}").json()
    assert lookup["text"] == citation["excerpt"]
    assert not any("confidence" in k.lower() for k in _walk_keys(body))


def test_query_with_no_documents_abstains(make_client):
    resp = make_client(ScriptedLLM(lambda p: pytest.fail("LLM must not be called"))).post(
        f"{API}/query", json={"question": "What is the payload limit?"}
    )
    assert resp.status_code == 200
    answer = resp.json()["answer"]
    assert answer["is_abstention"] is True
    assert answer["evidence_quality"] == "insufficient"
    assert answer["citations"] == []


def test_query_scoped_to_one_document(make_client):
    llm = CiteLLM(["garbage collector"])
    client = make_client(llm)
    _upload(client, "sample_spec.pdf")
    txt = _upload(client, "sample_document.txt").json()

    resp = client.post(f"{API}/query", json={
        "question": "How is memory managed?",
        "document_scope": {"mode": "specific", "document_id": txt["document_id"]},
    })
    assert resp.status_code == 200
    assert "sample_spec.pdf" not in llm.prompts[0]
    assert resp.json()["answer"]["citations"][0]["document_name"] == "sample_document.txt"


def test_query_scope_to_unknown_document(make_client):
    resp = make_client().post(f"{API}/query", json={
        "question": "q", "document_scope": {"mode": "specific", "document_id": "missing"},
    })
    _assert_error(resp, 404, "Document not found")


@pytest.mark.parametrize("payload", [
    {"question": ""},
    {"question": "q", "document_scope": {"mode": "specific"}},
    {"question": "q", "document_scope": {"mode": "all", "document_id": "x"}},
    {"question": "q", "document_scope": {"mode": "some"}},
])
def test_query_validation_errors(make_client, payload):
    _assert_error(make_client().post(f"{API}/query", json=payload), 422, "Invalid request")


def test_query_generation_failure_returns_502(make_client):
    client = make_client(ScriptedLLM(lambda p: "not json"))
    _upload(client, "sample_spec.md")
    _assert_error(client.post(f"{API}/query", json={"question": "Encryption?"}), 502,
                  "Generation failed")


def test_query_without_llm_configured_returns_503(tmp_path):
    settings = Settings(db_path=tmp_path / "x.db", llm=LLMSettings())
    with TestClient(create_app(settings)) as client:
        assert client.get(f"{API}/health").json() == {
            "status": "ok", "llm_configured": False, "registration_open": True,
        }
        sign_up(client)
        _assert_error(client.post(f"{API}/query", json={"question": "q"}), 503,
                      "LLM backend not configured")
        assert client.get(f"{API}/documents").status_code == 200  # rest still works


# ---------------------------------------------------------------------------
# Conversation (5.2 over HTTP)
# ---------------------------------------------------------------------------


def test_conversation_follow_up_flow(make_client):
    llm = CiteLLM(["rotated every 90 days", "public key"])
    client = make_client(llm)
    md = _upload(client, "sample_spec.md").json()
    _upload(client, "sample_document.txt")

    created = client.post(f"{API}/conversation", json={
        "initial_document_scope": {"mode": "specific", "document_id": md["document_id"]},
    })
    assert created.status_code == 201
    session_id = created.json()["session_id"]

    first = client.post(f"{API}/query", json={
        "question": "How often must the encryption key be rotated?", "session_id": session_id,
    })
    assert first.status_code == 200, first.text
    assert first.json()["session_id"] == session_id

    follow = client.post(f"{API}/conversation/{session_id}/follow-up",
                         json={"question": "What must be distributed after each one?"})
    assert follow.status_code == 200, follow.text
    body = follow.json()
    assert body["session_id"] == session_id
    assert "public key" in body["answer"]["citations"][0]["excerpt"]
    assert body["answer"]["citations"][0]["document_name"] == "sample_spec.md"  # scope kept
    assert "User: How often must the encryption key be rotated?" in llm.prompts[1]


def test_conversation_unknown_session(make_client):
    client = make_client()
    _assert_error(client.post(f"{API}/conversation/missing/follow-up", json={"question": "q"}),
                  404, "Session not found")
    _assert_error(client.post(f"{API}/query", json={"question": "q", "session_id": "missing"}),
                  404, "Session not found")


def test_start_conversation_without_body(make_client):
    resp = make_client().post(f"{API}/conversation")
    assert resp.status_code == 201
    assert resp.json()["session_id"]


# ---------------------------------------------------------------------------
# 7.6 — Wiring
# ---------------------------------------------------------------------------


def test_openapi_matches_contract_and_has_no_confidence(make_client):
    spec = make_client().get("/openapi.json").json()
    assert set(spec["paths"]) == {
        f"{API}/auth/register", f"{API}/auth/login", f"{API}/auth/logout", f"{API}/auth/me",
        f"{API}/documents/upload", f"{API}/documents", f"{API}/documents/{{document_id}}",
        f"{API}/query", f"{API}/conversation", f"{API}/conversation/{{session_id}}",
        f"{API}/conversation/{{session_id}}/follow-up",
        f"{API}/citations/{{passage_id}}", f"{API}/health",
    }
    assert "confidence" not in json.dumps(spec).lower()


def test_shutdown_closes_llm_client(tmp_path, monkeypatch):
    from app.generation import openai_compat

    closed = []
    monkeypatch.setattr(openai_compat.OpenAICompatibleLLM, "close", lambda self: closed.append(1))
    settings = Settings(
        db_path=tmp_path / "s.db",
        llm=LLMSettings(base_url="http://llm.invalid/v1", model="m"),
    )
    with TestClient(create_app(settings)) as client:
        assert client.get(f"{API}/health").json()["llm_configured"] is True
    assert closed == [1]


def test_concurrent_requests_do_not_share_connections_across_threads(make_client):
    """Regression: the UI loads several citations at once (found in a browser check).

    The request-scoped connection is opened in the dependency and used by the
    handler, which FastAPI may run on different worker threads.
    """
    from concurrent.futures import ThreadPoolExecutor

    client = make_client()
    md = _upload(client, "sample_spec.md").json()["document_id"]
    txt = _upload(client, "sample_document.txt").json()["document_id"]
    ids = [
        _pid(md, "Sample Technical Document:chunk:0"),
        _pid(txt, "paragraph:5:chunk:4"),
        "never-uploaded:page:1:chunk:0",  # 404 is fine, 500 is not
    ]

    def lookup(i):
        return client.get(f"{API}/citations/{quote(ids[i % len(ids)])}").status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lookup, range(60)))
    assert set(statuses) <= {200, 404}, statuses
    assert statuses.count(200) == 40



# ---------------------------------------------------------------------------
# Reopening conversations
# ---------------------------------------------------------------------------


def _ask_in(client, session_id, question):
    resp = client.post(f"{API}/query", json={"question": question, "session_id": session_id})
    assert resp.status_code == 200, resp.text
    return resp.json()["answer"]


def test_conversations_can_be_listed_and_reopened_in_full(make_client):
    client = make_client(CiteLLM(["rotated every 90 days"] * 7 + ["garbage collector"]))
    _upload(client, "sample_spec.md")
    _upload(client, "sample_document.txt")

    first = client.post(f"{API}/conversation").json()["session_id"]
    answers = [_ask_in(client, first, f"Key rotation question {i}?") for i in range(7)]
    second = client.post(f"{API}/conversation").json()["session_id"]
    _ask_in(client, second, "How is memory managed?")
    client.post(f"{API}/conversation")  # empty session: not listed

    listed = client.get(f"{API}/conversation").json()["sessions"]
    assert [s["session_id"] for s in listed] == [second, first]
    assert listed[1]["title"] == "Key rotation question 0?"
    assert listed[1]["turn_count"] == 7

    detail = client.get(f"{API}/conversation/{first}").json()
    assert detail["session_id"] == first
    assert [t["question"] for t in detail["turns"]] == [f"Key rotation question {i}?"
                                                        for i in range(7)]  # all 7, not 5
    stored = detail["turns"][0]["answer"]
    assert stored["answer_text"] == answers[0]["answer_text"]
    assert stored["citations"] == answers[0]["citations"]  # incl. location_label, excerpt
    assert stored["evidence_quality_narrative"] == answers[0]["evidence_quality_narrative"]
    assert detail["document_scope"] == {"mode": "all", "document_id": None}


def test_reopened_conversation_continues_with_its_context(make_client):
    llm = CiteLLM(["rotated every 90 days", "garbage collector", "public key"])
    client = make_client(llm)
    _upload(client, "sample_spec.md")
    _upload(client, "sample_document.txt")

    first = client.post(f"{API}/conversation").json()["session_id"]
    _ask_in(client, first, "How often must the encryption key be rotated?")
    other = client.post(f"{API}/conversation").json()["session_id"]
    _ask_in(client, other, "How is memory managed?")

    # Reopen the first conversation and follow up: its own history is the context.
    client.get(f"{API}/conversation/{first}")
    resp = client.post(f"{API}/conversation/{first}/follow-up",
                       json={"question": "What must be distributed after each one?"})
    assert resp.status_code == 200, resp.text
    prompt = llm.prompts[-1]
    assert "User: How often must the encryption key be rotated?" in prompt
    assert "How is memory managed?" not in prompt
    assert "public key" in resp.json()["answer"]["citations"][0]["excerpt"]


def test_delete_conversation(make_client):
    client = make_client(CiteLLM(["AES-256"]))
    _upload(client, "sample_spec.md")
    session_id = client.post(f"{API}/conversation").json()["session_id"]
    _ask_in(client, session_id, "What encryption?")

    resp = client.delete(f"{API}/conversation/{session_id}")
    assert resp.json() == {"session_id": session_id, "status": "removed"}
    assert client.get(f"{API}/conversation").json()["sessions"] == []
    _assert_error(client.get(f"{API}/conversation/{session_id}"), 404, "Session not found")
    _assert_error(client.delete(f"{API}/conversation/{session_id}"), 404, "Session not found")
