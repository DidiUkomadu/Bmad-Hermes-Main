"""Multi-user tests — Epic 8 (change proposal §3).

Two users share one server and one database, each with their own browser
(TestClient, with its own cookie jar). Nothing belonging to one may reach the
other: not documents, passages, answers, citations, nor conversations.
"""

from __future__ import annotations

import re
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.pipeline import ResearchPipeline
from tests.integration.fakes import (
    DEFAULT_PASSWORD,
    DOCS,
    HashEmbedding,
    ScriptedLLM,
    answer_json,
    passage_ids_in,
    sign_up,
)

API = "/api/v1"
PUBLIC = {
    ("GET", f"{API}/health"),
    ("POST", f"{API}/auth/register"),
    ("POST", f"{API}/auth/login"),
    ("POST", f"{API}/auth/logout"),
}


class CiteAllLLM(ScriptedLLM):
    """Cites every passage it is shown, recording the prompts per call."""

    def __init__(self):
        super().__init__(lambda prompt: answer_json(passage_ids_in(prompt)))


@pytest.fixture
def server(tmp_path):
    """(app, pipeline, llm, make_browser) over one shared database."""
    llm = CiteAllLLM()
    pipeline = ResearchPipeline(str(tmp_path / "multi.db"), llm, embedding_model=HashEmbedding())
    settings = Settings(db_path=tmp_path / "multi.db")
    app = create_app(settings, pipeline)
    browsers = []

    def make_browser():
        client = TestClient(app)
        client.__enter__()
        browsers.append(client)
        return client

    yield app, pipeline, llm, make_browser
    for b in browsers:
        b.__exit__(None, None, None)


@pytest.fixture
def two_users(server):
    _, _, llm, make_browser = server
    alice, bob = make_browser(), make_browser()
    sign_up(alice, "alice@example.com", display_name="Alice")
    sign_up(bob, "bob@example.com", display_name="Bob")
    return alice, bob, llm


def _upload(client, filename):
    resp = client.post(f"{API}/documents/upload",
                       files={"file": (filename, (DOCS / filename).read_bytes())})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _error(resp, status):
    assert resp.status_code == status, resp.text
    return resp.json()["error"]


# ---------------------------------------------------------------------------
# Sign-in
# ---------------------------------------------------------------------------


def test_every_non_public_endpoint_requires_sign_in(server):
    app, _, _, make_browser = server
    anonymous = make_browser()
    bodies = {
        f"{API}/query": {"question": "q"},
        f"{API}/conversation/{{session_id}}/follow-up": {"question": "q"},
    }
    checked = 0
    for path, ops in app.openapi()["paths"].items():
        for method in ops:
            if (method.upper(), path) in PUBLIC:
                continue
            url = re.sub(r"\{[^}]+\}", "x", path)
            kwargs = {"json": bodies[path]} if path in bodies else {}
            if path.endswith("/upload"):
                kwargs = {"files": {"file": ("a.txt", b"hello")}}
            resp = anonymous.request(method.upper(), url, **kwargs)
            assert resp.status_code == 401, f"{method.upper()} {path} -> {resp.status_code}"
            assert resp.json()["error"] == "Not signed in"
            checked += 1
    assert checked >= 12  # documents, query, conversations, citations, me, usage


def test_register_sets_secure_session_cookie_and_me_works(server):
    _, _, _, make_browser = server
    browser = make_browser()
    resp = browser.post(f"{API}/auth/register",
                        json={"email": "Ada@Example.com", "password": DEFAULT_PASSWORD,
                              "display_name": "Ada"})
    assert resp.status_code == 201
    cookie = resp.headers["set-cookie"].lower()
    assert "docuresearch_session=" in cookie
    assert "httponly" in cookie
    assert "samesite=lax" in cookie
    assert "max-age=604800" in cookie  # 7 days
    me = browser.get(f"{API}/auth/me").json()
    assert me["email"] == "ada@example.com"
    assert me["display_name"] == "Ada"
    assert "password" not in str(me).lower()


def test_sign_out_ends_the_session_on_the_server(server):
    _, _, _, make_browser = server
    browser = make_browser()
    sign_up(browser)
    stolen = browser.cookies.get("docuresearch_session")
    assert browser.post(f"{API}/auth/logout").status_code == 204
    assert _error(browser.get(f"{API}/auth/me"), 401) == "Not signed in"

    replay = make_browser()  # someone reusing the old cookie value
    replay.cookies.set("docuresearch_session", stolen)
    assert _error(replay.get(f"{API}/documents"), 401) == "Not signed in"


def test_sign_in_after_sign_out(server):
    _, _, _, make_browser = server
    browser = make_browser()
    sign_up(browser, "ada@example.com")
    browser.post(f"{API}/auth/logout")
    resp = browser.post(f"{API}/auth/login",
                        json={"email": "ADA@example.com", "password": DEFAULT_PASSWORD})
    assert resp.status_code == 200
    assert browser.get(f"{API}/auth/me").json()["email"] == "ada@example.com"


def test_wrong_password_and_throttling(server):
    _, _, _, make_browser = server
    browser = make_browser()
    sign_up(browser, "ada@example.com")
    browser.post(f"{API}/auth/logout")
    for _ in range(5):
        resp = browser.post(f"{API}/auth/login",
                            json={"email": "ada@example.com", "password": "wrong password"})
        assert _error(resp, 401) == "Invalid email or password"
    # Even the right password is refused during the lockout.
    resp = browser.post(f"{API}/auth/login",
                        json={"email": "ada@example.com", "password": DEFAULT_PASSWORD})
    assert _error(resp, 429) == "Too many attempts"


def test_unknown_email_and_wrong_password_look_the_same(server):
    _, _, _, make_browser = server
    browser = make_browser()
    sign_up(browser, "ada@example.com")
    unknown = browser.post(f"{API}/auth/login",
                           json={"email": "nobody@example.com", "password": DEFAULT_PASSWORD})
    wrong = browser.post(f"{API}/auth/login",
                         json={"email": "ada@example.com", "password": "wrong password"})
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json() == wrong.json()


def test_registration_errors(server):
    _, _, _, make_browser = server
    browser = make_browser()
    sign_up(browser, "ada@example.com")
    other = make_browser()
    resp = other.post(f"{API}/auth/register",
                      json={"email": "ADA@example.com", "password": DEFAULT_PASSWORD})
    assert _error(resp, 409) == "Email already registered"
    resp = other.post(f"{API}/auth/register", json={"email": "new@example.com", "password": "short"})
    assert _error(resp, 400) == "Invalid registration"


def test_registration_can_be_closed(tmp_path):
    pipeline = ResearchPipeline(str(tmp_path / "c.db"), CiteAllLLM(),
                                embedding_model=HashEmbedding())
    app = create_app(Settings(db_path=tmp_path / "c.db", allow_registration=False), pipeline)
    with TestClient(app) as client:
        assert client.get(f"{API}/health").json()["registration_open"] is False
        resp = client.post(f"{API}/auth/register",
                           json={"email": "ada@example.com", "password": DEFAULT_PASSWORD})
        assert _error(resp, 403) == "Registration is closed"


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_documents_are_private(two_users):
    alice, bob, _ = two_users
    doc = _upload(alice, "sample_spec.pdf")

    assert bob.get(f"{API}/documents").json()["documents"] == []
    assert _error(bob.delete(f"{API}/documents/{doc['document_id']}"), 404) == "Document not found"
    assert [d["document_id"] for d in alice.get(f"{API}/documents").json()["documents"]] == [
        doc["document_id"]
    ]


def test_citations_of_other_users_are_not_found(two_users):
    alice, bob, _ = two_users
    doc = _upload(alice, "sample_spec.pdf")
    pid = quote(f"{doc['document_id']}:page:1:chunk:0")
    assert alice.get(f"{API}/citations/{pid}").status_code == 200
    assert _error(bob.get(f"{API}/citations/{pid}"), 404) == "Passage not found"


def test_answers_never_use_another_users_passages(two_users):
    alice, bob, llm = two_users
    alice_doc = _upload(alice, "sample_spec.pdf")

    # Bob has no documents: he gets an abstention, and the model is never called.
    calls_before = len(llm.prompts)
    answer = bob.post(f"{API}/query", json={"question": "What encryption is required?"}).json()
    assert answer["answer"]["is_abstention"] is True
    assert answer["answer"]["citations"] == []
    assert len(llm.prompts) == calls_before

    # Bob uploads his own document; his prompt contains only his passages.
    bob_doc = _upload(bob, "sample_document.txt")
    resp = bob.post(f"{API}/query", json={"question": "encryption memory protocol"})
    prompt = llm.prompts[-1]
    shown = passage_ids_in(prompt)
    assert shown and all(pid.startswith(bob_doc["document_id"]) for pid in shown)
    assert alice_doc["document_id"] not in prompt
    assert "AES-256" not in prompt  # Alice's content
    cited_docs = {c["document_name"] for c in resp.json()["answer"]["citations"]}
    assert cited_docs == {"sample_document.txt"}


def test_cannot_scope_a_question_to_another_users_document(two_users):
    alice, bob, _ = two_users
    doc = _upload(alice, "sample_spec.pdf")
    resp = bob.post(f"{API}/query", json={
        "question": "q", "document_scope": {"mode": "specific", "document_id": doc["document_id"]},
    })
    assert _error(resp, 404) == "Document not found"
    resp = bob.post(f"{API}/conversation", json={
        "initial_document_scope": {"mode": "specific", "document_id": doc["document_id"]},
    })
    assert _error(resp, 404) == "Document not found"


def test_same_filename_for_two_users(two_users):
    alice, bob, _ = two_users
    a = _upload(alice, "sample_spec.pdf")
    b = _upload(bob, "sample_spec.pdf")
    assert a["document_id"] != b["document_id"]
    assert bob.delete(f"{API}/documents/{b['document_id']}").status_code == 200
    assert len(alice.get(f"{API}/documents").json()["documents"]) == 1  # Alice's copy intact


def test_conversations_are_private(two_users):
    alice, bob, _ = two_users
    _upload(alice, "sample_spec.md")
    session_id = alice.post(f"{API}/conversation").json()["session_id"]
    alice.post(f"{API}/query", json={"question": "What encryption?", "session_id": session_id})

    assert bob.get(f"{API}/conversation").json()["sessions"] == []
    assert _error(bob.get(f"{API}/conversation/{session_id}"), 404) == "Session not found"
    assert _error(bob.delete(f"{API}/conversation/{session_id}"), 404) == "Session not found"
    resp = bob.post(f"{API}/conversation/{session_id}/follow-up", json={"question": "And?"})
    assert _error(resp, 404) == "Session not found"
    resp = bob.post(f"{API}/query", json={"question": "And?", "session_id": session_id})
    assert _error(resp, 404) == "Session not found"

    detail = alice.get(f"{API}/conversation/{session_id}").json()
    assert [t["question"] for t in detail["turns"]] == ["What encryption?"]  # untouched


def test_existing_single_user_data_goes_to_the_first_account(server):
    _, pipeline, _, make_browser = server
    pipeline.ingest(DOCS / "sample_spec.pdf")  # uploaded before accounts existed
    first, second = make_browser(), make_browser()
    sign_up(first, "owner@example.com")
    sign_up(second, "later@example.com")
    assert [d["name"] for d in first.get(f"{API}/documents").json()["documents"]] == [
        "sample_spec.pdf"
    ]
    assert second.get(f"{API}/documents").json()["documents"] == []
