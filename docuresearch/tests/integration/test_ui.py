"""Citation UI tests — Epic 6 (stories 6.1, 6.2).

The UI renders API fields verbatim, so these tests check (a) the page and
assets are served, (b) the data the UI displays is correct for every format
and for multi-document answers, and (c) the script follows the display rules
(passage text from the content store, inserted as text, never as HTML).
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.pipeline import ResearchPipeline
from app.store.schema import get_connection, get_passage
from tests.integration.fakes import DOCS, HashEmbedding, ScriptedLLM, answer_json, passages_in

API = "/api/v1"
STATIC = Path(__file__).resolve().parents[2] / "app" / "static"


@pytest.fixture
def client_with(tmp_path):
    clients = []

    def make(llm, docs=("sample_spec.pdf", "sample_spec.md", "sample_document.txt")):
        pipeline = ResearchPipeline(str(tmp_path / "ui.db"), llm, embedding_model=HashEmbedding())
        client = TestClient(create_app(Settings(db_path=tmp_path / "ui.db"), pipeline))
        client.__enter__()
        clients.append(client)
        for name in docs:
            resp = client.post(f"{API}/documents/upload",
                               files={"file": (name, (DOCS / name).read_bytes())})
            assert resp.status_code == 200, resp.text
        return client

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def _cite_one_per_document(prompt: str) -> str:
    """Cite the top retrieved passage from each distinct document."""
    chosen, seen_docs = [], set()
    for pid, _ in passages_in(prompt):
        doc = pid.split(":", 1)[0]
        if doc not in seen_docs:
            seen_docs.add(doc)
            chosen.append(pid)
    return answer_json(chosen, text="Answer drawing on several documents.")


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------


def test_ui_page_and_assets_are_served(client_with):
    client = client_with(ScriptedLLM(lambda p: ""), docs=())
    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert '<script src="/static/app.js"' in page.text
    assert 'href="/static/styles.css"' in page.text
    for asset, kind in (("app.js", "javascript"), ("styles.css", "text/css")):
        resp = client.get(f"/static/{asset}")
        assert resp.status_code == 200
        assert kind in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# 6.1 — Citation content for every format
# ---------------------------------------------------------------------------


def test_citation_display_data_matches_store_for_all_formats(client_with, tmp_path):
    client = client_with(ScriptedLLM(_cite_one_per_document))
    answer = client.post(f"{API}/query", json={"question": "encryption memory protocol"}).json()[
        "answer"
    ]
    names = {c["document_name"] for c in answer["citations"]}
    assert names == {"sample_spec.pdf", "sample_spec.md", "sample_document.txt"}

    conn = get_connection()
    try:
        for citation in answer["citations"]:
            shown = client.get(f"{API}/citations/{quote(citation['passage_id'])}").json()
            stored = get_passage(conn, citation["passage_id"])
            # The UI displays `shown`; it must be the store's record, verbatim.
            assert shown["text"] == stored.text
            assert shown["document_name"] == citation["document_name"]
            assert shown["location_label"] == citation["location_label"]
    finally:
        conn.close()

    labels = {c["document_name"]: c["location_label"] for c in answer["citations"]}
    assert labels["sample_spec.pdf"].startswith("Page ")
    assert labels["sample_spec.md"].startswith("Section: ")
    assert labels["sample_document.txt"].startswith("Paragraph ")


# ---------------------------------------------------------------------------
# 6.2 — Multi-document answers
# ---------------------------------------------------------------------------


def test_multi_document_citations_resolve_to_their_own_passages(client_with):
    client = client_with(ScriptedLLM(_cite_one_per_document))
    answer = client.post(f"{API}/query", json={"question": "encryption memory protocol"}).json()[
        "answer"
    ]
    citations = answer["citations"]
    assert len(citations) == 3
    assert len({c["passage_id"] for c in citations}) == 3

    texts = []
    for citation in citations:
        shown = client.get(f"{API}/citations/{quote(citation['passage_id'])}").json()
        assert shown["passage_id"] == citation["passage_id"]
        assert shown["document_id"] == citation["passage_id"].split(":", 1)[0]
        assert shown["document_name"] == citation["document_name"]
        texts.append(shown["text"])
    assert len(set(texts)) == 3  # no two citations show the same passage


def test_citation_of_removed_document_reports_missing(client_with):
    client = client_with(ScriptedLLM(_cite_one_per_document), docs=("sample_spec.pdf",))
    [citation] = client.post(f"{API}/query", json={"question": "encryption"}).json()["answer"][
        "citations"
    ]
    client.delete(f"{API}/documents/sample-spec-pdf-d58d38e8")
    resp = client.get(f"{API}/citations/{quote(citation['passage_id'])}")
    assert resp.status_code == 404  # the UI shows "could not be loaded … removed"


# ---------------------------------------------------------------------------
# Script rules
# ---------------------------------------------------------------------------


def test_script_never_inserts_html():
    script = (STATIC / "app.js").read_text()
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert sink not in script, f"app.js must not use {sink}"


def test_script_displays_passage_from_content_store():
    script = (STATIC / "app.js").read_text()
    assert re.search(r"api\(`/citations/\$\{encodeURIComponent\(citation\.passage_id\)\}`\)",
                     script)
    assert 'text: passage.text' in script
    assert "citation.excerpt" not in script  # never display the answer's own copy
