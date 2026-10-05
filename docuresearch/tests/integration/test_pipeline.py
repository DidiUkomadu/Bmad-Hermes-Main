"""End-to-end pipeline tests with a scripted LLM and a deterministic embedder.

These exercise the real ingestion, store, hybrid retrieval, prompt, parsing,
citation resolution, and grounding enforcement together. Only the two
external dependencies (LLM and embedding model) are replaced.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.generation.interface import EvidenceQuality
from app.generation.openai_compat import LLMConfigError, OpenAICompatibleLLM
from app.generation.post_process import UNGROUNDED_ANSWER_TEXT
from app.pipeline import NO_PASSAGES_ANSWER_TEXT, GenerationError, ResearchPipeline
from tests.integration.fakes import DOCS, HashEmbedding, ScriptedLLM
from tests.integration.fakes import answer_json as _answer_json
from tests.integration.fakes import passage_ids_in as _passage_ids_in


def _pipeline(tmp_path: Path, llm) -> ResearchPipeline:
    return ResearchPipeline(str(tmp_path / "t.db"), llm, embedding_model=HashEmbedding())


def test_ingest_stores_embeddings_and_answer_cites_retrieved_passage(tmp_path):
    llm = ScriptedLLM(lambda prompt: _answer_json([_passage_ids_in(prompt)[0]]))
    pipeline = _pipeline(tmp_path, llm)
    info = pipeline.ingest(DOCS / "sample_spec.pdf")

    result = pipeline.answer("What encryption algorithm must be used for data transmissions?")

    assert info["passage_count"] > 0
    assert result.context.passages, "semantic+keyword retrieval returned nothing"
    top = result.context.passages[0]
    assert result.answer.evidence_quality == EvidenceQuality.SUFFICIENT
    assert [c.passage_id for c in result.answer.citations] == [top.passage_id]
    # Citation metadata comes from the store, not from the model.
    assert result.answer.citations[0].document_name == "sample_spec.pdf"
    assert result.answer.citations[0].excerpt == top.text
    assert "AES-256" in " ".join(p.text for p in result.context.passages[:3])


def test_semantic_signal_is_active_after_ingest(tmp_path):
    pipeline = _pipeline(tmp_path, ScriptedLLM(lambda p: ""))
    pipeline.ingest(DOCS / "sample_document.txt")
    # A query with no keyword overlap still retrieves via embeddings only if
    # embeddings were persisted; BM25 alone would return nothing useful.
    context = pipeline.retrieve("zzzz qqqq")
    assert context.passages


def test_fabricated_citation_is_withheld(tmp_path):
    llm = ScriptedLLM(lambda prompt: _answer_json(["not-a-real-passage"], text="Made up."))
    pipeline = _pipeline(tmp_path, llm)
    pipeline.ingest(DOCS / "sample_spec.md")

    answer = pipeline.answer("What encryption is required?").answer

    assert answer.answer_text == UNGROUNDED_ANSWER_TEXT
    assert answer.evidence_quality == EvidenceQuality.INSUFFICIENT
    assert answer.is_abstention is True
    assert answer.citations == []


def test_scope_restricts_passages_to_one_document(tmp_path):
    pipeline = _pipeline(tmp_path, ScriptedLLM(lambda p: ""))
    spec = pipeline.ingest(DOCS / "sample_spec.md")
    pipeline.ingest(DOCS / "sample_document.txt")

    context = pipeline.retrieve("memory management", document_id=spec["document_id"])

    assert context.passages
    assert {p.document_name for p in context.passages} == {"sample_spec.md"}


def test_empty_store_abstains_without_calling_llm(tmp_path):
    llm = ScriptedLLM(lambda p: pytest.fail("LLM must not be called"))
    answer = _pipeline(tmp_path, llm).answer("anything").answer
    assert answer.answer_text == NO_PASSAGES_ANSWER_TEXT
    assert answer.is_abstention is True
    assert llm.prompts == []


def test_unparsable_llm_output_raises_instead_of_answering(tmp_path):
    pipeline = _pipeline(tmp_path, ScriptedLLM(lambda p: "I think it's AES."))
    pipeline.ingest(DOCS / "sample_spec.md")
    with pytest.raises(GenerationError):
        pipeline.answer("What encryption is required?")


def test_ingest_bytes_keeps_original_name_and_stable_ids(tmp_path):
    pipeline = _pipeline(tmp_path, ScriptedLLM(lambda p: ""))
    data = (DOCS / "sample_spec.pdf").read_bytes()
    from_bytes = pipeline.ingest_bytes(data, "sample_spec.pdf")

    other = ResearchPipeline(str(tmp_path / "u.db"), ScriptedLLM(lambda p: ""),
                             embedding_model=HashEmbedding())
    from_path = other.ingest(DOCS / "sample_spec.pdf")

    assert from_bytes["document_name"] == "sample_spec.pdf"
    assert from_bytes["document_id"] == from_path["document_id"]
    assert from_bytes["passage_ids"] == from_path["passage_ids"]


# ---------------------------------------------------------------------------
# OpenAI-compatible backend
# ---------------------------------------------------------------------------


def _llm_with(handler) -> OpenAICompatibleLLM:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatibleLLM("https://llm.example/v1/", "test-model", api_key="k", client=client)


def test_openai_compat_sends_request_and_parses_answer():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        content = "```json\n" + _answer_json(["p1"]) + "\n```"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    resp = _llm_with(handler).generate("PROMPT")

    assert seen["url"] == "https://llm.example/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "test-model"
    assert seen["body"]["messages"][-1] == {"role": "user", "content": "PROMPT"}
    assert resp.parse_error is None
    assert resp.structured is not None
    assert resp.structured.citations[0].passage_id == "p1"


def test_openai_compat_reports_http_errors(monkeypatch):
    monkeypatch.setattr("app.generation.openai_compat.time.sleep", lambda s: None)
    resp = _llm_with(lambda r: httpx.Response(500, text="boom")).generate("PROMPT")
    assert resp.structured is None
    assert "LLM request failed" in resp.parse_error


def test_openai_compat_from_env_requires_config(monkeypatch):
    monkeypatch.delenv("DOCURESEARCH_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("DOCURESEARCH_LLM_MODEL", raising=False)
    with pytest.raises(LLMConfigError):
        OpenAICompatibleLLM.from_env()


def test_openai_compat_retries_rate_limits_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr("app.generation.openai_compat.time.sleep", sleeps.append)
    statuses = iter([429, 503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if status != 200:
            return httpx.Response(status, headers={"retry-after": "1"} if status == 429 else {})
        return httpx.Response(200, json={"choices": [{"message": {"content": _answer_json([])}}]})

    resp = _llm_with(handler).generate("PROMPT")
    assert resp.parse_error is None
    assert sleeps == [1.0, 4.0]  # Retry-After honored, then exponential backoff


def test_openai_compat_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr("app.generation.openai_compat.time.sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429)

    resp = _llm_with(handler).generate("PROMPT")
    assert "429" in resp.parse_error
    assert len(calls) == 4  # first try + 3 retries


def test_openai_compat_does_not_retry_client_errors(monkeypatch):
    monkeypatch.setattr("app.generation.openai_compat.time.sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401)

    assert "401" in _llm_with(handler).generate("PROMPT").parse_error
    assert calls == [1]


def test_openai_compat_retries_error_embedded_in_200_body(monkeypatch):
    monkeypatch.setattr("app.generation.openai_compat.time.sleep", lambda s: None)
    bodies = iter([
        {"error": {"code": 429, "message": "Provider returned error",
                   "metadata": {"raw": "rate-limited upstream"}}},
        {"choices": [{"message": {"content": _answer_json([])}}]},
    ])
    resp = _llm_with(lambda r: httpx.Response(200, json=next(bodies))).generate("PROMPT")
    assert resp.parse_error is None


def test_openai_compat_reports_embedded_error_message(monkeypatch):
    monkeypatch.setattr("app.generation.openai_compat.time.sleep", lambda s: None)
    body = {"error": {"code": 400, "message": "Provider returned error",
                      "metadata": {"raw": "model does not support this"}}}
    resp = _llm_with(lambda r: httpx.Response(200, json=body)).generate("PROMPT")
    assert "provider error 400" in resp.parse_error
    assert "model does not support this" in resp.parse_error


def test_openai_compat_retries_network_errors(monkeypatch):
    monkeypatch.setattr("app.generation.openai_compat.time.sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("Temporary failure in name resolution", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": _answer_json([])}}]})

    resp = _llm_with(handler).generate("PROMPT")
    assert resp.parse_error is None
    assert len(calls) == 3


def test_openai_compat_reports_persistent_network_failure(monkeypatch):
    monkeypatch.setattr("app.generation.openai_compat.time.sleep", lambda s: None)

    def handler(request):
        raise httpx.ConnectError("Temporary failure in name resolution", request=request)

    resp = _llm_with(handler).generate("PROMPT")
    assert resp.structured is None
    assert "name resolution" in resp.parse_error
