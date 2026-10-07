"""Handling the LLM provider's daily cap — Story 9.2.

Covers every row of the spec's I/O matrix: busy 429s still retried; the
daily-cap 429 in the body, in the headers, and inside a 200 response, with
and without a reset time; ``remaining: 0`` alone treated as busy; fail-fast
on all three question paths with no HTTP call, no retrieval and no counting;
the refund of the request that discovers the cap; expiry with a patched
clock; a restart; and the evaluation runner.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.generation import allowance as allowance_module
from app.generation.allowance import next_utc_midnight
from app.generation.openai_compat import OpenAICompatibleLLM
from app.main import create_app
from app.pipeline import GenerationError, ProviderAllowanceExhausted, ResearchPipeline
from evaluation.pipeline_system import GENERATION_FAILED_PREFIX, PipelineSystem
from tests.integration.fakes import DOCS, HashEmbedding, answer_json, passage_ids_in, sign_up
from tests.integration.test_limits_api import _seed, _used

API = "/api/v1"
QUESTION = "Which encryption algorithm is required?"
ALLOWANCE_EXHAUSTED = "The demo has used today's free AI allowance"
DAILY_MESSAGE = "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000."


def _ms(when: datetime) -> str:
    return str(int(when.timestamp() * 1000))


def _reset_in(hours: float = 3) -> datetime:
    # Whole seconds, so it round-trips through an epoch value exactly.
    return datetime.now(UTC).replace(microsecond=0) + timedelta(hours=hours)


def _ok(request: httpx.Request) -> httpx.Response:
    prompt = json.loads(request.content)["messages"][-1]["content"]
    ids = passage_ids_in(prompt)[:1] or ["p1"]
    return httpx.Response(200, json={"choices": [{"message": {"content": answer_json(ids)}}]})


def _daily_body(reset: datetime | None = None, code_status: int = 429) -> httpx.Response:
    error: dict = {"message": DAILY_MESSAGE, "code": 429}
    if reset is not None:
        error["metadata"] = {
            "headers": {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": _ms(reset)}
        }
    return httpx.Response(code_status, json={"error": error})


class Provider:
    """A MockTransport handler: serves scripted responses, then ``default``; counts calls."""

    def __init__(self, *responses, default=_ok):
        self.responses = list(responses)
        self.default = default
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.responses:
            r = self.responses.pop(0)
            return r(request) if callable(r) else r
        return self.default(request)


def _llm(provider: Provider) -> OpenAICompatibleLLM:
    client = httpx.Client(transport=httpx.MockTransport(provider))
    return OpenAICompatibleLLM("https://llm.example/v1/", "test-model", api_key="k", client=client)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("app.generation.openai_compat.time.sleep", sleeps.append)
    return sleeps


# ---------------------------------------------------------------------------
# Connector level
# ---------------------------------------------------------------------------


def test_busy_429_is_retried_then_answers(_no_sleep):
    provider = Provider(
        httpx.Response(429, json={"error": {"message": "Rate limit exceeded: per-minute"}}),
    )
    llm = _llm(provider)
    resp = llm.generate("PROMPT")
    assert resp.parse_error is None
    assert resp.provider_exhausted_until is None
    assert provider.calls == 2
    assert llm.allowance.exhausted_until() is None


def test_daily_cap_in_body_with_ms_reset(caplog):
    reset = _reset_in(5)
    provider = Provider(_daily_body(reset))
    llm = _llm(provider)
    with caplog.at_level("WARNING", logger="app.generation.openai_compat"):
        resp = llm.generate("PROMPT")
    assert provider.calls == 1
    assert resp.structured is None
    assert resp.provider_exhausted_until == reset
    assert llm.allowance.exhausted_until() == reset
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert reset.isoformat() in warnings[0].getMessage()
    assert "Bearer" not in warnings[0].getMessage()


def test_daily_cap_in_http_headers():
    reset = _reset_in(2)
    provider = Provider(
        httpx.Response(
            429,
            headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(reset.timestamp()))},
            json={"error": {"message": "Rate limit exceeded", "code": 429}},
        )
    )
    llm = _llm(provider)
    resp = llm.generate("PROMPT")
    assert provider.calls == 1
    assert resp.provider_exhausted_until == reset


def test_daily_cap_inside_200_body():
    reset = _reset_in(4)
    provider = Provider(_daily_body(reset, code_status=200))
    llm = _llm(provider)
    resp = llm.generate("PROMPT")
    assert provider.calls == 1
    assert resp.provider_exhausted_until == reset


FIXED_NOW = datetime(2026, 10, 8, 15, 30, tzinfo=UTC)


def test_daily_text_without_reset_uses_next_utc_midnight(monkeypatch):
    monkeypatch.setattr(allowance_module, "_now", lambda: FIXED_NOW)
    provider = Provider(_daily_body(None))
    resp = _llm(provider).generate("PROMPT")
    assert provider.calls == 1
    assert resp.provider_exhausted_until == next_utc_midnight(FIXED_NOW)


def test_daily_text_with_unusable_reset_uses_next_utc_midnight(monkeypatch):
    monkeypatch.setattr(allowance_module, "_now", lambda: FIXED_NOW)
    provider = Provider(_daily_body(FIXED_NOW - timedelta(hours=1)))
    resp = _llm(provider).generate("PROMPT")
    assert provider.calls == 1
    assert resp.provider_exhausted_until == next_utc_midnight(FIXED_NOW)


def test_http_header_reset_wins_over_body_metadata():
    header_reset = _reset_in(2)
    body_reset = _reset_in(7)
    response = _daily_body(body_reset)
    provider = Provider(
        httpx.Response(
            429,
            headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": _ms(header_reset)},
            content=response.content,
        )
    )
    resp = _llm(provider).generate("PROMPT")
    assert provider.calls == 1
    assert resp.provider_exhausted_until == header_reset


def test_busy_429_then_daily_cap_stops_retrying():
    reset = _reset_in(3)
    provider = Provider(
        httpx.Response(429, json={"error": {"message": "Rate limit exceeded: per-minute"}}),
        _daily_body(reset),
    )
    llm = _llm(provider)
    resp = llm.generate("PROMPT")
    assert provider.calls == 2
    assert resp.provider_exhausted_until == reset
    assert llm.allowance.exhausted_until() == reset


def test_remaining_zero_without_reset_or_daily_text_is_busy():
    provider = Provider(
        httpx.Response(
            429,
            headers={"X-RateLimit-Remaining": "0"},
            json={"error": {"message": "Rate limit exceeded", "code": 429}},
        )
    )
    llm = _llm(provider)
    resp = llm.generate("PROMPT")
    assert resp.parse_error is None
    assert provider.calls == 2
    assert llm.allowance.exhausted_until() is None


def test_5xx_and_network_errors_still_retried():
    def network_error(request):
        raise httpx.ConnectError("dns", request=request)

    provider = Provider(httpx.Response(503), network_error)
    resp = _llm(provider).generate("PROMPT")
    assert resp.parse_error is None
    assert provider.calls == 3


def test_while_exhausted_no_http_calls():
    provider = Provider(_daily_body(_reset_in(3)))
    llm = _llm(provider)
    llm.generate("PROMPT")
    for _ in range(3):
        resp = llm.generate("PROMPT")
        assert resp.provider_exhausted_until is not None
        assert resp.parse_error
    assert provider.calls == 1


def test_calls_resume_after_reset(monkeypatch):
    reset = _reset_in(1)
    provider = Provider(_daily_body(reset))
    llm = _llm(provider)
    llm.generate("PROMPT")
    assert llm.generate("PROMPT").provider_exhausted_until == reset
    assert provider.calls == 1

    monkeypatch.setattr(allowance_module, "_now", lambda: reset + timedelta(seconds=1))
    resp = llm.generate("PROMPT")
    assert resp.parse_error is None
    assert provider.calls == 2


def test_restart_is_not_exhausted():
    provider = Provider(_daily_body(_reset_in(3)))
    _llm(provider).generate("PROMPT")
    fresh = _llm(provider)
    assert fresh.allowance.exhausted_until() is None
    assert fresh.generate("PROMPT").parse_error is None
    assert provider.calls == 2


# ---------------------------------------------------------------------------
# Pipeline and evaluation runner
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_with(tmp_path):
    def make(provider: Provider) -> ResearchPipeline:
        pipeline = ResearchPipeline(
            str(tmp_path / "cap.db"), _llm(provider), embedding_model=HashEmbedding()
        )
        pipeline.ingest(DOCS / "sample_spec.md")
        return pipeline

    return make


def test_pipeline_raises_typed_generation_error(pipeline_with):
    reset = _reset_in(3)
    pipeline = pipeline_with(Provider(_daily_body(reset)))
    with pytest.raises(ProviderAllowanceExhausted) as info:
        pipeline.answer(QUESTION)
    assert isinstance(info.value, GenerationError)
    assert info.value.resets_at == reset
    assert pipeline.llm_allowance.exhausted_until() == reset


def test_pipeline_llm_allowance_is_none_for_fakes(tmp_path):
    from tests.integration.fakes import ScriptedLLM

    pipeline = ResearchPipeline(
        str(tmp_path / "f.db"), ScriptedLLM(lambda p: ""), embedding_model=HashEmbedding()
    )
    assert pipeline.llm_allowance is None


def test_evaluation_runner_records_generation_failure(pipeline_with, tmp_path):
    provider = Provider(_daily_body(_reset_in(3)))
    pipeline = pipeline_with(provider)
    system = PipelineSystem(pipeline, str(tmp_path / "cap.db"))
    for _ in range(2):
        passages = system.retrieve(QUESTION)
        assert passages
        result = system.generate(QUESTION, passages)
        assert result["answer"].startswith(GENERATION_FAILED_PREFIX)
        assert result["citations"] == []
    assert provider.calls == 1


# ---------------------------------------------------------------------------
# API level
# ---------------------------------------------------------------------------


@pytest.fixture
def server(tmp_path):
    """make(provider) -> (client, db, pipeline) with 9.1 limits on (5 per user, 40 per site)."""
    clients: list[TestClient] = []

    def make(provider: Provider):
        db = tmp_path / "cap-api.db"
        pipeline = ResearchPipeline(str(db), _llm(provider), embedding_model=HashEmbedding())
        settings = Settings(
            db_path=db, questions_per_user_per_day=5, questions_per_site_per_day=40
        )
        client = TestClient(create_app(settings, pipeline))
        client.__enter__()
        clients.append(client)
        client.user = sign_up(client)
        resp = client.post(
            f"{API}/documents/upload",
            files={"file": ("sample_spec.md", (DOCS / "sample_spec.md").read_bytes())},
        )
        assert resp.status_code == 200, resp.text
        return client, db, pipeline

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def _exhausted(resp, reset: datetime):
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["error"] == ALLOWANCE_EXHAUSTED
    assert reset.strftime("%Y-%m-%d %H:%M UTC") in body["detail"]


def _three_paths(client):
    session_id = client.post(f"{API}/conversation").json()["session_id"]
    return [
        lambda: client.post(f"{API}/query", json={"question": QUESTION}),
        lambda: client.post(
            f"{API}/query", json={"question": QUESTION, "session_id": session_id}
        ),
        lambda: client.post(
            f"{API}/conversation/{session_id}/follow-up", json={"question": QUESTION}
        ),
    ]


def test_discovering_request_gets_503_and_refund(server):
    reset = _reset_in(6)
    provider = Provider(_daily_body(reset))
    client, db, _ = server(provider)
    _seed(db, "user", client.user["id"], 1)
    resp = client.post(f"{API}/query", json={"question": QUESTION})
    _exhausted(resp, reset)
    assert provider.calls == 1
    assert _used(client) == (1, 1)  # user refunded, site count kept (9.1 rule)


def test_discovering_follow_up_gets_503_and_no_turn(server):
    reset = _reset_in(6)
    provider = Provider(_daily_body(reset))
    client, _, _ = server(provider)
    session_id = client.post(f"{API}/conversation").json()["session_id"]
    resp = client.post(f"{API}/conversation/{session_id}/follow-up", json={"question": QUESTION})
    _exhausted(resp, reset)
    assert provider.calls == 1
    assert _used(client) == (0, 1)
    assert client.get(f"{API}/conversation/{session_id}").json()["turns"] == []


def test_while_exhausted_all_paths_fail_fast_without_calls_retrieval_or_counting(
    server, monkeypatch
):
    reset = _reset_in(6)
    provider = Provider(_daily_body(reset))
    client, _, pipeline = server(provider)
    _exhausted(client.post(f"{API}/query", json={"question": QUESTION}), reset)
    counts_after_discovery = _used(client)

    retrievals = []
    original = pipeline.retrieve
    monkeypatch.setattr(
        pipeline, "retrieve", lambda *a, **k: retrievals.append(1) or original(*a, **k)
    )
    for ask in _three_paths(client):
        _exhausted(ask(), reset)
    assert provider.calls == 1
    assert retrievals == []
    assert _used(client) == counts_after_discovery


def test_api_resumes_after_reset(server, monkeypatch):
    reset = _reset_in(1)
    provider = Provider(_daily_body(reset))
    client, _, _ = server(provider)
    _exhausted(client.post(f"{API}/query", json={"question": QUESTION}), reset)
    _exhausted(client.post(f"{API}/query", json={"question": QUESTION}), reset)
    assert provider.calls == 1

    monkeypatch.setattr(allowance_module, "_now", lambda: reset + timedelta(seconds=1))
    resp = client.post(f"{API}/query", json={"question": QUESTION})
    assert resp.status_code == 200, resp.text
    assert provider.calls == 2


def test_unknown_document_is_404_while_exhausted(server):
    reset = _reset_in(6)
    provider = Provider(_daily_body(reset))
    client, _, _ = server(provider)
    _exhausted(client.post(f"{API}/query", json={"question": QUESTION}), reset)
    scope = {"mode": "specific", "document_id": "no-such-document"}
    session_id = client.post(f"{API}/conversation").json()["session_id"]
    for resp in (
        client.post(f"{API}/query", json={"question": QUESTION, "document_scope": scope}),
        client.post(
            f"{API}/query",
            json={"question": QUESTION, "session_id": session_id, "document_scope": scope},
        ),
        client.post(
            f"{API}/conversation/{session_id}/follow-up",
            json={"question": QUESTION, "document_scope": scope},
        ),
    ):
        assert resp.status_code == 404, resp.text
    assert provider.calls == 1
