"""Daily question limits through the API — Story 9.1.

Covers every row of the spec's I/O matrix: counting on all three question
paths, user and site refusals, no-passage abstentions, limits off, day
rollover, refunds after failed model calls (and their daily cap), refunds
across midnight, the concurrent last slot, bypass attempts through
conversations, and per-user isolation of ``GET /api/v1/usage``.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.limits import usage as usage_module
from app.main import create_app
from app.pipeline import ResearchPipeline
from tests.integration.fakes import (
    DOCS,
    HashEmbedding,
    ScriptedLLM,
    answer_json,
    passage_ids_in,
    sign_up,
)

API = "/api/v1"
LIMIT_REACHED = "Daily question limit reached"
QUESTION = "Which encryption algorithm is required?"


class ControlLLM(ScriptedLLM):
    """Cites the first passage it is shown; ``fail`` makes calls unparsable.

    ``on_call`` (if set) runs inside each model call, before it returns.
    """

    def __init__(self):
        self.fail = False
        self.on_call = None
        super().__init__(self._respond_to)

    def _respond_to(self, prompt: str) -> str:
        if self.on_call:
            self.on_call()
        if self.fail:
            return "this is not the answer schema"
        return answer_json(passage_ids_in(prompt)[:1])


@pytest.fixture
def server(tmp_path):
    """make(per_user, per_site) -> (llm, make_browser, db_path) over one shared database."""
    browsers: list[TestClient] = []

    def make(per_user=5, per_site=40):
        db = tmp_path / "limits.db"
        llm = ControlLLM()
        pipeline = ResearchPipeline(str(db), llm, embedding_model=HashEmbedding())
        settings = Settings(
            db_path=db,
            questions_per_user_per_day=per_user,
            questions_per_site_per_day=per_site,
        )
        app = create_app(settings, pipeline)

        def make_browser(email="user@example.com", upload=True):
            client = TestClient(app)
            client.__enter__()
            browsers.append(client)
            client.user = sign_up(client, email=email)
            if upload:
                resp = client.post(
                    f"{API}/documents/upload",
                    files={"file": ("sample_spec.md", (DOCS / "sample_spec.md").read_bytes())},
                )
                assert resp.status_code == 200, resp.text
            return client

        return llm, make_browser, db

    yield make
    for c in browsers:
        c.__exit__(None, None, None)


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _seed(db, scope, user_id, count, day=None):
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO usage_counters (day, scope, user_id, count) VALUES (?, ?, ?, ?)",
        (day or _today(), scope, user_id, count),
    )
    conn.commit()
    conn.close()


def _usage(client):
    resp = client.get(f"{API}/usage")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _used(client):
    body = _usage(client)
    return body["user"]["used"], body["site"]["used"]


def _ask(client, **extra):
    return client.post(f"{API}/query", json={"question": QUESTION, **extra})


def _refused(resp):
    assert resp.status_code == 429, resp.text
    body = resp.json()
    assert body["error"] == LIMIT_REACHED
    return body["detail"]


# ---------------------------------------------------------------------------
# Counting on every question path
# ---------------------------------------------------------------------------


def test_under_limits_answers_and_counts(server):
    llm, make_browser, db = server()
    client = make_browser()
    _seed(db, "user", client.user["id"], 2)
    _seed(db, "site", "", 10)
    resp = _ask(client)
    assert resp.status_code == 200, resp.text
    assert resp.json()["answer"]["citations"]
    assert _used(client) == (3, 11)
    body = _usage(client)
    assert body["user"] == {"used": 3, "limit": 5, "remaining": 2}
    assert body["site"] == {"used": 11, "limit": 40, "remaining": 29}
    assert body["day"] == _today()


def test_each_question_path_counts_exactly_one(server):
    llm, make_browser, _ = server()
    client = make_browser()

    assert _ask(client).status_code == 200
    assert _used(client) == (1, 1)

    session_id = client.post(f"{API}/conversation").json()["session_id"]
    assert _ask(client, session_id=session_id).status_code == 200
    assert _used(client) == (2, 2)

    resp = client.post(
        f"{API}/conversation/{session_id}/follow-up", json={"question": "And for keys?"}
    )
    assert resp.status_code == 200, resp.text
    assert _used(client) == (3, 3)
    assert len(llm.prompts) == 3


def test_usage_resets_at_next_utc_midnight(server):
    _, make_browser, _ = server()
    body = _usage(make_browser(upload=False))
    resets_at = datetime.fromisoformat(body["resets_at"].replace("Z", "+00:00"))
    assert resets_at.tzinfo is not None
    assert resets_at.astimezone(UTC).time().isoformat() == "00:00:00"
    assert resets_at.astimezone(UTC).date().isoformat() > body["day"]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_user_limit_refuses_without_model_call_or_count(server):
    llm, make_browser, db = server()
    client = make_browser()
    _seed(db, "user", client.user["id"], 5)
    _seed(db, "site", "", 5)
    detail = _refused(_ask(client))
    assert "your daily limit of 5 questions" in detail
    assert "00:00 UTC" in detail
    assert llm.prompts == []
    assert _used(client) == (5, 5)
    assert _usage(client)["user"]["remaining"] == 0


def test_user_limit_of_one_uses_singular(server):
    _, make_browser, _ = server(per_user=1)
    client = make_browser()
    assert _ask(client).status_code == 200
    detail = _refused(_ask(client))
    assert "limit of 1 question." in detail


def test_site_limit_refuses_with_site_wording(server):
    llm, make_browser, db = server()
    client = make_browser()
    _seed(db, "user", client.user["id"], 1)
    _seed(db, "site", "", 40)
    detail = _refused(_ask(client))
    assert "site's shared daily limit of 40 questions" in detail
    assert "00:00 UTC" in detail
    assert "your" not in detail.lower()
    assert llm.prompts == []
    assert _used(client) == (1, 40)


def test_one_user_at_limit_does_not_block_another(server):
    _, make_browser, _ = server(per_user=2, per_site=3)
    alice = make_browser("alice@example.com")
    bob = make_browser("bob@example.com")
    assert _ask(alice).status_code == 200
    assert _ask(alice).status_code == 200
    _refused(_ask(alice))
    assert _ask(bob).status_code == 200  # alice's limit is hers alone
    assert "site's shared" in _refused(_ask(bob))  # but the site's 3 are used


def test_follow_up_and_session_query_cannot_bypass(server):
    llm, make_browser, db = server()
    client = make_browser()
    session_id = client.post(f"{API}/conversation").json()["session_id"]
    _seed(db, "user", client.user["id"], 5)

    _refused(_ask(client, session_id=session_id))
    _refused(client.post(
        f"{API}/conversation/{session_id}/follow-up", json={"question": QUESTION}
    ))
    assert llm.prompts == []
    assert client.get(f"{API}/conversation/{session_id}").json()["turns"] == []
    assert _used(client)[0] == 5


# ---------------------------------------------------------------------------
# Not counted
# ---------------------------------------------------------------------------


def test_no_passages_is_not_counted(server):
    llm, make_browser, db = server()
    client = make_browser(upload=False)  # empty library
    _seed(db, "user", client.user["id"], 5)  # even at the limit: no model call, no refusal
    resp = _ask(client)
    assert resp.status_code == 200, resp.text
    assert resp.json()["answer"]["is_abstention"] is True
    assert llm.prompts == []
    assert _used(client) == (5, 0)


def test_limits_off_behaves_as_before(server):
    llm, make_browser, _ = server(per_user=0, per_site=0)
    client = make_browser()
    for _ in range(7):
        assert _ask(client).status_code == 200
    body = _usage(client)
    assert body["user"] == {"used": 0, "limit": None, "remaining": None}
    assert body["site"] == {"used": 0, "limit": None, "remaining": None}
    assert len(llm.prompts) == 7


def test_day_rollover_starts_at_zero(server):
    _, make_browser, db = server()
    client = make_browser()
    _seed(db, "user", client.user["id"], 5, day="2000-01-01")
    _seed(db, "site", "", 40, day="2000-01-01")
    assert _used(client) == (0, 0)
    assert _ask(client).status_code == 200
    assert _used(client) == (1, 1)


# ---------------------------------------------------------------------------
# Failed model calls
# ---------------------------------------------------------------------------


def test_failed_call_refunds_user_but_not_site(server):
    llm, make_browser, db = server()
    client = make_browser()
    _seed(db, "user", client.user["id"], 2)
    llm.fail = True
    resp = _ask(client)
    assert resp.status_code == 502, resp.text
    assert resp.json()["error"] == "Generation failed"
    assert _used(client) == (2, 1)


def test_failed_follow_up_is_refunded_too(server):
    llm, make_browser, _ = server()
    client = make_browser()
    session_id = client.post(f"{API}/conversation").json()["session_id"]
    llm.fail = True
    resp = client.post(f"{API}/conversation/{session_id}/follow-up", json={"question": QUESTION})
    assert resp.status_code == 502, resp.text
    assert _used(client) == (0, 1)


def test_refund_cap_counts_third_failure(server):
    llm, make_browser, db = server()
    client = make_browser()
    _seed(db, "user", client.user["id"], 2)
    _seed(db, "refund", client.user["id"], 2)  # two refunds already today
    llm.fail = True
    assert _ask(client).status_code == 502
    assert _used(client) == (3, 1)


def test_refunds_per_day_are_capped_at_two(server):
    llm, make_browser, _ = server()
    client = make_browser()
    llm.fail = True
    for _ in range(4):
        assert _ask(client).status_code == 502
    assert _used(client) == (2, 4)  # first two refunded, then failures count


def test_refund_across_midnight_applies_to_reservation_day(server, monkeypatch):
    llm, make_browser, db = server()
    client = make_browser()
    clock = {"now": datetime(2026, 10, 7, 23, 59, 59, tzinfo=UTC)}
    monkeypatch.setattr(usage_module, "utc_now", lambda: clock["now"])
    _seed(db, "user", client.user["id"], 2, day="2026-10-07")

    def cross_midnight():
        clock["now"] = datetime(2026, 10, 8, 0, 0, 1, tzinfo=UTC)

    llm.fail = True
    llm.on_call = cross_midnight
    assert _ask(client).status_code == 502

    conn = sqlite3.connect(db)
    rows = dict(
        ((day, scope), count)
        for day, scope, count in conn.execute(
            "SELECT day, scope, count FROM usage_counters WHERE user_id = ?",
            (client.user["id"],),
        )
    )
    conn.close()
    assert rows[("2026-10-07", "user")] == 2  # refunded on the reservation's day
    assert rows[("2026-10-07", "refund")] == 1
    assert ("2026-10-08", "user") not in rows
    assert _used(client) == (0, 0)  # the new day is untouched


# ---------------------------------------------------------------------------
# Concurrency and isolation
# ---------------------------------------------------------------------------


def test_concurrent_last_slot(server):
    llm, make_browser, db = server()
    client = make_browser()
    _seed(db, "user", client.user["id"], 4)
    gate = threading.Barrier(2, timeout=5)

    def slow():
        time.sleep(0.2)  # hold the model call so the two requests overlap

    llm.on_call = slow

    def ask(_):
        try:
            gate.wait()
        except threading.BrokenBarrierError:
            pass
        return _ask(client).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = sorted(pool.map(ask, range(2)))
    assert statuses == [200, 429]
    assert _used(client)[0] == 5
    assert len(llm.prompts) == 1


def test_usage_shows_only_the_callers_own_count(server):
    _, make_browser, _ = server()
    alice = make_browser("alice@example.com")
    bob = make_browser("bob@example.com")
    for _ in range(3):
        assert _ask(alice).status_code == 200
    assert _ask(bob).status_code == 200
    assert _used(alice) == (3, 4)
    assert _used(bob) == (1, 4)
    assert "alice" not in str(_usage(bob))


def test_usage_requires_sign_in(server):
    _, make_browser, _ = server()
    client = make_browser(upload=False)
    client.post(f"{API}/auth/logout")
    resp = client.get(f"{API}/usage")
    assert resp.status_code == 401
    assert resp.json()["error"] == "Not signed in"
