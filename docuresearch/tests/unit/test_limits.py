"""Tests for daily question limits — Story 9.1 (app.limits.usage and config)."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime

import pytest

from app.config import ConfigError, Settings
from app.limits import (
    MAX_REFUNDS_PER_DAY,
    DailyLimits,
    QuestionLimitReached,
    next_reset,
    refund_user_question,
    reserve_question,
    usage_summary,
    utc_day,
)
from app.store.schema import create_schema

NOON = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
NEXT_DAY = datetime(2026, 10, 8, 0, 0, 1, tzinfo=UTC)
LIMITS = DailyLimits(per_user=3, per_site=5)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "limits.db"
    conn = sqlite3.connect(path)
    create_schema(conn)
    conn.close()
    return path


@pytest.fixture
def conn(db):
    c = sqlite3.connect(db)
    yield c
    c.close()


def _ask(conn, user, n=1, limits=LIMITS, now=NOON):
    for _ in range(n):
        reserve_question(conn, user, limits, now)


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def test_reserve_counts_user_and_site(conn):
    reservation = reserve_question(conn, "alice", LIMITS, NOON)
    assert reservation.day == "2026-10-07"
    assert reservation.user_id == "alice"
    summary = usage_summary(conn, "alice", LIMITS, NOON)
    assert (summary.user.used, summary.user.limit, summary.user.remaining) == (1, 3, 2)
    assert (summary.site.used, summary.site.limit, summary.site.remaining) == (1, 5, 4)
    assert summary.day == "2026-10-07"
    assert summary.resets_at == datetime(2026, 10, 8, tzinfo=UTC)


def test_user_limit_refuses_without_counting(conn):
    _ask(conn, "alice", 3)
    with pytest.raises(QuestionLimitReached) as info:
        reserve_question(conn, "alice", LIMITS, NOON)
    assert info.value.scope == "user"
    assert info.value.limit == 3
    assert info.value.resets_at == datetime(2026, 10, 8, tzinfo=UTC)
    summary = usage_summary(conn, "alice", LIMITS, NOON)
    assert (summary.user.used, summary.site.used) == (3, 3)
    assert summary.user.remaining == 0


def test_users_are_counted_separately_until_site_limit(conn):
    _ask(conn, "alice", 3)
    _ask(conn, "bob", 2)  # alice is at her limit; bob can still ask
    assert usage_summary(conn, "bob", LIMITS, NOON).user.used == 2
    with pytest.raises(QuestionLimitReached) as info:
        reserve_question(conn, "carol", LIMITS, NOON)
    assert info.value.scope == "site"
    assert usage_summary(conn, "carol", LIMITS, NOON).user.used == 0


def test_day_rollover_starts_at_zero(conn):
    _ask(conn, "alice", 3)
    reserve_question(conn, "alice", LIMITS, NEXT_DAY)
    summary = usage_summary(conn, "alice", LIMITS, NEXT_DAY)
    assert summary.day == "2026-10-08"
    assert (summary.user.used, summary.site.used) == (1, 1)


def test_limits_off_never_refuses_and_reports_null_limits(conn):
    off = DailyLimits()
    assert off.enabled is False
    _ask(conn, "alice", 10, limits=off)
    summary = usage_summary(conn, "alice", off, NOON)
    assert (summary.user.limit, summary.user.remaining) == (None, None)
    assert (summary.site.limit, summary.site.remaining) == (None, None)


def test_remaining_is_clamped_at_zero_when_limit_lowered(conn):
    _ask(conn, "alice", 3, limits=DailyLimits(per_user=5))
    summary = usage_summary(conn, "alice", DailyLimits(per_user=2), NOON)
    assert summary.user.used == 3
    assert summary.user.remaining == 0


def test_only_one_limit_set(conn):
    site_only = DailyLimits(per_site=2)
    assert site_only.enabled
    _ask(conn, "alice", 2, limits=site_only)
    with pytest.raises(QuestionLimitReached) as info:
        reserve_question(conn, "alice", site_only, NOON)
    assert info.value.scope == "site"
    assert usage_summary(conn, "alice", site_only, NOON).user.limit is None


def test_utc_day_and_next_reset_handle_other_timezones():
    from datetime import timedelta, timezone

    lagos_late = datetime(2026, 10, 8, 0, 30, tzinfo=timezone(timedelta(hours=1)))
    assert utc_day(lagos_late) == "2026-10-07"  # still 23:30 UTC
    assert next_reset(lagos_late) == datetime(2026, 10, 8, tzinfo=UTC)
    assert next_reset(datetime(2026, 10, 7, 0, 0, tzinfo=UTC)) == datetime(2026, 10, 8, tzinfo=UTC)


def test_concurrent_reservations_never_exceed_limit(db):
    limits = DailyLimits(per_user=5, per_site=0)
    results: list[str] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(20, timeout=10)

    def worker():
        c = sqlite3.connect(db, timeout=30)
        try:
            barrier.wait()
            reserve_question(c, "alice", limits, NOON)
            results.append("ok")
        except QuestionLimitReached:
            results.append("refused")
        except BaseException as exc:  # recorded so the test fails clearly
            errors.append(exc)
        finally:
            c.close()

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "worker threads hung"
    assert errors == []
    assert results.count("ok") == 5
    assert results.count("refused") == 15
    c = sqlite3.connect(db)
    assert usage_summary(c, "alice", limits, NOON).user.used == 5
    c.close()


# ---------------------------------------------------------------------------
# Refunds
# ---------------------------------------------------------------------------


def test_refund_restores_user_but_keeps_site(conn):
    _ask(conn, "alice", 2)
    assert refund_user_question(conn, "alice", "2026-10-07") is True
    summary = usage_summary(conn, "alice", LIMITS, NOON)
    assert (summary.user.used, summary.site.used) == (1, 2)


def test_third_refund_of_the_day_is_refused(conn):
    assert MAX_REFUNDS_PER_DAY == 2
    _ask(conn, "alice", 3)
    assert refund_user_question(conn, "alice", "2026-10-07") is True
    assert refund_user_question(conn, "alice", "2026-10-07") is True
    assert refund_user_question(conn, "alice", "2026-10-07") is False
    assert usage_summary(conn, "alice", LIMITS, NOON).user.used == 1
    # The cap is per user: bob still gets his refunds.
    _ask(conn, "bob", 1)
    assert refund_user_question(conn, "bob", "2026-10-07") is True
    # And per day: alice gets refunds again tomorrow.
    reserve_question(conn, "alice", LIMITS, NEXT_DAY)
    assert refund_user_question(conn, "alice", "2026-10-08") is True


def test_refund_never_goes_below_zero(conn):
    assert refund_user_question(conn, "alice", "2026-10-07") is False
    assert usage_summary(conn, "alice", LIMITS, NOON).user.used == 0
    # A refused no-op refund does not use up the allowance.
    _ask(conn, "alice", 1)
    assert refund_user_question(conn, "alice", "2026-10-07") is True
    assert refund_user_question(conn, "alice", "2026-10-07") is False  # count is 0 again
    _ask(conn, "alice", 1)
    assert refund_user_question(conn, "alice", "2026-10-07") is True


def test_refund_across_midnight_applies_to_reservation_day(conn):
    late = datetime(2026, 10, 7, 23, 59, 59, tzinfo=UTC)
    reservation = reserve_question(conn, "alice", LIMITS, late)
    reserve_question(conn, "alice", LIMITS, NEXT_DAY)  # a question after midnight
    assert refund_user_question(conn, "alice", reservation.day) is True
    assert usage_summary(conn, "alice", LIMITS, late).user.used == 0
    assert usage_summary(conn, "alice", LIMITS, NEXT_DAY).user.used == 1


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def test_user_message_singular_and_plural():
    reset = datetime(2026, 10, 8, tzinfo=UTC)
    one = str(QuestionLimitReached("user", 1, reset))
    assert "your daily limit of 1 question." in one
    assert "1 questions" not in one
    assert "2026-10-08 00:00 UTC" in one
    assert "your daily limit of 5 questions" in str(QuestionLimitReached("user", 5, reset))


def test_site_message_names_the_shared_limit_without_blaming_the_user():
    msg = str(QuestionLimitReached("site", 40, datetime(2026, 10, 8, tzinfo=UTC)))
    assert "site's shared daily limit of 40 questions" in msg
    assert "2026-10-08 00:00 UTC" in msg
    assert "You have" not in msg
    assert "your" not in msg.lower()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load(tmp_path, toml="", env=None):
    cfg = tmp_path / "c.toml"
    cfg.write_text(toml)
    return Settings.load(cfg, env=env or {})


def test_limits_default_off(tmp_path):
    s = _load(tmp_path)
    assert (s.questions_per_user_per_day, s.questions_per_site_per_day) == (0, 0)


def test_limits_from_toml_and_env(tmp_path):
    toml = "[limits]\nquestions_per_user_per_day = 5\nquestions_per_site_per_day = 40\n"
    s = _load(tmp_path, toml)
    assert (s.questions_per_user_per_day, s.questions_per_site_per_day) == (5, 40)
    s = _load(tmp_path, toml, env={
        "DOCURESEARCH_QUESTIONS_PER_USER_PER_DAY": " 2 ",
        "DOCURESEARCH_QUESTIONS_PER_SITE_PER_DAY": "",  # blank: TOML value applies
    })
    assert (s.questions_per_user_per_day, s.questions_per_site_per_day) == (2, 40)


@pytest.mark.parametrize("value", ["-1", "true", "5.9", "'5'"])
def test_invalid_toml_limits_rejected(tmp_path, value):
    with pytest.raises(ConfigError):
        _load(tmp_path, f"[limits]\nquestions_per_user_per_day = {value}\n")


@pytest.mark.parametrize("value", ["-1", "true", "5.9", "five"])
def test_invalid_env_limits_rejected(tmp_path, value):
    with pytest.raises(ConfigError):
        _load(tmp_path, env={"DOCURESEARCH_QUESTIONS_PER_SITE_PER_DAY": value})


@pytest.mark.parametrize("value", [-1, True, 5.9])
def test_invalid_settings_values_rejected(value):
    with pytest.raises(ConfigError):
        Settings(questions_per_user_per_day=value)
