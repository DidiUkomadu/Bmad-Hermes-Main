"""ProviderAllowance and X-RateLimit-Reset parsing — Story 9.2."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

from app.generation.allowance import ProviderAllowance, next_utc_midnight, parse_reset

NOW = datetime(2026, 10, 8, 15, 30, tzinfo=UTC)


def test_not_exhausted_by_default():
    assert ProviderAllowance().exhausted_until(NOW) is None


def test_marked_exhausted_until_reset():
    allowance = ProviderAllowance()
    until = NOW + timedelta(hours=2)
    allowance.mark_exhausted(until)
    assert allowance.exhausted_until(NOW) == until
    assert allowance.exhausted_until(until - timedelta(seconds=1)) == until


def test_clears_itself_once_reset_passes():
    allowance = ProviderAllowance()
    until = NOW + timedelta(hours=2)
    allowance.mark_exhausted(until)
    assert allowance.exhausted_until(until) is None
    # Cleared for good: an earlier clock does not bring it back.
    assert allowance.exhausted_until(NOW) is None


def test_later_mark_extends_earlier_does_not_shorten():
    allowance = ProviderAllowance()
    allowance.mark_exhausted(NOW + timedelta(hours=2))
    allowance.mark_exhausted(NOW + timedelta(hours=1))
    assert allowance.exhausted_until(NOW) == NOW + timedelta(hours=2)
    allowance.mark_exhausted(NOW + timedelta(hours=3))
    assert allowance.exhausted_until(NOW) == NOW + timedelta(hours=3)


def test_thread_safe_marking_and_reading():
    allowance = ProviderAllowance()
    times = [NOW + timedelta(minutes=i) for i in range(1, 201)]
    errors = []

    def work(until):
        try:
            allowance.mark_exhausted(until)
            assert allowance.exhausted_until(NOW) is not None
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(t,)) for t in times]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert allowance.exhausted_until(NOW) == max(times)


def test_parse_reset_milliseconds():
    reset = NOW + timedelta(hours=8, minutes=30)
    assert parse_reset(str(int(reset.timestamp() * 1000)), NOW) == reset


def test_parse_reset_seconds():
    reset = NOW + timedelta(hours=8, minutes=30)
    assert parse_reset(str(int(reset.timestamp())), NOW) == reset
    assert parse_reset(int(reset.timestamp()), NOW) == reset


def test_parse_reset_missing_or_unparsable():
    assert parse_reset(None, NOW) is None
    assert parse_reset("", NOW) is None
    assert parse_reset("soon", NOW) is None
    assert parse_reset("nan", NOW) is None
    assert parse_reset("1e400", NOW) is None


def test_parse_reset_in_the_past():
    assert parse_reset(str(int((NOW - timedelta(minutes=1)).timestamp() * 1000)), NOW) is None
    assert parse_reset(str(int(NOW.timestamp())), NOW) is None


def test_parse_reset_more_than_48_hours_ahead():
    assert parse_reset(str(int((NOW + timedelta(hours=49)).timestamp())), NOW) is None
    assert parse_reset(str(int((NOW + timedelta(hours=48)).timestamp())), NOW) is not None


def test_next_utc_midnight():
    assert next_utc_midnight(NOW) == datetime(2026, 10, 9, tzinfo=UTC)
    assert next_utc_midnight(datetime(2026, 10, 8, tzinfo=UTC)) == datetime(2026, 10, 9, tzinfo=UTC)
