"""In-memory tracker for the LLM provider's daily allowance — Story 9.2.

When the provider reports that its free daily allowance is used up (e.g.
OpenRouter's ``free-models-per-day`` 429), the connector marks it exhausted
until the provider's reset time. While exhausted, no further calls are made.

The state lives only in this object (held by the connector instance), so a
restart clears it; the next cap response sets it again.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, time, timedelta

# Reset times further ahead than this are not trusted (a daily cap resets within a day).
MAX_RESET_AHEAD = timedelta(hours=48)

# X-RateLimit-Reset values above this are epoch milliseconds; below, seconds.
_MILLISECONDS_THRESHOLD = 10**11


def _now() -> datetime:
    """The current UTC time (patched in tests)."""
    return datetime.now(UTC)


def next_utc_midnight(now: datetime) -> datetime:
    """The next 00:00 UTC after *now*."""
    return datetime.combine(now.astimezone(UTC).date() + timedelta(days=1), time(0), tzinfo=UTC)


def parse_reset(value: object, now: datetime) -> datetime | None:
    """Parse an ``X-RateLimit-Reset`` value (epoch ms above 10^11, else seconds).

    Returns None when it is missing, unparsable, not in the future, or more
    than 48 hours ahead of *now*.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
        if number > _MILLISECONDS_THRESHOLD:
            number /= 1000.0
        reset = datetime.fromtimestamp(number, UTC)
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    if reset <= now or reset - now > MAX_RESET_AHEAD:
        return None
    return reset


class ProviderAllowance:
    """Thread-safe record of "exhausted until <time>" for one provider."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._until: datetime | None = None

    def mark_exhausted(self, until: datetime) -> None:
        """Mark the allowance exhausted until *until* (timezone-aware, UTC)."""
        with self._lock:
            if self._until is None or until > self._until:
                self._until = until

    def exhausted_until(self, now: datetime | None = None) -> datetime | None:
        """The reset time while exhausted, else None (clears once it has passed)."""
        now = now or _now()
        with self._lock:
            if self._until is not None and now >= self._until:
                self._until = None
            return self._until
