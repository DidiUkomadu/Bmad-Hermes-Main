"""Daily question limits — Story 9.1.

One place owns question counting. A question is counted when it is about to
be sent to the model (see ``ResearchPipeline.generate``'s ``before_model_call``
hook), per user and site-wide, per UTC day. Counts live in the
``usage_counters`` table, so they survive restarts.

Rows in ``usage_counters`` (primary key ``day, scope, user_id``):
    scope ``'user'``   — questions counted for ``user_id`` on ``day``
    scope ``'site'``   — questions counted site-wide (``user_id`` is ``''``)
    scope ``'refund'`` — refunds granted to ``user_id`` on ``day``

A limit of 0 means unlimited. Checking and incrementing happen in one
``BEGIN IMMEDIATE`` transaction, so concurrent requests can never push a count
past its limit. The connection passed in must not have a transaction open.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta

logger = logging.getLogger(__name__)

MAX_REFUNDS_PER_DAY = 2
"""Failed model calls refunded per user per UTC day; later failures count.

Unlimited refunds would let one user drain the site budget by provoking
failures (for example with a prompt-injecting document), because the site
count is always kept.
"""

_SITE_USER = ""  # user_id of the site-wide row (NULL would defeat the primary key)


@dataclass(frozen=True)
class DailyLimits:
    """Questions allowed per UTC day; 0 means unlimited."""

    per_user: int = 0
    per_site: int = 0

    @property
    def enabled(self) -> bool:
        return self.per_user > 0 or self.per_site > 0


@dataclass(frozen=True)
class Reservation:
    """A counted question; *day* is the UTC day it was counted against."""

    user_id: str
    day: str


@dataclass(frozen=True)
class UsageCount:
    used: int
    limit: int | None  # None = unlimited

    @property
    def remaining(self) -> int | None:
        return None if self.limit is None else max(self.limit - self.used, 0)


@dataclass(frozen=True)
class UsageSummary:
    day: str
    resets_at: datetime
    user: UsageCount
    site: UsageCount


def _questions(n: int) -> str:
    return f"{n} question" if n == 1 else f"{n} questions"


def _format_reset(resets_at: datetime) -> str:
    return resets_at.strftime("%Y-%m-%d %H:%M UTC")


class QuestionLimitReached(Exception):  # noqa: N818 — name fixed by the Story 9.1 spec
    """A daily question limit is reached; the question must not be sent to the model.

    Attributes:
        scope: ``"user"`` (the asker's own limit) or ``"site"`` (the shared limit).
        limit: The limit that was reached.
        resets_at: When counts reset (the next 00:00 UTC).
    """

    def __init__(self, scope: str, limit: int, resets_at: datetime) -> None:
        self.scope = scope
        self.limit = limit
        self.resets_at = resets_at
        when = _format_reset(resets_at)
        if scope == "user":
            message = (
                f"You have reached your daily limit of {_questions(limit)}. "
                f"It resets at {when}."
            )
        else:
            message = (
                f"The site's shared daily limit of {_questions(limit)} has been reached "
                f"for all users. It resets at {when}."
            )
        super().__init__(message)


def utc_now() -> datetime:
    """The current time in UTC (module-level so tests can replace it)."""
    return datetime.now(UTC)


def utc_day(now: datetime) -> str:
    """The UTC date of *now* as ``YYYY-MM-DD``."""
    return _as_utc(now).date().isoformat()


def next_reset(now: datetime) -> datetime:
    """The next 00:00 UTC after *now*."""
    tomorrow = _as_utc(now).date() + timedelta(days=1)
    return datetime.combine(tomorrow, time(0), tzinfo=UTC)


def _as_utc(now: datetime) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now.astimezone(UTC)


def _count(conn: sqlite3.Connection, day: str, scope: str, user_id: str) -> int:
    row = conn.execute(
        "SELECT count FROM usage_counters WHERE day = ? AND scope = ? AND user_id = ?",
        (day, scope, user_id),
    ).fetchone()
    return int(row[0]) if row else 0


def _increment(conn: sqlite3.Connection, day: str, scope: str, user_id: str) -> None:
    conn.execute(
        """INSERT INTO usage_counters (day, scope, user_id, count) VALUES (?, ?, ?, 1)
           ON CONFLICT(day, scope, user_id) DO UPDATE SET count = count + 1""",
        (day, scope, user_id),
    )


def reserve_question(
    conn: sqlite3.Connection,
    user_id: str,
    limits: DailyLimits,
    now: datetime | None = None,
) -> Reservation:
    """Atomically check both limits and count one question for *user_id*.

    Raises:
        QuestionLimitReached: if the user's or the site's limit is reached.
            Nothing is counted in that case.
    """
    now = now or utc_now()
    day = utc_day(now)
    conn.execute("BEGIN IMMEDIATE")
    try:
        if limits.per_user > 0 and _count(conn, day, "user", user_id) >= limits.per_user:
            raise QuestionLimitReached("user", limits.per_user, next_reset(now))
        if limits.per_site > 0 and _count(conn, day, "site", _SITE_USER) >= limits.per_site:
            raise QuestionLimitReached("site", limits.per_site, next_reset(now))
        _increment(conn, day, "user", user_id)
        _increment(conn, day, "site", _SITE_USER)
    except QuestionLimitReached as exc:
        conn.rollback()
        logger.warning(
            "Question refused: %s daily limit of %d reached (user %s, day %s)",
            exc.scope, exc.limit, user_id, day,
        )
        raise
    except BaseException:
        conn.rollback()
        raise
    conn.commit()
    return Reservation(user_id=user_id, day=day)


def refund_user_question(conn: sqlite3.Connection, user_id: str, day: str) -> bool:
    """Refund one of *user_id*'s questions on *day* after a failed model call.

    At most ``MAX_REFUNDS_PER_DAY`` refunds are granted per user per day; the
    site count is never refunded. The count never goes below 0.

    Returns:
        True if the question was refunded.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        cap_reached = _count(conn, day, "refund", user_id) >= MAX_REFUNDS_PER_DAY
        refunded = not cap_reached and _count(conn, day, "user", user_id) > 0
        if cap_reached:
            logger.warning(
                "Refund refused: daily refund cap of %d reached (user %s, day %s)",
                MAX_REFUNDS_PER_DAY, user_id, day,
            )
        if refunded:
            conn.execute(
                """UPDATE usage_counters SET count = count - 1
                   WHERE day = ? AND scope = 'user' AND user_id = ? AND count > 0""",
                (day, user_id),
            )
            _increment(conn, day, "refund", user_id)
    except BaseException:
        conn.rollback()
        raise
    conn.commit()
    return refunded


def usage_summary(
    conn: sqlite3.Connection,
    user_id: str,
    limits: DailyLimits,
    now: datetime | None = None,
) -> UsageSummary:
    """Today's counts for *user_id* (only theirs) and for the whole site."""
    now = now or utc_now()
    day = utc_day(now)
    return UsageSummary(
        day=day,
        resets_at=next_reset(now),
        user=UsageCount(_count(conn, day, "user", user_id), limits.per_user or None),
        site=UsageCount(_count(conn, day, "site", _SITE_USER), limits.per_site or None),
    )

