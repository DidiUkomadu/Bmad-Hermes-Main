"""Accounts and sign-in sessions — Epic 8 (stories 8.1, 8.2, 8.6).

See the change proposal (``_bmad-output/planning-artifacts/
docuresearch-change-proposal-multi-user.md``) for the decisions referenced
below (D1–D10).

- Users sign in with email and password (D1). Emails are unique and
  case-insensitive; passwords are stored as scrypt hashes (D2).
- A sign-in creates a server-side session: a random token is given to the
  client and only its SHA-256 hash is stored, so a leaked database does not
  leak usable sessions (D3). Sessions expire and are deleted on sign-out.
- The first account registered adopts all unowned documents and
  conversations, so data from the single-user era is not lost (D7).
- Repeated failed sign-ins for an email are throttled in memory (D9).
"""

from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.auth.passwords import hash_password, verify_password

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 256
MAX_DISPLAY_NAME_LENGTH = 80
DEFAULT_SESSION_LIFETIME = timedelta(days=7)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# Verified against when the email is unknown, so timing does not reveal accounts.
_DUMMY_HASH = hash_password("not-a-real-password")


class AuthError(ValueError):
    """Invalid registration or sign-in input; the message is safe to show."""


class EmailTakenError(AuthError):
    pass


class TooManyAttemptsError(AuthError):
    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__(
            f"Too many failed sign-in attempts. Try again in {retry_after_seconds} seconds."
        )
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class User:
    id: str
    email: str
    display_name: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------


def normalize_email(email: str) -> str:
    return email.strip().lower()


def register_user(
    conn: sqlite3.Connection,
    email: str,
    password: str,
    display_name: str | None = None,
) -> User:
    """Create an account. The first account adopts unowned data (D7).

    Raises:
        AuthError: invalid email, password, or display name.
        EmailTakenError: an account with this email already exists.
    """
    email = normalize_email(email)
    if not _EMAIL_RE.match(email):
        raise AuthError("Enter a valid email address.")
    _check_password(password)
    name = (display_name or "").strip() or email.split("@", 1)[0]
    if len(name) > MAX_DISPLAY_NAME_LENGTH:
        raise AuthError(f"Display name must be at most {MAX_DISPLAY_NAME_LENGTH} characters.")

    user = User(
        id=str(uuid.uuid4()), email=email, display_name=name, created_at=datetime.now(UTC)
    )
    try:
        first_user = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        conn.execute(
            "INSERT INTO users (id, email, display_name, password_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user.id, user.email, user.display_name, hash_password(password),
             user.created_at.isoformat()),
        )
        if first_user:
            _adopt_unowned_data(conn, user.id)
        conn.commit()
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise EmailTakenError("An account with this email already exists.") from exc
    return user


def authenticate(conn: sqlite3.Connection, email: str, password: str) -> User | None:
    """Return the user if the email and password match, else None."""
    row = conn.execute(
        "SELECT * FROM users WHERE email = ?", (normalize_email(email),)
    ).fetchone()
    if row is None:
        verify_password(password, _DUMMY_HASH)  # equalise timing
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    return _row_to_user(row)


def set_password(conn: sqlite3.Connection, email: str, new_password: str) -> bool:
    """Replace a user's password and sign them out everywhere. False if no such user."""
    _check_password(new_password)
    row = conn.execute(
        "SELECT id FROM users WHERE email = ?", (normalize_email(email),)
    ).fetchone()
    if row is None:
        return False
    conn.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(new_password), row["id"])
    )
    conn.execute("DELETE FROM auth_sessions WHERE user_id = ?", (row["id"],))
    conn.commit()
    return True


def _check_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at most {MAX_PASSWORD_LENGTH} characters.")


def _adopt_unowned_data(conn: sqlite3.Connection, user_id: str) -> None:
    conn.execute("UPDATE documents SET owner_id = ? WHERE owner_id IS NULL", (user_id,))
    conn.execute("UPDATE conversations SET owner_id = ? WHERE owner_id IS NULL", (user_id,))


def _row_to_user(row: sqlite3.Row) -> User:
    return User(
        id=row["id"],
        email=row["email"],
        display_name=row["display_name"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def create_auth_session(
    conn: sqlite3.Connection,
    user_id: str,
    lifetime: timedelta = DEFAULT_SESSION_LIFETIME,
) -> str:
    """Start a session for *user_id* and return its token (shown to the client once)."""
    token = secrets.token_urlsafe(32)
    now = datetime.now(UTC)
    conn.execute(
        "INSERT INTO auth_sessions (token_hash, user_id, created_at, expires_at) "
        "VALUES (?, ?, ?, ?)",
        (_token_hash(token), user_id, now.isoformat(), (now + lifetime).isoformat()),
    )
    conn.commit()
    return token


def user_for_token(conn: sqlite3.Connection, token: str | None) -> User | None:
    """The signed-in user for a session token, or None if missing, unknown, or expired."""
    if not token:
        return None
    row = conn.execute(
        "SELECT u.*, s.expires_at FROM auth_sessions s JOIN users u ON u.id = s.user_id "
        "WHERE s.token_hash = ?",
        (_token_hash(token),),
    ).fetchone()
    if row is None:
        return None
    if datetime.fromisoformat(row["expires_at"]) <= datetime.now(UTC):
        conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (_token_hash(token),))
        conn.commit()
        return None
    return _row_to_user(row)


def revoke_auth_session(conn: sqlite3.Connection, token: str | None) -> None:
    """Sign out: delete the session for *token* (no-op if unknown)."""
    if token:
        conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (_token_hash(token),))
        conn.commit()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Throttling (D9)
# ---------------------------------------------------------------------------


class LoginThrottle:
    """Refuse sign-in for an email after repeated failures, for a cool-down period.

    In-memory and per process: adequate for the single-process local server.
    """

    def __init__(self, max_failures: int = 5, lockout_seconds: int = 300) -> None:
        self._max_failures = max_failures
        self._lockout_seconds = lockout_seconds
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, email: str) -> None:
        """Raise TooManyAttemptsError if *email* is currently locked out."""
        key = normalize_email(email)
        now = time.monotonic()
        with self._lock:
            recent = [t for t in self._failures.get(key, []) if now - t < self._lockout_seconds]
            self._failures[key] = recent
            if len(recent) >= self._max_failures:
                retry = int(self._lockout_seconds - (now - recent[0])) + 1
                raise TooManyAttemptsError(retry)

    def record_failure(self, email: str) -> None:
        with self._lock:
            self._failures.setdefault(normalize_email(email), []).append(time.monotonic())

    def reset(self, email: str) -> None:
        with self._lock:
            self._failures.pop(normalize_email(email), None)
