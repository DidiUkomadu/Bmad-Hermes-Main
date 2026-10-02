"""Tests for accounts, sessions, password hashing and throttling — Epic 8."""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.auth import (
    AuthError,
    EmailTakenError,
    LoginThrottle,
    TooManyAttemptsError,
    authenticate,
    create_auth_session,
    register_user,
    revoke_auth_session,
    set_password,
    user_for_token,
)
from app.auth.passwords import hash_password, verify_password
from app.conversation import create_session, get_session
from app.models import DocumentFormat, DocumentMeta, generate_document_id
from app.store.schema import (
    create_document,
    create_schema,
    get_connection,
    get_document,
    set_db_path,
)

PASSWORD = "correct horse battery"


@pytest.fixture
def conn(tmp_path):
    set_db_path(str(tmp_path / "auth.db"))
    c = get_connection()
    create_schema(c)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------


def test_password_hash_round_trip():
    encoded = hash_password(PASSWORD)
    assert encoded.startswith("scrypt$")
    assert PASSWORD not in encoded
    assert verify_password(PASSWORD, encoded)
    assert not verify_password("wrong password", encoded)


def test_password_hashes_are_salted():
    assert hash_password(PASSWORD) != hash_password(PASSWORD)


@pytest.mark.parametrize("encoded", ["", "plain", "bcrypt$1$2$3$4$5", "scrypt$x$8$1$aa$bb"])
def test_malformed_hash_never_verifies(encoded):
    assert verify_password(PASSWORD, encoded) is False


# ---------------------------------------------------------------------------
# Registration and sign-in
# ---------------------------------------------------------------------------


def test_register_normalises_email_and_defaults_display_name(conn):
    user = register_user(conn, "  Ada@Example.COM ", PASSWORD)
    assert user.email == "ada@example.com"
    assert user.display_name == "ada"
    stored = conn.execute("SELECT password_hash FROM users").fetchone()[0]
    assert PASSWORD not in stored


def test_email_is_unique_case_insensitively(conn):
    register_user(conn, "ada@example.com", PASSWORD)
    with pytest.raises(EmailTakenError):
        register_user(conn, "ADA@example.com", PASSWORD)


@pytest.mark.parametrize(
    ("email", "password", "name"),
    [
        ("not-an-email", PASSWORD, None),
        ("ada@example.com", "short", None),
        ("ada@example.com", "x" * 257, None),
        ("ada@example.com", PASSWORD, "n" * 81),
    ],
)
def test_invalid_registration_rejected(conn, email, password, name):
    with pytest.raises(AuthError):
        register_user(conn, email, password, name)


def test_authenticate(conn):
    user = register_user(conn, "ada@example.com", PASSWORD, "Ada")
    assert authenticate(conn, "ADA@example.com", PASSWORD) == user
    assert authenticate(conn, "ada@example.com", "wrong password") is None
    assert authenticate(conn, "nobody@example.com", PASSWORD) is None


def test_first_user_adopts_unowned_data_and_later_users_do_not(conn):
    create_document(conn, DocumentMeta(id="legacy-doc", name="old.pdf", format=DocumentFormat.PDF))
    legacy_session = create_session(conn)

    first = register_user(conn, "first@example.com", PASSWORD)
    second = register_user(conn, "second@example.com", PASSWORD)

    assert get_document(conn, "legacy-doc", owner_id=first.id) is not None
    assert get_document(conn, "legacy-doc", owner_id=second.id) is None
    assert get_session(conn, legacy_session.session_id, owner_id=first.id) is not None
    assert get_session(conn, legacy_session.session_id, owner_id=second.id) is None


def test_set_password_changes_password_and_signs_out_everywhere(conn):
    user = register_user(conn, "ada@example.com", PASSWORD)
    token = create_auth_session(conn, user.id)
    assert set_password(conn, "ada@example.com", "a brand new password")
    assert authenticate(conn, "ada@example.com", PASSWORD) is None
    assert authenticate(conn, "ada@example.com", "a brand new password") == user
    assert user_for_token(conn, token) is None
    assert set_password(conn, "nobody@example.com", "a brand new password") is False
    with pytest.raises(AuthError):
        set_password(conn, "ada@example.com", "short")


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def test_session_token_resolves_to_user_and_is_stored_hashed(conn):
    user = register_user(conn, "ada@example.com", PASSWORD)
    token = create_auth_session(conn, user.id)
    assert user_for_token(conn, token) == user
    stored = conn.execute("SELECT token_hash FROM auth_sessions").fetchone()[0]
    assert stored != token and token not in stored
    assert user_for_token(conn, "forged-token") is None
    assert user_for_token(conn, None) is None


def test_expired_session_is_rejected_and_removed(conn):
    user = register_user(conn, "ada@example.com", PASSWORD)
    token = create_auth_session(conn, user.id, lifetime=timedelta(seconds=-1))
    assert user_for_token(conn, token) is None
    assert conn.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0] == 0


def test_revoked_session_is_rejected(conn):
    user = register_user(conn, "ada@example.com", PASSWORD)
    token = create_auth_session(conn, user.id)
    other = create_auth_session(conn, user.id)
    revoke_auth_session(conn, token)
    assert user_for_token(conn, token) is None
    assert user_for_token(conn, other) == user  # other devices stay signed in


# ---------------------------------------------------------------------------
# Throttling
# ---------------------------------------------------------------------------


def test_throttle_locks_out_after_repeated_failures(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("app.auth.service.time.monotonic", lambda: clock[0])
    throttle = LoginThrottle(max_failures=3, lockout_seconds=60)
    for _ in range(3):
        throttle.check("Ada@example.com")
        throttle.record_failure("ada@example.com")
    with pytest.raises(TooManyAttemptsError) as exc:
        throttle.check("ADA@example.com")
    assert 0 < exc.value.retry_after_seconds <= 61
    throttle.check("someone-else@example.com")  # other accounts unaffected

    clock[0] += 61
    throttle.check("ada@example.com")  # lockout expired


def test_throttle_reset_on_success():
    throttle = LoginThrottle(max_failures=2, lockout_seconds=60)
    throttle.record_failure("ada@example.com")
    throttle.reset("ada@example.com")
    throttle.record_failure("ada@example.com")
    throttle.check("ada@example.com")


# ---------------------------------------------------------------------------
# Owner-scoped document IDs
# ---------------------------------------------------------------------------


def test_document_ids_are_owner_scoped_only_when_owned():
    assert generate_document_id("sample_spec.pdf") == "sample-spec-pdf-d58d38e8"  # eval IDs
    a = generate_document_id("sample_spec.pdf", "user-a")
    b = generate_document_id("sample_spec.pdf", "user-b")
    assert a != b
    assert a.startswith("sample-spec-pdf-") and b.startswith("sample-spec-pdf-")
