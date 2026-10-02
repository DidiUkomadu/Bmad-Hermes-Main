"""Accounts, sign-in sessions, and password hashing — Epic 8."""

from app.auth.service import (
    AuthError,
    EmailTakenError,
    LoginThrottle,
    TooManyAttemptsError,
    User,
    authenticate,
    create_auth_session,
    register_user,
    revoke_auth_session,
    set_password,
    user_for_token,
)

__all__ = [
    "AuthError",
    "EmailTakenError",
    "LoginThrottle",
    "TooManyAttemptsError",
    "User",
    "authenticate",
    "create_auth_session",
    "register_user",
    "revoke_auth_session",
    "set_password",
    "user_for_token",
]
