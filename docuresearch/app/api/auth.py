"""Sign-in endpoints — Epic 8 (stories 8.1, 8.2).

The session token is only ever sent as an HttpOnly, SameSite=Lax cookie, so
page scripts cannot read it and other sites cannot submit requests with it
(change proposal D3).
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta

from fastapi import APIRouter, Depends, Request, Response

from app.api.deps import SESSION_COOKIE, AppState, current_user, get_conn, get_state
from app.api.errors import APIError
from app.api.schemas import LoginRequest, RegisterRequest, UserResponse
from app.auth import (
    AuthError,
    EmailTakenError,
    TooManyAttemptsError,
    User,
    authenticate,
    create_auth_session,
    register_user,
    revoke_auth_session,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def _user_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id, email=user.email, display_name=user.display_name, created_at=user.created_at
    )


def _start_session(response: Response, conn: sqlite3.Connection, state: AppState, user: User):
    lifetime = timedelta(days=state.settings.session_lifetime_days)
    token = create_auth_session(conn, user.id, lifetime)
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=int(lifetime.total_seconds()),
        httponly=True,
        samesite="lax",
        secure=state.settings.cookie_secure,
        path="/",
    )


@router.post("/register", response_model=UserResponse, status_code=201)
def register(
    req: RegisterRequest,
    response: Response,
    state: AppState = Depends(get_state),
    conn: sqlite3.Connection = Depends(get_conn),
) -> UserResponse:
    """Create an account and sign in."""
    if not state.settings.allow_registration:
        raise APIError(403, "Registration is closed", "Ask the administrator for an account.")
    try:
        user = register_user(conn, req.email, req.password, req.display_name)
    except EmailTakenError as exc:
        raise APIError(409, "Email already registered", str(exc)) from exc
    except AuthError as exc:
        raise APIError(400, "Invalid registration", str(exc)) from exc
    _start_session(response, conn, state, user)
    return _user_response(user)


@router.post("/login", response_model=UserResponse)
def login(
    req: LoginRequest,
    response: Response,
    state: AppState = Depends(get_state),
    conn: sqlite3.Connection = Depends(get_conn),
) -> UserResponse:
    """Sign in with email and password."""
    try:
        state.login_throttle.check(req.email)
    except TooManyAttemptsError as exc:
        raise APIError(429, "Too many attempts", str(exc)) from exc
    user = authenticate(conn, req.email, req.password)
    if user is None:
        state.login_throttle.record_failure(req.email)
        raise APIError(401, "Invalid email or password", None)
    state.login_throttle.reset(req.email)
    _start_session(response, conn, state, user)
    return _user_response(user)


@router.post("/logout", status_code=204)
def logout(request: Request, conn: sqlite3.Connection = Depends(get_conn)) -> Response:
    """Sign out: the session is deleted on the server and the cookie cleared."""
    revoke_auth_session(conn, request.cookies.get(SESSION_COOKIE))
    response = Response(status_code=204)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@router.get("/me", response_model=UserResponse)
def me(user: User = Depends(current_user)) -> UserResponse:
    """The signed-in user (401 when signed out)."""
    return _user_response(user)
