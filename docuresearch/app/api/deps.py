"""Application state shared by route handlers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field

from fastapi import Depends, Request

from app.api.errors import APIError
from app.auth import LoginThrottle, User, user_for_token
from app.config import Settings
from app.pipeline import ResearchPipeline
from app.store.schema import get_connection, get_document, set_db_path

SESSION_COOKIE = "docuresearch_session"


@dataclass
class AppState:
    settings: Settings
    pipeline: ResearchPipeline
    llm_unavailable_reason: str | None = None
    login_throttle: LoginThrottle = field(default_factory=LoginThrottle)


def get_state(request: Request) -> AppState:
    return request.app.state.docuresearch


def get_conn(request: Request) -> Iterator[sqlite3.Connection]:
    """A store connection for the duration of one request.

    FastAPI may run this dependency and the (sync) handler on different worker
    threads, so the same-thread check is disabled. The connection is still
    private to this request and used sequentially.
    """
    set_db_path(get_state(request).pipeline.db_path)
    conn = get_connection(check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


def current_user(
    request: Request, conn: sqlite3.Connection = Depends(get_conn)
) -> User:
    """The signed-in user for this request; 401 if there is no valid session."""
    user = user_for_token(conn, request.cookies.get(SESSION_COOKIE))
    if user is None:
        raise APIError(401, "Not signed in", "Sign in to continue.")
    return user


def require_llm(state: AppState) -> None:
    if state.llm_unavailable_reason:
        raise APIError(503, "LLM backend not configured", state.llm_unavailable_reason)


def require_document(conn: sqlite3.Connection, document_id: str | None, owner_id: str) -> None:
    """404 unless *document_id* (if given) exists and belongs to *owner_id*."""
    if document_id and get_document(conn, document_id, owner_id=owner_id) is None:
        raise APIError(404, "Document not found", document_id)
