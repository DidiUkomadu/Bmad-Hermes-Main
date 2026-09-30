"""Application state shared by route handlers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass

from fastapi import Request

from app.api.errors import APIError
from app.config import Settings
from app.pipeline import ResearchPipeline
from app.store.schema import get_connection, get_document, set_db_path


@dataclass
class AppState:
    settings: Settings
    pipeline: ResearchPipeline
    llm_unavailable_reason: str | None = None


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


def require_llm(state: AppState) -> None:
    if state.llm_unavailable_reason:
        raise APIError(503, "LLM backend not configured", state.llm_unavailable_reason)


def require_document(conn: sqlite3.Connection, document_id: str | None) -> None:
    if document_id and get_document(conn, document_id) is None:
        raise APIError(404, "Document not found", document_id)
