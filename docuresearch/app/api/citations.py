"""Citation lookup — Story 7.4 (implementation plan §5.5).

Passage IDs can contain "/" and spaces (Markdown section paths), so the
route uses a path converter; clients URL-encode the ID.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends

from app.api.deps import get_conn
from app.api.errors import APIError
from app.api.schemas import PassageResponse
from app.citation import resolve_passage

router = APIRouter(prefix="/citations", tags=["citations"])


@router.get("/{passage_id:path}", response_model=PassageResponse)
def get_citation(
    passage_id: str,
    conn: sqlite3.Connection = Depends(get_conn),
) -> PassageResponse:
    """Return a passage exactly as stored, with its document name and location."""
    passage = resolve_passage(conn, passage_id)
    if passage is None:
        raise APIError(404, "Passage not found", passage_id)
    return PassageResponse(
        passage_id=passage.passage_id,
        document_id=passage.document_id,
        document_name=passage.document_name,
        location=passage.location,
        text=passage.text,
        start_offset=passage.start_offset,
        end_offset=passage.end_offset,
    )
