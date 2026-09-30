"""Citation resolution — used by the citation lookup API (story 7.4) and UI (Epic 6).

Resolves a passage ID to the passage text exactly as stored in the content
store, plus its document name and location. Nothing here is generated.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedPassage:
    passage_id: str
    document_id: str
    document_name: str
    location: str
    text: str
    start_offset: int
    end_offset: int


def resolve_passage(conn: sqlite3.Connection, passage_id: str) -> ResolvedPassage | None:
    """Return the stored passage and its document name, or None if unknown."""
    row = conn.execute(
        "SELECT p.id, p.document_id, d.name AS document_name, p.location, p.text, "
        "p.start_offset, p.end_offset "
        "FROM passages p JOIN documents d ON p.document_id = d.id WHERE p.id = ?",
        (passage_id,),
    ).fetchone()
    if row is None:
        return None
    return ResolvedPassage(
        passage_id=row["id"],
        document_id=row["document_id"],
        document_name=row["document_name"],
        location=row["location"],
        text=row["text"],
        start_offset=row["start_offset"],
        end_offset=row["end_offset"],
    )
