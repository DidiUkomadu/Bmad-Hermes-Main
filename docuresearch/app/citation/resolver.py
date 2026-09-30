"""Citation resolution — used by the citation lookup API (story 7.4) and UI (Epic 6).

Resolves a passage ID to the passage text exactly as stored in the content
store, plus its document name and location. Nothing here is generated.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

_CHUNK_SUFFIX = re.compile(r":chunk:\d+$")


def format_location(location: str) -> str:
    """Human-readable form of a stored passage location.

    Stored locations carry a chunk suffix used for stable passage IDs:
        page:2:chunk:1                 -> "Page 2"
        paragraph:5:chunk:4            -> "Paragraph 5"
        Doc / Security:chunk:1         -> "Section: Doc › Security"
        ¶chunk:3                       -> "Passage 4"
    Unrecognised values are returned unchanged.
    """
    if location.startswith("¶chunk:"):
        index = location.removeprefix("¶chunk:")
        return f"Passage {int(index) + 1}" if index.isdigit() else location
    base = _CHUNK_SUFFIX.sub("", location)
    kind, _, number = base.partition(":")
    if kind == "page" and number.isdigit():
        return f"Page {number}"
    if kind == "paragraph" and number.isdigit():
        return f"Paragraph {number}"
    if base and base != location:
        return "Section: " + " › ".join(part.strip() for part in base.split(" / "))
    return location


@dataclass(frozen=True)
class ResolvedPassage:
    passage_id: str
    document_id: str
    document_name: str
    location: str
    text: str
    start_offset: int
    end_offset: int

    @property
    def location_label(self) -> str:
        return format_location(self.location)


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
