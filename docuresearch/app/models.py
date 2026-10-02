"""Shared data models for DocuResearch.

These types are used across ingestion, chunking, store, and evaluation.
They represent the contracts between layers as defined by the architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class DocumentFormat(StrEnum):
    """Supported document formats for MVP."""

    PDF = "pdf"
    MARKDOWN = "markdown"
    TXT = "txt"


@dataclass
class DocumentMeta:
    """Document-level metadata (architecture §3 Document)."""

    id: str
    name: str
    format: DocumentFormat
    uploaded_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    page_count: int | None = None
    section_count: int | None = None
    owner_id: str | None = None  # user who uploaded it; None for unowned (legacy/eval) data

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "format": self.format.value,
            "uploaded_at": self.uploaded_at.isoformat(),
            "page_count": self.page_count,
            "section_count": self.section_count,
        }


@dataclass
class Passage:
    """A retrievable and citable text unit (architecture §3 Passage).

    The id is document-scoped and incorporates location information, e.g.
    ``{doc_id}:page:{page_num}:chunk:{chunk_idx}``.
    """

    id: str
    document_id: str
    text: str
    location: str
    start_offset: int
    end_offset: int
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,
            "location": self.location,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }


@dataclass
class RawDocument:
    """Output of a format-specific ingestion handler.

    This is the intermediate representation passed to the chunker. It is not
    stored directly — the chunker converts it into :class:`Passage` objects.
    """

    document_id: str
    name: str
    format: DocumentFormat
    uploaded_at: datetime
    # Full text of the document (concatenation of all pages/blocks/paragraphs)
    full_text: str
    # Format-specific units, each with text + location info
    units: list[dict[str, Any]]

    def __post_init__(self) -> None:
        # Recompute full_text from units if not explicitly set, to stay consistent
        if not self.full_text and self.units:
            self.full_text = "".join(u["text"] for u in self.units)


def generate_document_id(name: str, owner_id: str | None = None) -> str:
    """Generate a stable, deterministic document ID from a name.

    Uses an MD5 hash of the filename to produce a stable ID that is
    identical across re-ingestion of the same file. This makes the
    evaluation dataset's gold passage references stable across runs.
    The ID format is compatible with the UUID4 contract (a short
    hex string) while being deterministic rather than random.

    When *owner_id* is given (uploads by a signed-in user), it is part of the
    hash, so two users uploading the same filename get different IDs. Without
    it the ID depends on the filename only, keeping evaluation IDs stable.
    """
    import hashlib
    import re

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip()).strip("-").lower()
    slug = re.sub(r"-+", "-", slug)
    if not slug:
        slug = "document"
    slug = slug[:40]
    # Deterministic suffix from hash of the filename
    key = f"{owner_id}:{name}" if owner_id else name
    suffix = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{suffix}"


def generate_passage_id(document_id: str, location: str, chunk_index: int) -> str:
    """Generate a stable passage ID scoped to a document and location."""
    return f"{document_id}:{location}:chunk:{chunk_index}"
