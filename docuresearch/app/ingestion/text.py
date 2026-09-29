"""Plain text ingestion with paragraph-based chunking (Story 1.3).

Splits plain text on paragraph boundaries (blank-line-separated groups).
Provides paragraph index and character offset locators.

API:
    parse_plain_text(file_bytes: bytes, filename: str) -> RawDocument
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from app.models import DocumentFormat, RawDocument, generate_document_id

logger = logging.getLogger(__name__)

# Minimum characters to consider a group a "paragraph" (below this, treat
# as a line group within the previous paragraph)
MIN_PARAGRAPH_CHARS = 10


def parse_plain_text(file_bytes: bytes, filename: str) -> RawDocument:
    """Ingest a plain text file, splitting into paragraph-based units.

    Args:
        file_bytes: The raw bytes of the text file.
        filename: Original filename, used for the document name.

    Returns:
        A RawDocument with paragraph-based units, each with paragraph index
        and character offset range as the locator.
    """
    document_id = generate_document_id(filename)
    name = filename

    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")

    # Split on blank-line boundaries (one or more empty lines)
    raw_paragraphs = _split_paragraphs(text)

    units: list[dict[str, Any]] = []
    full_text_parts: list[str] = []
    offset = 0
    para_index = 0

    for para_text in raw_paragraphs:
        para_text = para_text.strip()
        if not para_text:
            continue

        para_index += 1
        full_text_parts.append(para_text)

        # Locator: paragraph index + offset range
        units.append(
            {
                "type": "paragraph",
                "text": para_text,
                "location": f"paragraph:{para_index}",
                "para_index": para_index,
                "line_start": _count_lines_before(text, offset),
                "start_offset": offset,
                "end_offset": offset + len(para_text),
            }
        )
        offset += len(para_text) + 2  # \n\n separator

    full_text = "\n\n".join(full_text_parts)

    return RawDocument(
        document_id=document_id,
        name=name,
        format=DocumentFormat.TXT,
        uploaded_at=_utc_now(),
        full_text=full_text,
        units=units,
    )


def _split_paragraphs(text: str) -> list[str]:
    """Split text on blank-line boundaries.

    A blank line is one or more consecutive newlines surrounded by whitespace.
    Consecutive non-empty lines form a paragraph group.
    """
    import re

    # Split on two or more newlines (blank line separator)
    parts = re.split(r"\n\s*\n", text)

    result: list[str] = []
    for part in parts:
        stripped = part.strip()
        if stripped:
            result.append(stripped)

    # Fallback: if no blank-line splits found, split on single newlines
    # but only if the text has meaningful line breaks
    if not result and text.strip():
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            result = ["\n".join(lines)]

    return result


def _count_lines_before(text: str, offset: int) -> int:
    """Count how many lines come before a given character offset."""
    return text[:offset].count("\n") + 1


def _utc_now():
    return datetime.now(UTC)


def paragraph_count(units: list[dict[str, Any]]) -> int:
    """Return the number of paragraphs in parsed text units."""
    return sum(1 for u in units if u.get("type") == "paragraph")
