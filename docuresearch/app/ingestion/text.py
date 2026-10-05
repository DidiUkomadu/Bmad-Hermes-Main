"""Plain text ingestion with paragraph-based chunking (Story 1.3).

Splits plain text on paragraph boundaries (blank-line-separated groups).
Provides paragraph index and character offset locators.

Paginated plain text (e.g. IETF RFCs) is cleaned before chunking, because
running page headers/footers become tiny passages that crowd out real content
in keyword retrieval, and headings separated from their text are unfindable:
- Running heads: short one-line paragraphs that repeat at least
  ``RUNNING_HEAD_MIN_REPEATS`` times (ignoring digits, e.g. page numbers) are
  dropped, as are form-feed characters.
- Numbered section headings ("4.1.4. ...", "Appendix A. ...") are joined to
  the paragraph that follows them.

API:
    parse_plain_text(file_bytes: bytes, filename: str) -> RawDocument
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from app.models import DocumentFormat, RawDocument, generate_document_id

logger = logging.getLogger(__name__)

# Minimum characters to consider a group a "paragraph" (below this, treat
# as a line group within the previous paragraph)
MIN_PARAGRAPH_CHARS = 10

RUNNING_HEAD_MIN_REPEATS = 3
RUNNING_HEAD_MAX_CHARS = 100
_NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\.?|Appendix [A-Z](?:\.\d+)*\.?)\s+\S")


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

    # Split on blank-line boundaries (one or more empty lines), then clean up
    # pagination artefacts.
    raw_paragraphs = _attach_numbered_headings(
        _drop_running_heads(_split_paragraphs(text.replace("\f", "\n")))
    )

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


def _running_head_key(paragraph: str) -> str | None:
    """Normalised form of a possible page header/footer, or None if it can't be one."""
    if "\n" in paragraph or len(paragraph) > RUNNING_HEAD_MAX_CHARS:
        return None
    return re.sub(r"\d+", "#", " ".join(paragraph.split()))


def _drop_running_heads(paragraphs: list[str]) -> list[str]:
    """Remove short one-line paragraphs repeated on many pages (headers/footers)."""
    counts = Counter(k for p in paragraphs if (k := _running_head_key(p)) is not None)
    repeated = {k for k, n in counts.items() if n >= RUNNING_HEAD_MIN_REPEATS}
    if not repeated:
        return paragraphs
    kept = [p for p in paragraphs if _running_head_key(p) not in repeated]
    logger.info("Dropped %d running header/footer paragraphs", len(paragraphs) - len(kept))
    return kept


def _attach_numbered_headings(paragraphs: list[str]) -> list[str]:
    """Join a numbered section heading to the paragraph that follows it."""
    result: list[str] = []
    pending: str | None = None
    for p in paragraphs:
        is_heading = "\n" not in p and len(p) <= RUNNING_HEAD_MAX_CHARS and bool(
            _NUMBERED_HEADING.match(p)
        )
        if pending is not None:
            if is_heading:
                result.append(pending)  # consecutive headings: keep the first alone
                pending = p
                continue
            result.append(f"{pending}\n{p}")
            pending = None
        elif is_heading:
            pending = p
        else:
            result.append(p)
    if pending is not None:
        result.append(pending)
    return result


def _count_lines_before(text: str, offset: int) -> int:
    """Count how many lines come before a given character offset."""
    return text[:offset].count("\n") + 1


def _utc_now():
    return datetime.now(UTC)


def paragraph_count(units: list[dict[str, Any]]) -> int:
    """Return the number of paragraphs in parsed text units."""
    return sum(1 for u in units if u.get("type") == "paragraph")
