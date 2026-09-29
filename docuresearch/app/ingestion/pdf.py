"""PDF text extraction with page boundaries (Story 1.1).

Uses pypdf to extract text from text-based PDFs, preserving page boundaries.
Supports text-based PDFs only — scanned/image PDFs will extract nothing or fail
clearly, not return garbage.

API:
    extract_text_from_pdf(file_bytes: bytes, filename: str) -> RawDocument
"""

from __future__ import annotations

import io
import logging
from datetime import UTC, datetime
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.models import DocumentFormat, RawDocument, generate_document_id

logger = logging.getLogger(__name__)

# Max pages we'll process without warning — protects against huge PDFs
MAX_PAGES_WARNING = 500


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> RawDocument:
    """Extract text from a text-based PDF, preserving page boundaries.

    Args:
        file_bytes: The raw bytes of the PDF file.
        filename: Original filename, used for the document name.

    Returns:
        A RawDocument with per-page text units and page numbers as locations.

    Raises:
        PdfReadError: If the PDF is corrupt or unreadable.
        ValueError: If the file is not a valid PDF.
    """
    document_id = generate_document_id(filename)
    name = filename

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except PdfReadError as e:
        logger.error("Failed to read PDF %s: %s", filename, e)
        raise
    except Exception as e:
        logger.error("Unexpected error reading PDF %s: %s", filename, e)
        raise ValueError(f"Unable to read PDF file: {e}") from e

    num_pages = len(reader.pages)
    if num_pages > MAX_PAGES_WARNING:
        logger.warning(
            "PDF %s has %d pages — exceeding warning threshold of %d",
            filename, num_pages, MAX_PAGES_WARNING,
        )

    units: list[dict[str, Any]] = []
    full_text_parts: list[str] = []

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            logger.warning(
                "Failed to extract text from page %d of %s: %s",
                page_num, filename, e,
            )
            page_text = ""

        page_text = page_text.strip()
        if page_text:
            full_text_parts.append(page_text)
            units.append(
                {
                    "text": page_text,
                    "location": f"page:{page_num}",
                    "page_number": page_num,
                    "start_offset": sum(len(t) for t in full_text_parts[:-1]),
                    "end_offset": sum(len(t) for t in full_text_parts),
                }
            )
        else:
            # Page with no extractable text — still record it for page count
            units.append(
                {
                    "text": "",
                    "location": f"page:{page_num}",
                    "page_number": page_num,
                    "start_offset": sum(len(t) for t in full_text_parts),
                    "end_offset": sum(len(t) for t in full_text_parts),
                }
            )

    full_text = "\n\n".join(full_text_parts)

    return RawDocument(
        document_id=document_id,
        name=name,
        format=DocumentFormat.PDF,
        uploaded_at=datetime.now(UTC),
        full_text=full_text,
        units=units,
    )


def get_page_count(file_bytes: bytes) -> int:
    """Return the number of pages in a PDF without full extraction."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        return len(reader.pages)
    except Exception:
        return 0
