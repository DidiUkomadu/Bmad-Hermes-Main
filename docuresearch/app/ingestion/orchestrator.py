"""Ingestion orchestration (Story 1.5).

Top-level ingestion: detect format, invoke the correct handler, chunk, persist.

Entry point: ingest_document()
"""

from __future__ import annotations

import logging
from pathlib import Path, PurePath
from typing import Any

from app.chunking.strategy import ChunkingConfig, chunk_passages
from app.ingestion.markdown import parse_markdown
from app.ingestion.pdf import extract_text_from_pdf
from app.ingestion.text import parse_plain_text
from app.models import DocumentFormat, RawDocument
from app.store.schema import (
    create_document,
    create_passage,
    create_schema,
    get_connection,
    set_db_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: dict[str, DocumentFormat] = {
    ".pdf": DocumentFormat.PDF,
    ".md": DocumentFormat.MARKDOWN,
    ".markdown": DocumentFormat.MARKDOWN,
    ".txt": DocumentFormat.TXT,
}


def detect_format(source: str | Path | bytes) -> DocumentFormat:
    """Detect the document format from a file path or extension hint.

    Args:
        source: A file path (str or Path) or raw bytes.

    Returns:
        The detected DocumentFormat.

    Raises:
        ValueError: If the format cannot be determined or is unsupported.
    """
    if isinstance(source, (str, Path)):
        ext = PurePath(source).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported document format: '{ext}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS.keys()))}"
            )
        return SUPPORTED_EXTENSIONS[ext]

    # bytes: try to detect by content
    if isinstance(source, bytes):
        if source.startswith(b"%PDF"):
            return DocumentFormat.PDF
        # Heuristic: if the content looks like markdown (has # headings),
        # treat as markdown. Otherwise fall back to txt.
        try:
            text_preview = source[:2048].decode("utf-8", errors="ignore")
        except Exception:
            text_preview = ""
        if text_preview.lstrip().startswith("#"):
            return DocumentFormat.MARKDOWN
        return DocumentFormat.TXT

    raise ValueError(f"Cannot detect format from source of type {type(source)}")


# ---------------------------------------------------------------------------
# Ingestion handlers
# ---------------------------------------------------------------------------


def _ingest_pdf(source: str | Path) -> RawDocument:
    """Ingest a PDF file and return a RawDocument."""
    path = Path(source)
    file_bytes = path.read_bytes()
    raw = extract_text_from_pdf(file_bytes, path.name)
    page_count = len(set(u.get("page_number", 0) for u in raw.units if u.get("page_number")))
    logger.info("Ingested PDF: %s (%d pages, %d chars)",
                raw.name, page_count, len(raw.full_text))
    return raw


def _ingest_markdown(source: str | Path) -> RawDocument:
    """Ingest a Markdown file and return a RawDocument."""
    path = Path(source)
    file_bytes = path.read_bytes()
    raw = parse_markdown(file_bytes, path.name)
    logger.info("Ingested Markdown: %s (%d blocks, %d chars)",
                raw.name, len(raw.units), len(raw.full_text))
    return raw


def _ingest_txt(source: str | Path) -> RawDocument:
    """Ingest a plain text file and return a RawDocument."""
    path = Path(source)
    file_bytes = path.read_bytes()
    raw = parse_plain_text(file_bytes, path.name)
    logger.info("Ingested TXT: %s (%d paragraphs, %d chars)",
                raw.name, len(raw.units), len(raw.full_text))
    return raw


_HANDLERS: dict[DocumentFormat, Any] = {
    DocumentFormat.PDF: _ingest_pdf,
    DocumentFormat.MARKDOWN: _ingest_markdown,
    DocumentFormat.TXT: _ingest_txt,
}


# ---------------------------------------------------------------------------
# Top-level ingestion
# ---------------------------------------------------------------------------


def ingest_document(
    source: str | Path,
    db_path: str | None = None,
    chunking_config: ChunkingConfig | None = None,
) -> dict[str, Any]:
    """Ingest a document: detect format, parse, chunk, and persist.

    This is the top-level entry point for document ingestion. It is
    synchronous and supports PDF, Markdown, and TXT formats only.

    Args:
        source: Path to the document file.
        db_path: Optional path to the SQLite database. If not provided,
            uses the currently configured database.
        chunking_config: Optional chunking configuration. Uses defaults
            if not provided.

    Returns:
        A dict with:
        - document_id: the persisted document ID
        - document_name: the document name
        - format: the detected format
        - passage_count: number of passages created
        - passage_ids: list of passage IDs
        - db_path: the database path used
    """
    source_path = Path(source)

    if not source_path.exists():
        raise FileNotFoundError(f"Document not found: {source_path}")

    if not source_path.is_file():
        raise ValueError(f"Source is not a file: {source_path}")

    # 1. Detect format
    fmt = detect_format(source_path)
    logger.info("Detected format '%s' for %s", fmt.value, source_path)

    # 2. Ingest with the appropriate handler
    raw = _HANDLERS[fmt](source_path)

    # 3. Chunk
    config = chunking_config if chunking_config is not None else ChunkingConfig()
    passages = chunk_passages(raw, config)
    logger.info("Chunked %s into %d passages", raw.name, len(passages))

    # 4. Persist
    if db_path is not None:
        set_db_path(db_path)
    conn = get_connection()

    # Ensure schema exists
    create_schema(conn)

    # Determine page_count for DocumentMeta
    if fmt == DocumentFormat.PDF:
        page_count = len(set(u.get("page_number", 0) for u in raw.units if u.get("page_number")))
    else:
        page_count = None

    from app.models import DocumentMeta
    doc_meta = DocumentMeta(
        id=raw.document_id,
        name=raw.name,
        format=raw.format,
        uploaded_at=raw.uploaded_at,
        page_count=page_count,
    )
    create_document(conn, doc_meta)

    passage_ids: list[str] = []
    for p in passages:
        create_passage(conn, p)
        passage_ids.append(p.id)

    conn.commit()

    logger.info("Persisted %s: %d passages in %s",
                raw.name, len(passages), db_path or "default db")

    return {
        "document_id": raw.document_id,
        "document_name": raw.name,
        "format": raw.format.value,
        "passage_count": len(passages),
        "passage_ids": passage_ids,
        "db_path": db_path,
    }


def ingest_document_bytes(
    source_bytes: bytes,
    filename: str,
    db_path: str | None = None,
    chunking_config: ChunkingConfig | None = None,
) -> dict[str, Any]:
    """Ingest a document from raw bytes with a given filename.

    This is useful for API upload scenarios where the file content
    comes as bytes rather than a file path. The filename is used for
    format detection and document naming.

    Args:
        source_bytes: The raw file content.
        filename: The original filename (used for format detection).
        db_path: Optional path to the SQLite database.
        chunking_config: Optional chunking configuration.

    Returns:
        Same structure as ingest_document().
    """
    fmt = detect_format(filename)

    # Write to a temporary file for the handlers (they expect paths)
    import tempfile
    with tempfile.NamedTemporaryFile(
        suffix=PurePath(filename).suffix,
        delete=False,
        mode="wb",
    ) as tmp:
        tmp.write(source_bytes)
        tmp_path = tmp.name

    try:
        # Set the raw document name from the filename
        # The handlers will use the temp path as the name; fix it after
        result = ingest_document(
            tmp_path,
            db_path=db_path,
            chunking_config=chunking_config,
        )
        # Override the document_name in the result
        result["document_name"] = filename
        return result
    finally:
        import os
        os.unlink(tmp_path)


__all__ = [
    "detect_format",
    "ingest_document",
    "ingest_document_bytes",
    "SUPPORTED_EXTENSIONS",
]
