"""Document upload, listing, and removal — Stories 7.2 and 7.5.

Ingestion is synchronous (implementation plan §14.5): upload returns 200 with
status "available" once the document and all its passages are stored.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import PurePath

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pypdf.errors import PyPdfError

from app.api.deps import AppState, current_user, get_conn, get_state
from app.api.errors import APIError
from app.api.schemas import (
    DocumentListResponse,
    DocumentResponse,
    RemovalResponse,
    UploadResponse,
)
from app.auth import User
from app.ingestion.orchestrator import (
    SUPPORTED_EXTENSIONS,
    DuplicateDocumentError,
    detect_format,
)
from app.models import DocumentMeta
from app.store.schema import delete_document, get_document, list_documents, rename_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


def _document_response(doc: DocumentMeta) -> DocumentResponse:
    return DocumentResponse(
        document_id=doc.id,
        name=doc.name,
        format=doc.format.value,
        uploaded_at=doc.uploaded_at,
        page_count=doc.page_count,
        section_count=doc.section_count,
    )


@router.post("/upload", response_model=UploadResponse)
def upload_document(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    state: AppState = Depends(get_state),
    conn: sqlite3.Connection = Depends(get_conn),
    user: User = Depends(current_user),
) -> UploadResponse:
    """Upload a PDF, Markdown, or TXT file and ingest it."""
    filename = PurePath(file.filename or "").name
    try:
        detect_format(filename)
    except ValueError as exc:
        raise APIError(
            400,
            "Invalid file format",
            f"'{filename or '(no filename)'}' is not supported. Supported extensions: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        ) from exc

    data = file.file.read()
    if not data:
        raise APIError(400, "Empty file", f"'{filename}' contains no data")

    try:
        result = state.pipeline.ingest_bytes(data, filename, owner_id=user.id)
    except DuplicateDocumentError as exc:
        raise APIError(409, "Document already exists", str(exc)) from exc
    except (ValueError, PyPdfError) as exc:
        # Covers corrupt files, undecodable text, and documents with no
        # extractable text (EmptyDocumentError). Nothing is stored on failure.
        logger.warning("Extraction failed for %s: %s", filename, exc)
        raise APIError(422, "Extraction failed", str(exc)) from exc

    document_id = result["document_id"]
    if name and name.strip():
        rename_document(conn, document_id, name.strip(), owner_id=user.id)

    doc = get_document(conn, document_id, owner_id=user.id)
    assert doc is not None  # just ingested
    return UploadResponse(
        **_document_response(doc).model_dump(), passage_count=result["passage_count"]
    )


@router.get("", response_model=DocumentListResponse)
def list_uploaded_documents(
    conn: sqlite3.Connection = Depends(get_conn),
    user: User = Depends(current_user),
) -> DocumentListResponse:
    """List the signed-in user's documents, oldest first."""
    docs = sorted(list_documents(conn, owner_id=user.id), key=lambda d: d.uploaded_at)
    return DocumentListResponse(documents=[_document_response(d) for d in docs])


@router.delete("/{document_id}", response_model=RemovalResponse)
def remove_document(
    document_id: str,
    conn: sqlite3.Connection = Depends(get_conn),
    user: User = Depends(current_user),
) -> RemovalResponse:
    """Remove one of the signed-in user's documents and all its passages."""
    if not delete_document(conn, document_id, owner_id=user.id):
        raise APIError(404, "Document not found", document_id)
    return RemovalResponse(document_id=document_id)
