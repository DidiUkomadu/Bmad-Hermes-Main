"""SQLite content store schema (Story 2.1).

Defines the documents and passages tables plus the repository interface.
Schema aligns with the architecture data model (§3).

Tables:
    documents — id, name, format, uploaded_at, page_count, section_count
    passages  — id, document_id (FK), text, location, start_offset, end_offset,
                embedding (nullable)
    conversations      — session_id, created_at, current_document_scope (JSON, nullable)
    conversation_turns — session_id (FK, cascade), turn_index, user_query, answer_text,
                         citations (JSON), evidence_quality, is_abstention, created_at
                         (managed by app.conversation.session, Story 5.1)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from app.models import DocumentFormat, DocumentMeta, Passage

# SQLite database path — set at runtime by the application
DB_PATH: str | None = None


def set_db_path(path: str) -> None:
    """Set the SQLite database file path."""
    global DB_PATH
    DB_PATH = path


def get_db_path() -> str:
    """Return the current database path, raising if not configured."""
    if DB_PATH is None:
        raise RuntimeError("Database path not configured. Call set_db_path() first.")
    return DB_PATH


def get_connection(check_same_thread: bool = True) -> sqlite3.Connection:
    """Get a connection to the SQLite database.

    Args:
        check_same_thread: Passed to ``sqlite3.connect``. Set False only for a
            connection owned by one unit of work that may hop threads but is
            never used concurrently (e.g. one API request, whose dependency and
            handler can run on different worker threads).
    """
    path = get_db_path()
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the documents and passages tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            format TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            page_count INTEGER,
            section_count INTEGER
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS passages (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            text TEXT NOT NULL,
            location TEXT NOT NULL,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            embedding TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        )
    """)

    # Index for fast passage lookup by ID
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_passages_id ON passages(id)
    """)

    # Index for listing passages by document
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_passages_doc ON passages(document_id)
    """)

    # Conversation sessions and turns (Story 5.1, implementation plan §6.3–6.4)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            current_document_scope TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL
                REFERENCES conversations(session_id) ON DELETE CASCADE,
            turn_index INTEGER NOT NULL,
            user_query TEXT NOT NULL,
            answer_text TEXT NOT NULL,
            citations TEXT NOT NULL,
            evidence_quality TEXT NOT NULL CHECK (
                evidence_quality IN ('sufficient', 'partial', 'insufficient', 'conflicting')
            ),
            is_abstention INTEGER NOT NULL CHECK (is_abstention IN (0, 1)),
            created_at TEXT NOT NULL,
            UNIQUE(session_id, turn_index)
        )
    """)

    conn.commit()


# ---------------------------------------------------------------------------
# Repository operations (Story 2.1 + 2.2)
# ---------------------------------------------------------------------------

def create_document(conn: sqlite3.Connection, doc: DocumentMeta) -> DocumentMeta:
    """Insert a document record and return it."""
    conn.execute(
        """INSERT INTO documents (id, name, format, uploaded_at, page_count, section_count)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            doc.id,
            doc.name,
            doc.format.value,
            doc.uploaded_at.isoformat(),
            doc.page_count,
            doc.section_count,
        ),
    )
    conn.commit()
    return doc


def get_document(conn: sqlite3.Connection, doc_id: str) -> DocumentMeta | None:
    """Retrieve a document by ID, or None if not found."""
    row = conn.execute(
        "SELECT * FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    if row is None:
        return None
    return _row_to_document(row)


def list_documents(conn: sqlite3.Connection) -> list[DocumentMeta]:
    """List all documents ordered by upload time ( newest first )."""
    rows = conn.execute(
        "SELECT * FROM documents ORDER BY uploaded_at DESC"
    ).fetchall()
    return [_row_to_document(r) for r in rows]


def create_passage(conn: sqlite3.Connection, passage: Passage) -> Passage:
    """Insert a passage record."""
    embedding_blob = None
    if passage.embedding is not None:
        import json
        embedding_blob = json.dumps(passage.embedding)

    conn.execute(
        """INSERT INTO passages
               (id, document_id, text, location, start_offset, end_offset, embedding)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            passage.id,
            passage.document_id,
            passage.text,
            passage.location,
            passage.start_offset,
            passage.end_offset,
            embedding_blob,
        ),
    )
    conn.commit()
    return passage


def get_passage(conn: sqlite3.Connection, passage_id: str) -> Passage | None:
    """Retrieve a single passage by ID, or None if not found."""
    row = conn.execute(
        "SELECT * FROM passages WHERE id = ?", (passage_id,)
    ).fetchone()
    if row is None:
        return None
    return _row_to_passage(row)


def list_passages_for_document(
    conn: sqlite3.Connection, doc_id: str
) -> list[Passage]:
    """Retrieve all passages for a given document."""
    rows = conn.execute(
        "SELECT * FROM passages WHERE document_id = ? ORDER BY start_offset",
        (doc_id,),
    ).fetchall()
    return [_row_to_passage(r) for r in rows]


def rename_document(conn: sqlite3.Connection, doc_id: str, name: str) -> bool:
    """Set a document's user-visible name. Returns False if it does not exist."""
    cur = conn.execute("UPDATE documents SET name = ? WHERE id = ?", (name, doc_id))
    conn.commit()
    return cur.rowcount > 0


def delete_document(conn: sqlite3.Connection, doc_id: str) -> bool:
    """Delete a document and all its passages (CASCADE).

    Returns True if a document was deleted, False if the document didn't exist.
    """
    # DELETE on documents triggers CASCADE on passages (ON DELETE CASCADE)
    cur = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    return cur.rowcount > 0


def _row_to_document(row: sqlite3.Row) -> DocumentMeta:
    """Convert a database row to a DocumentMeta."""
    return DocumentMeta(
        id=row["id"],
        name=row["name"],
        format=DocumentFormat(row["format"]),
        uploaded_at=_parse_dt(row["uploaded_at"]),
        page_count=row["page_count"],
        section_count=row["section_count"],
    )


def _row_to_passage(row: sqlite3.Row) -> Passage:
    """Convert a database row to a Passage."""
    embedding: list[float] | None = None
    if row["embedding"] is not None:
        import json
        try:
            embedding = json.loads(row["embedding"])
        except (json.JSONDecodeError, TypeError):
            embedding = None

    return Passage(
        id=row["id"],
        document_id=row["document_id"],
        text=row["text"],
        location=row["location"],
        start_offset=row["start_offset"],
        end_offset=row["end_offset"],
        embedding=embedding,
    )


def _parse_dt(iso: str) -> datetime:
    """Parse an ISO-format datetime string."""
    return datetime.fromisoformat(iso)
