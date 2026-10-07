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
    users              — id, email (unique, case-insensitive), display_name,
                         password_hash, created_at (Epic 8)
    auth_sessions      — token_hash, user_id (FK, cascade), created_at, expires_at
    usage_counters     — day (UTC date), scope (user/site/refund), user_id, count
                         (managed by app.limits.usage, Story 9.1)

Ownership (Epic 8): documents and conversations carry an ``owner_id``. Repository
functions take an optional ``owner_id``; when given, only that owner's rows are
visible. ``None`` means unfiltered, used only by trusted internal callers such
as the evaluation runner, never by the API.
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
            evidence_quality_narrative TEXT NOT NULL DEFAULT '',
            UNIQUE(session_id, turn_index)
        )
    """)
    _add_column_if_missing(
        conn, "conversation_turns", "evidence_quality_narrative", "TEXT NOT NULL DEFAULT ''"
    )

    # Accounts and sign-in sessions (Epic 8)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            display_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS auth_sessions (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
    """)

    # Ownership; NULL for data created before accounts existed (adopted by the
    # first registered user, see app.auth.service).
    _add_column_if_missing(conn, "documents", "owner_id", "TEXT")
    _add_column_if_missing(conn, "conversations", "owner_id", "TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations(owner_id)")

    # Daily question counts (Story 9.1, managed by app.limits.usage)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_counters (
            day TEXT NOT NULL,
            scope TEXT NOT NULL,
            user_id TEXT NOT NULL,
            count INTEGER NOT NULL,
            PRIMARY KEY (day, scope, user_id)
        )
    """)

    conn.commit()


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, definition: str
) -> None:
    """Minimal migration for databases created before a column existed."""
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}  # 1 = name
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


# ---------------------------------------------------------------------------
# Repository operations (Story 2.1 + 2.2)
# ---------------------------------------------------------------------------

def create_document(conn: sqlite3.Connection, doc: DocumentMeta) -> DocumentMeta:
    """Insert a document record and return it."""
    conn.execute(
        """INSERT INTO documents
               (id, name, format, uploaded_at, page_count, section_count, owner_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            doc.id,
            doc.name,
            doc.format.value,
            doc.uploaded_at.isoformat(),
            doc.page_count,
            doc.section_count,
            doc.owner_id,
        ),
    )
    conn.commit()
    return doc


def _owner_clause(owner_id: str | None) -> tuple[str, tuple[str, ...]]:
    """SQL fragment restricting documents to an owner (no restriction for None)."""
    return ("", ()) if owner_id is None else (" AND owner_id = ?", (owner_id,))


def get_document(
    conn: sqlite3.Connection, doc_id: str, owner_id: str | None = None
) -> DocumentMeta | None:
    """Retrieve a document by ID, or None if not found (or not owned by *owner_id*)."""
    clause, args = _owner_clause(owner_id)
    row = conn.execute(
        f"SELECT * FROM documents WHERE id = ?{clause}", (doc_id, *args)
    ).fetchone()
    if row is None:
        return None
    return _row_to_document(row)


def list_documents(
    conn: sqlite3.Connection, owner_id: str | None = None
) -> list[DocumentMeta]:
    """List documents (only *owner_id*'s when given), newest first."""
    clause, args = _owner_clause(owner_id)
    rows = conn.execute(
        f"SELECT * FROM documents WHERE 1 = 1{clause} ORDER BY uploaded_at DESC", args
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


def rename_document(
    conn: sqlite3.Connection, doc_id: str, name: str, owner_id: str | None = None
) -> bool:
    """Set a document's user-visible name. Returns False if it does not exist."""
    clause, args = _owner_clause(owner_id)
    cur = conn.execute(
        f"UPDATE documents SET name = ? WHERE id = ?{clause}", (name, doc_id, *args)
    )
    conn.commit()
    return cur.rowcount > 0


def delete_document(
    conn: sqlite3.Connection, doc_id: str, owner_id: str | None = None
) -> bool:
    """Delete a document and all its passages (CASCADE).

    Returns True if a document was deleted, False if the document didn't exist
    (or is not owned by *owner_id*).
    """
    # DELETE on documents triggers CASCADE on passages (ON DELETE CASCADE)
    clause, args = _owner_clause(owner_id)
    cur = conn.execute(f"DELETE FROM documents WHERE id = ?{clause}", (doc_id, *args))
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
        owner_id=row["owner_id"] if "owner_id" in row.keys() else None,
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
