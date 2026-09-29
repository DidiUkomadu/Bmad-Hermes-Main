"""Unit tests for content store (Stories 2.1, 2.2).

Tests:
- Document persistence (create, retrieve, list)
- Passage persistence (create, retrieve by ID, list by document)
- Cascade delete on document removal
- Referential integrity
"""

from __future__ import annotations

import pytest

from app.models import DocumentFormat, DocumentMeta, Passage
from app.store.schema import (
    create_document,
    create_passage,
    create_schema,
    delete_document,
    get_connection,
    get_document,
    get_passage,
    list_documents,
    list_passages_for_document,
    set_db_path,
)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test_store.db"
    set_db_path(str(path))
    conn = get_connection()
    create_schema(conn)
    return str(path)


@pytest.fixture
def sample_doc(db_path):
    from app.store.schema import get_connection
    conn = get_connection()
    doc = DocumentMeta(
        id="doc-001",
        name="test_document.pdf",
        format=DocumentFormat.PDF,
        uploaded_at=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ),
        page_count=5,
    )
    create_document(conn, doc)
    return doc


@pytest.fixture
def sample_passage(db_path, sample_doc):
    from app.store.schema import get_connection
    conn = get_connection()
    passage = Passage(
        id="doc-001:page:1:chunk:0",
        document_id=sample_doc.id,
        text="This is the first page of the document.",
        location="page:1",
        start_offset=0,
        end_offset=38,
    )
    create_passage(conn, passage)
    return passage


class TestDocumentPersistence:
    """Story 2.1: Document persistence."""

    def test_create_and_retrieve_document(self, db_path):
        conn = get_connection()
        doc = DocumentMeta(
            id="doc-test-001",
            name="test_doc.pdf",
            format=DocumentFormat.PDF,
            uploaded_at=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ),
            page_count=10,
        )
        create_document(conn, doc)

        retrieved = get_document(conn, "doc-test-001")
        assert retrieved is not None
        assert retrieved.id == "doc-test-001"
        assert retrieved.name == "test_doc.pdf"
        assert retrieved.format == DocumentFormat.PDF
        assert retrieved.page_count == 10

    def test_get_nonexistent_document_returns_none(self, db_path):
        conn = get_connection()
        result = get_document(conn, "nonexistent-id")
        assert result is None

    def test_list_documents(self, db_path):
        conn = get_connection()
        doc1 = DocumentMeta(
            id="doc-list-001",
            name="doc_a.pdf",
            format=DocumentFormat.PDF,
            uploaded_at=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ),
        )
        doc2 = DocumentMeta(
            id="doc-list-002",
            name="doc_b.md",
            format=DocumentFormat.MARKDOWN,
            uploaded_at=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ),
        )
        create_document(conn, doc1)
        create_document(conn, doc2)

        docs = list_documents(conn)
        assert len(docs) == 2
        doc_ids = {d.id for d in docs}
        assert "doc-list-001" in doc_ids
        assert "doc-list-002" in doc_ids

    def test_document_metadata_preserved(self, db_path):
        conn = get_connection()
        import datetime

        dt = datetime.datetime(2025, 1, 15, 10, 0, 0, tzinfo=datetime.UTC)
        doc = DocumentMeta(
            id="doc-meta-001",
            name="metadata_test.txt",
            format=DocumentFormat.TXT,
            uploaded_at=dt,
            section_count=5,
        )
        create_document(conn, doc)

        retrieved = get_document(conn, "doc-meta-001")
        assert retrieved.uploaded_at == dt
        assert retrieved.section_count == 5
        assert retrieved.format == DocumentFormat.TXT


class TestPassagePersistence:
    """Story 2.1: Passage persistence."""

    def test_create_and_retrieve_passage(self, db_path, sample_doc):
        conn = get_connection()
        passage = Passage(
            id="doc-001:page:2:chunk:0",
            document_id=sample_doc.id,
            text="Second page content here.",
            location="page:2",
            start_offset=38,
            end_offset=60,
        )
        create_passage(conn, passage)

        retrieved = get_passage(conn, "doc-001:page:2:chunk:0")
        assert retrieved is not None
        assert retrieved.id == "doc-001:page:2:chunk:0"
        assert retrieved.document_id == sample_doc.id
        assert retrieved.text == "Second page content here."
        assert retrieved.location == "page:2"
        assert retrieved.start_offset == 38
        assert retrieved.end_offset == 60

    def test_get_nonexistent_passage_returns_none(self, db_path):
        conn = get_connection()
        result = get_passage(conn, "nonexistent-passage-id")
        assert result is None

    def test_list_passages_for_document(self, db_path, sample_doc, sample_passage):
        conn = get_connection()
        # Add a second passage with a DIFFERENT ID than the fixture
        passage2 = Passage(
            id="doc-001:page:2:chunk:1",
            document_id=sample_doc.id,
            text="Second page content.",
            location="page:2",
            start_offset=38,
            end_offset=55,
        )
        create_passage(conn, passage2)

        passages = list_passages_for_document(conn, sample_doc.id)
        assert len(passages) == 2
        passage_ids = {p.id for p in passages}
        assert "doc-001:page:1:chunk:0" in passage_ids
        assert "doc-001:page:2:chunk:1" in passage_ids

    def test_passage_text_preserved_exactly(self, db_path, sample_doc):
        conn = get_connection()
        long_text = "This is a longer passage with enough text to verify that the full content is stored and retrievable without truncation or modification."
        passage = Passage(
            id="doc-001:page:3:chunk:0",
            document_id=sample_doc.id,
            text=long_text,
            location="page:3",
            start_offset=0,
            end_offset=len(long_text),
        )
        create_passage(conn, passage)

        retrieved = get_passage(conn, passage.id)
        assert retrieved is not None
        assert retrieved.text == long_text  # Exact match

    def test_multiple_passages_same_document(self, db_path):
        conn = get_connection()
        # Create a fresh document with 6 passages to avoid fixture interaction issues
        doc_id = "multi-doc-001"
        doc = DocumentMeta(
            id=doc_id,
            name="multi_test.pdf",
            format=DocumentFormat.PDF,
            uploaded_at=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ),
        )
        create_document(conn, doc)

        for i in range(6):
            passage = Passage(
                id=f"{doc_id}:page:{i+1}:chunk:0",
                document_id=doc_id,
                text=f"Page {i+1} content.",
                location=f"page:{i+1}",
                start_offset=i * 20,
                end_offset=(i + 1) * 20,
            )
            create_passage(conn, passage)

        passages = list_passages_for_document(conn, doc_id)
        assert len(passages) == 6
        assert all(p.document_id == doc_id for p in passages)
        assert passages[0].id == f"{doc_id}:page:1:chunk:0"
        assert passages[5].id == f"{doc_id}:page:6:chunk:0"


class TestDocumentRemoval:
    """Story 2.2: Document removal and cascade behavior."""

    def test_delete_document_cascades_to_passages(self, db_path, sample_doc, sample_passage):
        conn = get_connection()

        # Verify passages exist before deletion
        passages_before = list_passages_for_document(conn, sample_doc.id)
        assert len(passages_before) >= 1

        # Delete the document
        result = delete_document(conn, sample_doc.id)
        assert result is True

        # Passages should be gone
        passages_after = list_passages_for_document(conn, sample_doc.id)
        assert len(passages_after) == 0

        # Document should be gone
        doc = get_document(conn, sample_doc.id)
        assert doc is None

    def test_delete_nonexistent_document_returns_false(self, db_path):
        conn = get_connection()
        result = delete_document(conn, "nonexistent-id")
        assert result is False

    def test_delete_one_document_does_not_affect_others(self, db_path):
        conn = get_connection()
        # Create two documents
        doc1 = DocumentMeta(
            id="del-doc-001",
            name="doc1.pdf",
            format=DocumentFormat.PDF,
            uploaded_at=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ),
        )
        doc2 = DocumentMeta(
            id="del-doc-002",
            name="doc2.pdf",
            format=DocumentFormat.PDF,
            uploaded_at=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ),
        )
        create_document(conn, doc1)
        create_document(conn, doc2)

        # Add passages to both
        for doc_id in [doc1.id, doc2.id]:
            create_passage(conn, Passage(
                id=f"{doc_id}:chunk:0",
                document_id=doc_id,
                text="Test content",
                location="page:1",
                start_offset=0,
                end_offset=12,
            ))

        # Delete doc1
        delete_document(conn, doc1.id)

        # doc1 gone
        assert get_document(conn, doc1.id) is None
        assert list_passages_for_document(conn, doc1.id) == []

        # doc2 still exists
        assert get_document(conn, doc2.id) is not None
        assert len(list_passages_for_document(conn, doc2.id)) == 1

    def test_passage_lookup_fails_after_document_deletion(self, db_path, sample_doc, sample_passage):
        conn = get_connection()
        passage_id = sample_passage.id

        delete_document(conn, sample_doc.id)

        # Passage should no longer be retrievable
        passage = get_passage(conn, passage_id)
        assert passage is None
