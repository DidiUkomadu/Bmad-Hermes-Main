"""Tests for ingestion orchestration (Story 1.5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.chunking.strategy import ChunkingConfig
from app.ingestion.orchestrator import (
    SUPPORTED_EXTENSIONS,
    detect_format,
    ingest_document,
)
from app.ingestion.text import parse_plain_text
from app.models import DocumentFormat
from app.store.schema import (
    delete_document,
    get_connection,
    list_documents,
    list_passages_for_document,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pdf_path() -> Path:
    """Path to the sample PDF document."""
    return Path(__file__).resolve().parent.parent.parent / "evaluation" / "dataset" / "documents" / "sample_spec.pdf"


@pytest.fixture
def sample_md_path() -> Path:
    """Path to the sample Markdown document."""
    return Path(__file__).resolve().parent.parent.parent / "evaluation" / "dataset" / "documents" / "sample_spec.md"


@pytest.fixture
def sample_txt_path() -> Path:
    """Path to the sample text document."""
    return Path(__file__).resolve().parent.parent.parent / "evaluation" / "dataset" / "documents" / "sample_document.txt"


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite database path."""
    return tmp_path / "test_ingestion.db"


# ---------------------------------------------------------------------------
# Format detection tests
# ---------------------------------------------------------------------------


class TestFormatDetection:
    """Verify format detection from file paths."""

    def test_detect_pdf_by_extension(self):
        fmt = detect_format("document.pdf")
        assert fmt == DocumentFormat.PDF

    def test_detect_pdf_by_path_object(self, sample_pdf_path):
        fmt = detect_format(sample_pdf_path)
        assert fmt == DocumentFormat.PDF

    def test_detect_markdown_by_extension(self):
        fmt = detect_format("document.md")
        assert fmt == DocumentFormat.MARKDOWN

    def test_detect_markdown_alternate_extension(self):
        fmt = detect_format("document.markdown")
        assert fmt == DocumentFormat.MARKDOWN

    def test_detect_txt_by_extension(self):
        fmt = detect_format("document.txt")
        assert fmt == DocumentFormat.TXT

    def test_detect_case_insensitive(self):
        fmt = detect_format("document.PDF")
        assert fmt == DocumentFormat.PDF

        fmt = detect_format("document.MD")
        assert fmt == DocumentFormat.MARKDOWN

        fmt = detect_format("document.TXT")
        assert fmt == DocumentFormat.TXT

    def test_detect_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported document format"):
            detect_format("document.xyz")

    def test_detect_docx_raises(self):
        with pytest.raises(ValueError, match="Unsupported document format"):
            detect_format("document.docx")

    def test_supported_extensions_list(self):
        assert DocumentFormat.PDF in SUPPORTED_EXTENSIONS.values()
        assert DocumentFormat.MARKDOWN in SUPPORTED_EXTENSIONS.values()
        assert DocumentFormat.TXT in SUPPORTED_EXTENSIONS.values()
        # Both .md and .markdown map to MARKDOWN
        assert len(SUPPORTED_EXTENSIONS) == 4
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".markdown" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# PDF orchestration tests
# ---------------------------------------------------------------------------


class TestPdfOrchestration:
    """End-to-end PDF ingestion through the orchestrator."""

    def test_pdf_orchestration_creates_document(self, sample_pdf_path, temp_db):
        result = ingest_document(str(sample_pdf_path), db_path=str(temp_db))
        assert result["format"] == "pdf"
        assert result["document_name"] == sample_pdf_path.name
        assert len(result["document_id"]) > 0
        assert result["passage_count"] > 0

    def test_pdf_orchestration_creates_passages(self, sample_pdf_path, temp_db):
        result = ingest_document(str(sample_pdf_path), db_path=str(temp_db))
        conn = get_connection()
        stored = conn.execute(
            "SELECT COUNT(*) FROM passages WHERE document_id = ?",
            (result["document_id"],),
        ).fetchone()[0]
        assert stored == result["passage_count"]

    def test_pdf_orchestration_passage_ids_match(self, sample_pdf_path, temp_db):
        result = ingest_document(str(sample_pdf_path), db_path=str(temp_db))
        conn = get_connection()
        passage_ids = [
            row[0] for row in conn.execute(
                "SELECT id FROM passages WHERE document_id = ? ORDER BY id",
                (result["document_id"],),
            ).fetchall()
        ]
        assert passage_ids == result["passage_ids"]

    def test_pdf_orchestration_passage_text_preserved(self, sample_pdf_path, temp_db):
        result = ingest_document(str(sample_pdf_path), db_path=str(temp_db))
        conn = get_connection()
        rows = conn.execute(
            "SELECT text FROM passages WHERE document_id = ?",
            (result["document_id"],),
        ).fetchall()
        full_text = "".join(row[0] for row in rows)
        assert len(full_text) > 0
        # Should contain content from the PDF
        assert "specification" in full_text.lower() or "version" in full_text.lower()

    def test_pdf_orchestration_passage_location_format(self, sample_pdf_path, temp_db):
        result = ingest_document(str(sample_pdf_path), db_path=str(temp_db))
        conn = get_connection()
        rows = conn.execute(
            "SELECT location FROM passages WHERE document_id = ?",
            (result["document_id"],),
        ).fetchall()
        for (loc,) in rows:
            assert loc.startswith("page:"), f"Expected page-based location, got: {loc}"
            # Verify it contains :chunk:N suffix from chunker
            assert ":chunk:" in loc, f"Expected chunk suffix in location, got: {loc}"


# ---------------------------------------------------------------------------
# Markdown orchestration tests
# ---------------------------------------------------------------------------


class TestMarkdownOrchestration:
    """End-to-end Markdown ingestion through the orchestrator."""

    def test_markdown_orchestration_creates_document(self, sample_md_path, temp_db):
        result = ingest_document(str(sample_md_path), db_path=str(temp_db))
        assert result["format"] == "markdown"
        assert result["document_name"] == sample_md_path.name
        assert len(result["document_id"]) > 0
        assert result["passage_count"] > 0

    def test_markdown_orchestration_creates_passages(self, sample_md_path, temp_db):
        result = ingest_document(str(sample_md_path), db_path=str(temp_db))
        conn = get_connection()
        stored = conn.execute(
            "SELECT COUNT(*) FROM passages WHERE document_id = ?",
            (result["document_id"],),
        ).fetchone()[0]
        assert stored == result["passage_count"]

    def test_markdown_orchestration_heading_locations(self, sample_md_path, temp_db):
        result = ingest_document(str(sample_md_path), db_path=str(temp_db))
        conn = get_connection()
        rows = conn.execute(
            "SELECT location FROM passages WHERE document_id = ?",
            (result["document_id"],),
        ).fetchall()
        locations = {row[0] for row in rows}
        # Should have section-based locations (section path from headings)
        # Format: "<heading text>/<sub-heading>[:chunk:N]"
        has_section = any("/" in loc for loc in locations)
        assert has_section, f"Expected section-based locations, got: {locations}"

    def test_markdown_orchestration_document_retrievable(self, sample_md_path, temp_db):
        result = ingest_document(str(sample_md_path), db_path=str(temp_db))
        conn = get_connection()
        doc = conn.execute(
            "SELECT name, format FROM documents WHERE id = ?",
            (result["document_id"],),
        ).fetchone()
        assert doc is not None
        assert doc[0] == sample_md_path.name


# ---------------------------------------------------------------------------
# TXT orchestration tests
# ---------------------------------------------------------------------------


class TestTxtOrchestration:
    """End-to-end TXT ingestion through the orchestrator."""

    def test_txt_orchestration_creates_document(self, sample_txt_path, temp_db):
        result = ingest_document(str(sample_txt_path), db_path=str(temp_db))
        assert result["format"] == "txt"
        assert result["document_name"] == sample_txt_path.name
        assert len(result["document_id"]) > 0
        assert result["passage_count"] > 0

    def test_txt_orchestration_creates_passages(self, sample_txt_path, temp_db):
        result = ingest_document(str(sample_txt_path), db_path=str(temp_db))
        conn = get_connection()
        stored = conn.execute(
            "SELECT COUNT(*) FROM passages WHERE document_id = ?",
            (result["document_id"],),
        ).fetchone()[0]
        assert stored == result["passage_count"]

    def test_txt_orchestration_paragraph_locations(self, sample_txt_path, temp_db):
        result = ingest_document(str(sample_txt_path), db_path=str(temp_db))
        conn = get_connection()
        rows = conn.execute(
            "SELECT location FROM passages WHERE document_id = ?",
            (result["document_id"],),
        ).fetchall()
        for (loc,) in rows:
            assert loc.startswith("paragraph:"), f"Expected paragraph-based location, got: {loc}"
            assert ":chunk:" in loc, f"Expected chunk suffix in location, got: {loc}"

    def test_txt_orchestration_full_text_match(self, sample_txt_path, temp_db):
        """Verify that concatenated passage text matches the source text
        (concatenation of unit texts, which is what the chunker processes)."""
        result = ingest_document(str(sample_txt_path), db_path=str(temp_db))
        conn = get_connection()

        # Get stored passages
        rows = conn.execute(
            "SELECT text FROM passages WHERE document_id = ? ORDER BY id",
            (result["document_id"],),
        ).fetchall()
        stored_text = "".join(row[0] for row in rows)

        # Get original units and their concatenated text
        with open(sample_txt_path, "rb") as f:
            file_bytes = f.read()
        raw = parse_plain_text(file_bytes, sample_txt_path.name)
        unit_text_concat = "".join(u["text"] for u in raw.units)

        assert stored_text == unit_text_concat, (
            f"Stored text ({len(stored_text)} chars) != unit text concatenation "
            f"({len(unit_text_concat)} chars). "
            f"full_text includes paragraph separators ({len(raw.full_text)} chars)."
        )


# ---------------------------------------------------------------------------
# End-to-end tests: multiple documents
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end multi-document ingestion scenarios."""

    def test_all_three_formats_in_same_db(self, sample_pdf_path, sample_md_path, sample_txt_path, temp_db):
        """Ingest all three formats into the same database and verify."""
        pdf_result = ingest_document(str(sample_pdf_path), db_path=str(temp_db))
        md_result = ingest_document(str(sample_md_path), db_path=str(temp_db))
        txt_result = ingest_document(str(sample_txt_path), db_path=str(temp_db))

        conn = get_connection()

        # Verify all three documents exist
        docs = conn.execute("SELECT id, name, format FROM documents ORDER BY name").fetchall()
        assert len(docs) == 3

        doc_names = {row[1] for row in docs}
        assert sample_pdf_path.name in doc_names
        assert sample_md_path.name in doc_names
        assert sample_txt_path.name in doc_names

        # Verify passage counts
        total_passages = conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        expected_total = (
            pdf_result["passage_count"]
            + md_result["passage_count"]
            + txt_result["passage_count"]
        )
        assert total_passages == expected_total

    def test_documents_are_listable(self, sample_pdf_path, sample_md_path, sample_txt_path, temp_db):
        ingest_document(str(sample_pdf_path), db_path=str(temp_db))
        ingest_document(str(sample_md_path), db_path=str(temp_db))
        ingest_document(str(sample_txt_path), db_path=str(temp_db))

        conn = get_connection()
        docs = list_documents(conn)
        assert len(docs) == 3
        doc_ids = {d.id for d in docs}
        assert len(doc_ids) == 3

    def test_passages_are_retrievable(self, sample_pdf_path, temp_db):
        result = ingest_document(str(sample_pdf_path), db_path=str(temp_db))
        conn = get_connection()
        passages = list_passages_for_document(conn, result["document_id"])
        assert len(passages) == result["passage_count"]
        for p in passages:
            assert p.document_id == result["document_id"]

    def test_document_removal_cascades(self, sample_pdf_path, temp_db):
        result = ingest_document(str(sample_pdf_path), db_path=str(temp_db))
        doc_id = result["document_id"]
        passage_count = result["passage_count"]

        # Verify passages exist
        conn = get_connection()
        stored = conn.execute(
            "SELECT COUNT(*) FROM passages WHERE document_id = ?",
            (doc_id,),
        ).fetchone()[0]
        assert stored == passage_count

        # Delete document
        deleted = delete_document(conn, doc_id)
        assert deleted is True

        # Passages should be gone
        stored_after = conn.execute(
            "SELECT COUNT(*) FROM passages WHERE document_id = ?",
            (doc_id,),
        ).fetchone()[0]
        assert stored_after == 0

        # Document should be gone
        docs = list_documents(conn)
        assert doc_id not in {d.id for d in docs}


# ---------------------------------------------------------------------------
# Unsupported format tests
# ---------------------------------------------------------------------------


class TestUnsupportedFormat:
    """Verify unsupported formats are rejected."""

    def test_unsupported_extension_rejected(self, temp_db, tmp_path):
        # Create a file with an unsupported extension
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("some content")
        with pytest.raises(ValueError, match="Unsupported document format"):
            ingest_document(str(bad_file), db_path=str(temp_db))

    def test_nonexistent_file_rejected(self, temp_db):
        with pytest.raises(FileNotFoundError):
            ingest_document("/nonexistent/path.pdf", db_path=str(temp_db))

    def test_directory_rejected(self, temp_db, tmp_path):
        dir_path = tmp_path / "a_directory"
        dir_path.mkdir()
        with pytest.raises(ValueError, match="not a file"):
            ingest_document(str(dir_path), db_path=str(temp_db))


# ---------------------------------------------------------------------------
# Chunker invocation tests
# ---------------------------------------------------------------------------


class TestChunkerInvocation:
    """Verify the orchestrator correctly invokes the chunker."""

    def test_custom_chunking_config_respected(self, sample_txt_path, temp_db):
        """With a very small target_length, many passages should be created."""
        small_config = ChunkingConfig(target_length=50, overlap=10)
        result = ingest_document(
            str(sample_txt_path),
            db_path=str(temp_db),
            chunking_config=small_config,
        )
        # Small chunks -> many passages
        assert result["passage_count"] > 5

    def test_default_config_produces_sensible_chunks(self, sample_txt_path, temp_db):
        """Default config should produce a moderate number of passages."""
        result = ingest_document(str(sample_txt_path), db_path=str(temp_db))
        # sample_document.txt has 9 paragraphs; with default config
        # (target_length=450) each paragraph becomes its own chunk
        assert result["passage_count"] == 9, \
            f"Expected 9 passages (one per paragraph), got {result['passage_count']}"

    def test_passage_offsets_present(self, sample_txt_path, temp_db):
        result = ingest_document(str(sample_txt_path), db_path=str(temp_db))
        conn = get_connection()
        rows = conn.execute(
            "SELECT start_offset, end_offset FROM passages WHERE document_id = ? ORDER BY id",
            (result["document_id"],),
        ).fetchall()
        for start, end in rows:
            assert isinstance(start, int) and start >= 0
            assert isinstance(end, int) and end >= start


# ---------------------------------------------------------------------------
# Location and offset preservation tests
# ---------------------------------------------------------------------------


class TestLocationOffsetPreservation:
    """Verify locations and offsets are preserved through orchestration."""

    def test_pdf_location_and_offsets(self, sample_pdf_path, temp_db):
        result = ingest_document(str(sample_pdf_path), db_path=str(temp_db))
        conn = get_connection()
        rows = conn.execute(
            "SELECT id, location, start_offset, end_offset, text FROM passages WHERE document_id = ? ORDER BY id",
            (result["document_id"],),
        ).fetchall()
        for pid, loc, start, end, text in rows:
            assert loc.startswith("page:"), f"Expected page-based location, got: {loc}"
            assert ":chunk:" in loc
            assert end >= start
            assert len(text) > 0

    def test_md_location_and_offsets(self, sample_md_path, temp_db):
        result = ingest_document(str(sample_md_path), db_path=str(temp_db))
        conn = get_connection()
        rows = conn.execute(
            "SELECT id, location, start_offset, end_offset, text FROM passages WHERE document_id = ? ORDER BY id",
            (result["document_id"],),
        ).fetchall()
        for pid, loc, start, end, text in rows:
            assert len(loc) > 0
            assert end >= start
            assert len(text) > 0

    def test_txt_location_and_offsets(self, sample_txt_path, temp_db):
        result = ingest_document(str(sample_txt_path), db_path=str(temp_db))
        conn = get_connection()
        rows = conn.execute(
            "SELECT id, location, start_offset, end_offset, text FROM passages WHERE document_id = ? ORDER BY id",
            (result["document_id"],),
        ).fetchall()
        for pid, loc, start, end, text in rows:
            assert loc.startswith("paragraph:"), f"Expected paragraph-based location, got: {loc}"
            assert ":chunk:" in loc
            assert end >= start
            assert len(text) > 0


# ---------------------------------------------------------------------------
# Result structure tests
# ---------------------------------------------------------------------------


class TestResultStructure:
    """Verify the orchestrator returns the expected result structure."""

    def test_result_has_required_fields(self, sample_txt_path, temp_db):
        result = ingest_document(str(sample_txt_path), db_path=str(temp_db))
        assert "document_id" in result
        assert "document_name" in result
        assert "format" in result
        assert "passage_count" in result
        assert "passage_ids" in result
        assert "db_path" in result

    def test_result_passage_ids_are_strings(self, sample_txt_path, temp_db):
        result = ingest_document(str(sample_txt_path), db_path=str(temp_db))
        assert isinstance(result["passage_ids"], list)
        assert len(result["passage_ids"]) == result["passage_count"]
        for pid in result["passage_ids"]:
            assert isinstance(pid, str)
            assert len(pid) > 0

    def test_result_passage_ids_unique(self, sample_txt_path, temp_db):
        result = ingest_document(str(sample_txt_path), db_path=str(temp_db))
        assert len(result["passage_ids"]) == len(set(result["passage_ids"])), \
            "Duplicate passage IDs found"
