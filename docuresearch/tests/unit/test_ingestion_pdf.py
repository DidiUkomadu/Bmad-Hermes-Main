"""Unit tests for PDF ingestion (Story 1.1).

Tests:
- Text extraction from a text-based PDF
- Page boundaries preserved
- Page count derivable
- Corruption handling
- Non-PDF file rejection
"""

from __future__ import annotations

import io

import pytest

from app.ingestion.pdf import (
    extract_text_from_pdf,
    get_page_count,
)
from app.models import DocumentFormat

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pdf_bytes():
    """Create a simple 2-page text-based PDF for testing using fpdf2."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, "Document Title", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.cell(0, 8, "This is page one content.", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "It has multiple lines.", new_x="LMARGIN", new_y="NEXT")

    pdf.add_page()
    pdf.cell(0, 10, "This is page two.", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Different content here.", new_x="LMARGIN", new_y="NEXT")

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


@pytest.fixture
def empty_pdf_bytes():
    """Create a PDF with no text content."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


@pytest.fixture
def corrupt_pdf_bytes():
    """Return bytes that are not a valid PDF."""
    return b"This is not a PDF file at all. Just random garbage."


@pytest.fixture
def multi_page_pdf_bytes():
    """Create a 5-page PDF."""
    from fpdf import FPDF

    pdf = FPDF()
    for i in range(1, 6):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 10, f"This is page {i}.", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Content for page {i} goes here.", new_x="LMARGIN", new_y="NEXT")
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPdfExtraction:
    """Story 1.1 acceptance criteria tests."""

    def test_extracts_text_from_valid_pdf(self, sample_pdf_bytes):
        result = extract_text_from_pdf(sample_pdf_bytes, "test.pdf")

        assert result.format == DocumentFormat.PDF
        assert result.name == "test.pdf"
        assert len(result.full_text) > 0
        assert "Document Title" in result.full_text
        assert "page one" in result.full_text
        assert "page two" in result.full_text

    def test_preserves_page_boundaries(self, sample_pdf_bytes):
        result = extract_text_from_pdf(sample_pdf_bytes, "test.pdf")

        assert len(result.units) == 2  # 2 pages

        # Page 1
        page1 = result.units[0]
        assert page1["location"] == "page:1"
        assert "Document Title" in page1["text"]
        assert "page one" in page1["text"]

        # Page 2
        page2 = result.units[1]
        assert page2["location"] == "page:2"
        assert "page two" in page2["text"]

    def test_page_count_derivable(self, multi_page_pdf_bytes):
        result = extract_text_from_pdf(multi_page_pdf_bytes, "multi.pdf")
        assert len(result.units) == 5

        # Also test get_page_count helper
        count = get_page_count(multi_page_pdf_bytes)
        assert count == 5

    def test_page_numbers_start_at_one(self, sample_pdf_bytes):
        result = extract_text_from_pdf(sample_pdf_bytes, "test.pdf")
        page_nums = [u.get("page_number", 0) for u in result.units]
        assert page_nums == [1, 2]

    def test_start_and_end_offsets(self, sample_pdf_bytes):
        result = extract_text_from_pdf(sample_pdf_bytes, "test.pdf")

        # Offsets should be monotonically increasing
        offsets = [(u["start_offset"], u["end_offset"]) for u in result.units if u["text"]]
        for i, (start, end) in enumerate(offsets):
            assert start < end
            if i > 0:
                prev_end = offsets[i - 1][1]
                assert start >= prev_end

    def test_full_text_combines_pages(self, sample_pdf_bytes):
        result = extract_text_from_pdf(sample_pdf_bytes, "test.pdf")
        # Full text should contain content from all pages
        assert "Document Title" in result.full_text
        assert "This is page two" in result.full_text
        # Pages should be separated
        assert "\n\n" in result.full_text

    def test_document_id_is_generated(self, sample_pdf_bytes):
        result = extract_text_from_pdf(sample_pdf_bytes, "my_document.pdf")
        assert result.document_id.startswith("my-document-")
        assert len(result.document_id) > 10

    def test_format_is_pdf(self, sample_pdf_bytes):
        result = extract_text_from_pdf(sample_pdf_bytes, "test.pdf")
        assert result.format == DocumentFormat.PDF

    def test_empty_pdf_returns_empty_text(self, empty_pdf_bytes):
        result = extract_text_from_pdf(empty_pdf_bytes, "empty.pdf")
        assert len(result.full_text) == 0
        # Still has page records
        assert len(result.units) == 1
        assert result.units[0]["location"] == "page:1"

    def test_corrupt_pdf_raises_error(self, corrupt_pdf_bytes):
        with pytest.raises((ValueError, Exception)):
            extract_text_from_pdf(corrupt_pdf_bytes, "corrupt.pdf")

    def test_invalid_bytes_raise_error(self):
        with pytest.raises((ValueError, Exception)):
            extract_text_from_pdf(b"not a pdf", "bad.pdf")

    def test_upload_timestamp_set(self, sample_pdf_bytes):
        import datetime

        before = datetime.datetime.now(datetime.UTC)
        result = extract_text_from_pdf(sample_pdf_bytes, "test.pdf")
        after = datetime.datetime.now(datetime.UTC)

        assert before <= result.uploaded_at <= after

    def test_page_text_stripped(self, sample_pdf_bytes):
        result = extract_text_from_pdf(sample_pdf_bytes, "test.pdf")
        for unit in result.units:
            text = unit["text"]
            if text:
                assert text == text.strip()


class TestPdfEdgeCases:
    """Additional edge case tests for PDF extraction."""

    def test_large_page_count_warns(self, caplog, tmp_path):
        import logging

        from fpdf import FPDF

        from app.ingestion.pdf import MAX_PAGES_WARNING

        # Create a PDF with more pages than the warning threshold
        pdf = FPDF()
        for i in range(MAX_PAGES_WARNING + 5):
            pdf.add_page()
            pdf.set_font("Helvetica", size=12)
            pdf.cell(0, 8, f"Page {i+1}", new_x="LMARGIN", new_y="NEXT")

        buf = io.BytesIO()
        pdf.output(buf)
        pdf_bytes = buf.getvalue()

        caplog.set_level(logging.WARNING)
        result = extract_text_from_pdf(pdf_bytes, "large.pdf")

        assert len(result.units) == MAX_PAGES_WARNING + 5
        assert "exceeding warning threshold" in caplog.text

    def test_single_page_pdf(self, tmp_path):
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 10, "Single page document.", new_x="LMARGIN", new_y="NEXT")
        buf = io.BytesIO()
        pdf.output(buf)

        result = extract_text_from_pdf(buf.getvalue(), "single.pdf")
        assert len(result.units) == 1
        assert result.units[0]["location"] == "page:1"

    def test_pdf_with_special_characters(self, tmp_path):
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 10, "Special chars: !@#$%^&*()", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, "Unicode: \u00e9\u00e8\u00ea", new_x="LMARGIN", new_y="NEXT")
        buf = io.BytesIO()
        pdf.output(buf)

        result = extract_text_from_pdf(buf.getvalue(), "special.pdf")
        assert len(result.full_text) > 0
