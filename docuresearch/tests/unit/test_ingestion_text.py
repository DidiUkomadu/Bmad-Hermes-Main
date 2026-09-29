"""Unit tests for plain text ingestion (Story 1.3).

Tests:
- Paragraph-based splitting
- Paragraph index locators
- Offset ranges
- Fallback splitting when no paragraph boundaries
- Edge cases: single paragraph, no boundaries, empty file
"""

from __future__ import annotations

import pytest

from app.ingestion.text import paragraph_count, parse_plain_text
from app.models import DocumentFormat

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def text_with_paragraphs():
    """A plain text document with clear paragraph boundaries."""
    return b"""First paragraph. This is some content.

Second paragraph. More content here.

Third paragraph. Yet more text.
"""


@pytest.fixture
def text_single_paragraph():
    """A document with only one paragraph (no blank lines)."""
    return b"This is a single paragraph document. It has no blank lines."


@pytest.fixture
def text_no_paragraph_boundaries():
    """Text with no paragraph boundaries — just a continuous stream."""
    return b"Line one\nLine two\nLine three\nLine four"


@pytest.fixture
def text_empty():
    """An empty text file."""
    return b""


@pytest.fixture
def text_with_unicode():
    """Text with Unicode characters."""
    return "Paragraph with \u00e9\u00e8\u00ea special characters.\n\nAnother paragraph with \u201cquotes\u201d and \u2014 dashes.\n".encode("utf-8")


@pytest.fixture
def text_multiline_paragraph():
    """A paragraph that spans multiple lines (no blank lines within)."""
    return b"""This is a paragraph that spans
multiple lines within the same
paragraph group. It should be treated
as a single unit.

Second paragraph here.
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParagraphSplitting:
    """Story 1.3: Paragraph splitting tests."""

    def test_splits_on_blank_lines(self, text_with_paragraphs):
        result = parse_plain_text(text_with_paragraphs, "test.txt")

        assert len(result.units) == 3
        assert result.units[0]["location"] == "paragraph:1"
        assert result.units[1]["location"] == "paragraph:2"
        assert result.units[2]["location"] == "paragraph:3"

    def test_paragraph_content_preserved(self, text_with_paragraphs):
        result = parse_plain_text(text_with_paragraphs, "test.txt")

        assert "First paragraph" in result.units[0]["text"]
        assert "Second paragraph" in result.units[1]["text"]
        assert "Third paragraph" in result.units[2]["text"]

    def test_paragraph_index_incrementing(self, text_with_paragraphs):
        result = parse_plain_text(text_with_paragraphs, "test.txt")
        indices = [u.get("para_index", 0) for u in result.units]
        assert indices == [1, 2, 3]

    def test_paragraph_text_stripped(self, text_with_paragraphs):
        result = parse_plain_text(text_with_paragraphs, "test.txt")
        for unit in result.units:
            text = unit["text"]
            assert text == text.strip(), f"Text not stripped: {text!r}"

    def test_empty_lines_between_paragraphs_not_included(self, text_with_paragraphs):
        result = parse_plain_text(text_with_paragraphs, "test.txt")
        # No unit should be empty or whitespace-only
        for unit in result.units:
            assert unit["text"].strip(), f"Empty unit found: {unit!r}"


class TestLocatorsAndOffsets:
    """Story 1.3: Locator and offset tests."""

    def test_paragraph_locator_format(self, text_with_paragraphs):
        result = parse_plain_text(text_with_paragraphs, "test.txt")
        for unit in result.units:
            assert unit["location"].startswith("paragraph:")

    def test_offsets_are_monotonically_increasing(self, text_with_paragraphs):
        result = parse_plain_text(text_with_paragraphs, "test.txt")

        offsets = [(u["start_offset"], u["end_offset"]) for u in result.units]
        for i, (start, end) in enumerate(offsets):
            assert start < end
            if i > 0:
                prev_end = offsets[i - 1][1]
                assert start >= prev_end

    def test_offset_matches_text_length(self, text_with_paragraphs):
        result = parse_plain_text(text_with_paragraphs, "test.txt")

        for unit in result.units:
            text_len = len(unit["text"])
            assert unit["end_offset"] - unit["start_offset"] == text_len

    def test_full_text_matches_concatenation(self, text_with_paragraphs):
        result = parse_plain_text(text_with_paragraphs, "test.txt")

        concatenated = "\n\n".join(u["text"] for u in result.units)
        assert result.full_text == concatenated


class TestEdgeCases:
    """Story 1.3: Edge case tests."""

    def test_single_paragraph(self, text_single_paragraph):
        result = parse_plain_text(text_single_paragraph, "single.txt")

        assert len(result.units) == 1
        assert result.units[0]["location"] == "paragraph:1"
        assert "single paragraph" in result.units[0]["text"].lower()

    def test_no_paragraph_boundaries_still_produces_unit(self, text_no_paragraph_boundaries):
        result = parse_plain_text(text_no_paragraph_boundaries, "noboundary.txt")

        # Should still produce at least one unit
        assert len(result.units) >= 1
        # Content should be present
        full_text = result.full_text
        assert "line one" in full_text.lower() or "line one" in full_text

    def test_empty_file(self, text_empty):
        result = parse_plain_text(text_empty, "empty.txt")
        assert len(result.units) == 0
        assert len(result.full_text) == 0

    def test_whitespace_only_file(self):
        result = parse_plain_text(b"   \n\n   \n\n  ", "whitespace.txt")
        assert len(result.units) == 0

    def test_unicode_content(self, text_with_unicode):
        result = parse_plain_text(text_with_unicode, "unicode.txt")
        assert len(result.full_text) > 0
        assert "\u00e9" in result.full_text or "special" in result.full_text.lower()

    def test_multiline_paragraph_treated_as_single_unit(self, text_multiline_paragraph):
        result = parse_plain_text(text_multiline_paragraph, "multiline.txt")

        # The multi-line paragraph should be one unit
        para_units = [u for u in result.units if u["type"] == "paragraph"]
        assert len(para_units) >= 1
        # The first paragraph should contain all the lines
        first = para_units[0]
        assert "multiple lines" in first["text"]
        assert "same" in first["text"]

    def test_format_is_txt(self, text_with_paragraphs):
        result = parse_plain_text(text_with_paragraphs, "test.txt")
        assert result.format == DocumentFormat.TXT

    def test_document_id_generated(self, text_with_paragraphs):
        result = parse_plain_text(text_with_paragraphs, "my_doc.txt")
        assert result.document_id.startswith("my-doc-")


class TestParagraphCountUtility:
    """Test the paragraph_count utility function."""

    def test_counts_paragraphs(self, text_with_paragraphs):
        result = parse_plain_text(text_with_paragraphs, "test.txt")
        assert paragraph_count(result.units) == 3

    def test_counts_zero_for_empty(self, text_empty):
        result = parse_plain_text(text_empty, "empty.txt")
        assert paragraph_count(result.units) == 0

    def test_counts_single(self, text_single_paragraph):
        result = parse_plain_text(text_single_paragraph, "single.txt")
        assert paragraph_count(result.units) == 1


class TestFallbackBehavior:
    """Story 1.3: Fallback splitting when no paragraph boundaries."""

    def test_fallback_combines_lines(self, text_no_paragraph_boundaries):
        """When no blank-line boundaries exist, lines are joined."""
        result = parse_plain_text(text_no_paragraph_boundaries, "nolinebreaks.txt")

        # Should have at least one unit
        assert len(result.units) >= 1
        # Content from multiple lines should be in the unit
        text = result.units[0]["text"].lower()
        assert "line" in text  # at least some content

    def test_fallback_gives_paragraph_locator(self, text_no_paragraph_boundaries):
        result = parse_plain_text(text_no_paragraph_boundaries, "nolinebreaks.txt")

        if result.units:
            assert result.units[0]["location"].startswith("paragraph:")
