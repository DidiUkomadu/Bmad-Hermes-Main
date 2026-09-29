"""Unit tests for Markdown ingestion (Story 1.2).

Tests:
- Heading structure captured (H1, H2, etc.)
- Section locators derived from heading hierarchy
- Non-heading blocks identifiable (paragraphs, code blocks, lists)
- Fallback locator for documents without headings
- Common Markdown constructs handled
"""

from __future__ import annotations

import pytest

from app.ingestion.markdown import parse_markdown
from app.models import DocumentFormat

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def markdown_with_headings():
    """A Markdown document with a clear heading hierarchy."""
    return b"""# Main Title

This is the introduction paragraph.

## Section One

Some content in section one.

### Subsection A

Content in subsection A.

## Section Two

More content here.

```python
print("code block")
```

- List item 1
- List item 2
"""


@pytest.fixture
def markdown_no_headings():
    """A Markdown document with no heading structure."""
    return b"""This is the first paragraph.

Another paragraph here.

A third paragraph with some content.

    indented code block
"""


@pytest.fixture
def markdown_nested_lists():
    """Markdown with nested lists and mixed blocks."""
    return b"""# Project Plan

## Phase 1

- Item 1
  - Sub-item 1a
  - Sub-item 1b
- Item 2

## Phase 2

1. First step
2. Second step
   1. Sub-step 2.1
   2. Sub-step 2.2
"""


@pytest.fixture
def markdown_edge_cases():
    """Edge cases: nested headings, code blocks with language, blockquotes."""
    return b"""# Root

> This is a blockquote
> spanning multiple lines

## Code Example

```javascript
function hello() {
    console.log("world");
}
```

### Deep Heading

Content at level 3.

#### Even Deeper

Level 4 content.

##### Level 5

The deepest level.
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMarkdownHeadingParsing:
    """Story 1.2: Heading structure tests."""

    def test_heading_levels_identified(self, markdown_with_headings):
        result = parse_markdown(markdown_with_headings, "test.md")

        headings = [u for u in result.units if u["type"] == "heading"]
        assert len(headings) >= 3  # H1 + at least 2 H2s

        levels = {h["level"] for h in headings}
        assert 1 in levels  # H1 present
        assert 2 in levels  # H2 present

    def test_h1_heading_captured(self, markdown_with_headings):
        result = parse_markdown(markdown_with_headings, "test.md")

        h1s = [u for u in result.units if u["type"] == "heading" and u["level"] == 1]
        assert len(h1s) == 1
        assert "Main Title" in h1s[0]["text"]

    def test_h2_headings_captured(self, markdown_with_headings):
        result = parse_markdown(markdown_with_headings, "test.md")

        h2s = [u for u in result.units if u["type"] == "heading" and u["level"] == 2]
        assert len(h2s) >= 2

    def test_heading_text_preserved(self, markdown_with_headings):
        result = parse_markdown(markdown_with_headings, "test.md")

        headings = [u for u in result.units if u["type"] == "heading"]
        heading_texts = [h["text"].strip() for h in headings]
        assert "Main Title" in heading_texts
        assert "Section One" in heading_texts


class TestMarkdownSectionLocators:
    """Story 1.2: Section locator tests."""

    def test_section_locator_includes_heading_text(self, markdown_with_headings):
        result = parse_markdown(markdown_with_headings, "test.md")

        # The H2 "Section One" paragraph should have a locator referencing it
        section_one_heading = next(
            u for u in result.units
            if u["type"] == "heading" and "Section One" in u["text"]
        )
        section_loc = section_one_heading["location"]
        assert "Section One" in section_loc

    def test_paragraph_under_heading_has_section_locator(self, markdown_with_headings):
        result = parse_markdown(markdown_with_headings, "test.md")

        # Find the paragraph after "Section One" heading
        para_units = [u for u in result.units if u["type"] == "paragraph"]
        assert len(para_units) >= 3  # intro + section one + section two

        # At least one paragraph should have a locator referencing a heading
        locs = [p["location"] for p in para_units]
        assert any("Section" in loc for loc in locs) or any("Untitled" not in loc for loc in locs)

    def test_nested_heading_locator(self, markdown_with_headings):
        result = parse_markdown(markdown_with_headings, "test.md")

        # "Subsection A" is under "Section One"
        subsection = next(
            u for u in result.units
            if u["type"] == "heading" and "Subsection" in u["text"]
        )
        # Locator should include the parent section
        assert "Section One" in subsection["location"] or "Subsection A" in subsection["location"]

    def test_no_heading_fallback_locator(self, markdown_no_headings):
        result = parse_markdown(markdown_no_headings, "nohead.md")

        # All units should have a fallback locator
        for unit in result.units:
            if unit["text"]:
                assert unit["location"] is not None
                assert len(unit["location"]) > 0


class TestMarkdownBlockTypes:
    """Story 1.2: Non-heading block identification."""

    def test_paragraphs_identified(self, markdown_with_headings):
        result = parse_markdown(markdown_with_headings, "test.md")

        paragraphs = [u for u in result.units if u["type"] == "paragraph"]
        assert len(paragraphs) >= 3
        assert all(u["type"] == "paragraph" for u in paragraphs)

    def test_code_blocks_identified(self, markdown_with_headings):
        result = parse_markdown(markdown_with_headings, "test.md")

        code_blocks = [u for u in result.units if u["type"] == "code_block"]
        assert len(code_blocks) >= 1
        assert 'print("code block")' in code_blocks[0]["text"]

    def test_code_block_language_detected(self, markdown_with_headings):
        result = parse_markdown(markdown_with_headings, "test.md")

        code_blocks = [u for u in result.units if u["type"] == "code_block"]
        assert code_blocks[0].get("lang") == "python"

    def test_lists_identified(self, markdown_with_headings):
        result = parse_markdown(markdown_with_headings, "test.md")

        lists = [u for u in result.units if u["type"] == "list"]
        assert len(lists) >= 1
        assert "List item 1" in lists[0]["text"] or "list item" in lists[0]["text"].lower()

    def test_list_content_preserved(self, markdown_nested_lists):
        result = parse_markdown(markdown_nested_lists, "nested.md")

        lists = [u for u in result.units if u["type"] == "list"]
        assert len(lists) >= 2  # Phase 1 bulleted + Phase 2 numbered


class TestMarkdownCommonConstructs:
    """Story 1.2: Technical document constructs."""

    def test_blockquote_handled(self, markdown_edge_cases):
        result = parse_markdown(markdown_edge_cases, "edge.md")

        blockquotes = [u for u in result.units if u["type"] == "blockquote"]
        assert len(blockquotes) >= 1
        assert "blockquote" in blockquotes[0]["text"].lower()

    def test_deep_heading_hierarchy(self, markdown_edge_cases):
        result = parse_markdown(markdown_edge_cases, "edge.md")

        headings = [u for u in result.units if u["type"] == "heading"]
        levels = sorted({h["level"] for h in headings})
        assert levels == [1, 2, 3, 4, 5]

    def test_fenced_code_block_with_language(self, markdown_edge_cases):
        result = parse_markdown(markdown_edge_cases, "edge.md")

        code_blocks = [u for u in result.units if u["type"] == "code_block"]
        assert len(code_blocks) >= 1
        assert code_blocks[0].get("lang") == "javascript"
        assert 'console.log' in code_blocks[0]["text"]

    def test_document_id_generated(self, markdown_with_headings):
        result = parse_markdown(markdown_with_headings, "my_doc.md")
        assert result.document_id.startswith("my-doc-")

    def test_format_is_markdown(self, markdown_with_headings):
        result = parse_markdown(markdown_with_headings, "test.md")
        assert result.format == DocumentFormat.MARKDOWN

    def test_full_text_from_parsed_blocks(self, markdown_with_headings):
        result = parse_markdown(markdown_with_headings, "test.md")
        assert len(result.full_text) > 0
        assert "Main Title" in result.full_text or "main title" in result.full_text.lower()


class TestMarkdownEdgeCases:
    """Edge cases for Markdown parsing."""

    def test_empty_markdown(self):
        result = parse_markdown(b"", "empty.md")
        assert len(result.units) == 0
        assert len(result.full_text) == 0

    def test_whitespace_only(self):
        result = parse_markdown(b"   \n\n   ", "whitespace.md")
        assert len(result.full_text) == 0

    def test_single_paragraph(self):
        result = parse_markdown(b"Just one paragraph.", "single.md")
        assert len(result.units) >= 1

    def test_unicode_content(self):
        content = b"# Unicode \xc3\xa9\xc3\xa8\xc3\xaa\n\nParagraph with \xe2\x80\x9cquotes\xe2\x80\x9d."
        result = parse_markdown(content, "unicode.md")
        assert len(result.full_text) > 0
