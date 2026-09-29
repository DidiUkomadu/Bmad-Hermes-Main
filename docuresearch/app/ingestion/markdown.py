"""Markdown parsing with heading-based sectioning (Story 1.2).

Uses mistune v3 to parse Markdown into structured blocks (headings, paragraphs,
code blocks, lists). Heading hierarchy is captured for section-level locators.

API:
    parse_markdown(file_bytes: bytes, filename: str) -> RawDocument
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import mistune

from app.models import DocumentFormat, RawDocument, generate_document_id

logger = logging.getLogger(__name__)


def parse_markdown(file_bytes: bytes, filename: str) -> RawDocument:
    """Parse a Markdown file into structured blocks with heading-based locators.

    Uses mistune v3's block AST (renderer=None returns a list of block nodes).
    Heading structure is tracked to produce section locators for each block.

    Args:
        file_bytes: The raw bytes of the Markdown file.
        filename: Original filename, used for the document name.

    Returns:
        A RawDocument with parsed blocks, heading hierarchy, and section locators.
    """
    document_id = generate_document_id(filename)
    name = filename

    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")

    # mistune v3: renderer=None returns the block AST as a list of nodes
    md = mistune.create_markdown(renderer=None)
    ast = md(text)

    units = _ast_to_units(ast)

    # Build full_text from non-empty unit texts
    full_text = "\n\n".join(u["text"] for u in units if u["text"])

    return RawDocument(
        document_id=document_id,
        name=name,
        format=DocumentFormat.MARKDOWN,
        uploaded_at=_utc_now(),
        full_text=full_text,
        units=units,
    )


def _utc_now():
    return datetime.now(UTC)


def _ast_to_units(ast: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert mistune v3 block AST into a list of unit dicts.

    Each unit has: text, type, location, start_offset, end_offset, plus
    format-specific metadata (level for headings, lang for code blocks, etc.).
    """
    units: list[dict[str, Any]] = []
    heading_stack: list[tuple[int, str]] = []
    offset = 0

    for node in ast:
        node_type = node.get("type", "")

        # Skip blank_line nodes — they're structural, not content
        if node_type == "blank_line":
            continue

        text = _extract_text_from_node(node)
        section_loc = _current_section(heading_stack) if heading_stack else "Untitled"

        if node_type == "heading":
            level = node.get("attrs", {}).get("level", 1)
            heading_text = _extract_text_from_node(node).strip()
            # Update heading stack
            heading_stack = [(lvl, h) for lvl, h in heading_stack if lvl < level]
            heading_stack.append((level, heading_text))
            section_loc = _current_section(heading_stack)

            units.append({
                "type": "heading",
                "text": text,
                "location": section_loc,
                "level": level,
                "start_offset": offset,
                "end_offset": offset + len(text),
            })
            offset += len(text) + 2  # account for the \n\n separator

        elif node_type == "paragraph":
            units.append({
                "type": "paragraph",
                "text": text,
                "location": section_loc,
                "start_offset": offset,
                "end_offset": offset + len(text),
            })
            offset += len(text) + 2

        elif node_type == "block_code":
            lang = node.get("attrs", {}).get("info", "")
            units.append({
                "type": "code_block",
                "text": text,
                "location": section_loc,
                "lang": lang,
                "start_offset": offset,
                "end_offset": offset + len(text),
            })
            offset += len(text) + 2

        elif node_type == "list":
            units.append({
                "type": "list",
                "text": text,
                "location": section_loc,
                "start_offset": offset,
                "end_offset": offset + len(text),
            })
            offset += len(text) + 2

        elif node_type == "block_quote":
            units.append({
                "type": "blockquote",
                "text": text,
                "location": section_loc,
                "start_offset": offset,
                "end_offset": offset + len(text),
            })
            offset += len(text) + 2

        elif node_type in ("emphasis", "strong", "link", "code", "image"):
            # Inline elements shouldn't appear at block level normally,
            # but handle gracefully
            units.append({
                "type": "inline",
                "text": text,
                "location": section_loc,
                "start_offset": offset,
                "end_offset": offset + len(text),
            })
            offset += len(text) + 2

        else:
            # Any other block type — treat as generic block
            units.append({
                "type": f"block_{node_type}",
                "text": text,
                "location": section_loc,
                "start_offset": offset,
                "end_offset": offset + len(text),
            })
            offset += len(text) + 2

    return units


def _extract_text_from_node(node: dict[str, Any]) -> str:
    """Extract plain text from a mistune v3 block node."""
    node_type = node.get("type", "")

    if node_type in ("heading", "paragraph", "block_quote"):
        # These have children that are text nodes
        parts: list[str] = []

        def walk(children):
            if not children:
                return
            for child in children:
                ctype = child.get("type", "")
                if ctype in ("text", "block_text", "inline_text"):
                    parts.append(child.get("raw", child.get("text", "")))
                elif ctype in ("softbreak", "linebreak"):
                    parts.append("\n")
                elif "children" in child:
                    walk(child["children"])

        walk(node.get("children", []))
        return "".join(parts)

    elif node_type == "block_code":
        return node.get("raw", "")

    elif node_type == "list":
        parts: list[str] = []

        def walk_list(children):
            for child in children:
                if child.get("type") == "list_item":
                    walk_list(child.get("children", []))
                elif child.get("type") in ("block_text",):
                    for sub in child.get("children", []):
                        if sub.get("type") == "text":
                            parts.append(sub.get("raw", sub.get("text", "")))

        walk_list(node.get("children", []))
        return "".join(parts)

    elif node_type == "hr":
        return ""

    else:
        # Fallback: join any text-like children
        parts: list[str] = []

        def walk(children):
            if not children:
                return
            for child in children:
                ctype = child.get("type", "")
                if ctype in ("text", "block_text", "inline_text"):
                    parts.append(child.get("raw", child.get("text", "")))
                elif ctype in ("softbreak", "linebreak"):
                    parts.append("\n")
                elif "children" in child:
                    walk(child["children"])

        walk(node.get("children", []))
        return "".join(parts)


def _current_section(heading_stack: list[tuple[int, str]]) -> str:
    """Build a section locator from the current heading stack."""
    parts = [text for _level, text in heading_stack if text.strip()]
    return " / ".join(parts) if parts else "Untitled"


def extract_heading_structure(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract just the heading structure from parsed Markdown units.

    Returns a list of {level, text, location} for each heading block.
    """
    return [
        {
            "level": u["level"],
            "text": u["text"].strip(),
            "location": u["location"],
        }
        for u in units
        if u.get("type") == "heading"
    ]
