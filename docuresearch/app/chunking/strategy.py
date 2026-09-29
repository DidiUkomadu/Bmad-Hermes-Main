"""Configurable passage chunking strategy (Story 1.4).

Splits a RawDocument into Passage objects suitable for storage and retrieval.

Design:
- Prefer paragraph boundaries for PDF and TXT (already identified in RawDocument units)
- Prefer heading/section boundaries for Markdown: a section (heading + all its
  content paragraphs) is treated as one logical block for chunking
- Fall back to sized chunks with overlap when no structural boundaries are available
- Configuration is explicit and injectable (ChunkingConfig)
- Produces Passage objects compatible with the approved store contract
- Preserves stable document_id, location, and offset metadata
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models import Passage, RawDocument

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ChunkingConfig:
    """Configuration for the chunking strategy.

    All values are tunable for the MVP. Default values target the approved
    range: 250-600 characters per passage, 80-150 characters overlap.
    """

    target_length: int = 450
    """Target characters per passage (250-600 range)."""

    overlap: int = 100
    """Characters of overlap between consecutive sized chunks (80-150 range)."""

    prefer_paragraph_boundaries: bool = True
    """Prefer paragraph boundaries for PDF/TXT content."""

    prefer_heading_boundaries: bool = True
    """Prefer heading/section boundaries for Markdown content."""

    min_chunk_size: int = 50
    """Minimum characters per chunk. Below this, merge with adjacent chunk."""

    max_chunk_size: int = 1200
    """Maximum characters per chunk. Above this, force-split at overlap boundary."""


# ---------------------------------------------------------------------------
# Chunking strategy
# ---------------------------------------------------------------------------

def chunk_passages(raw_doc: RawDocument, config: ChunkingConfig | None = None) -> list[Passage]:
    """Split a RawDocument into Passage objects.

    Args:
        raw_doc: The parsed RawDocument (from PDF, Markdown, or TXT handler).
        config: Chunking configuration. Uses defaults if None.

    Returns:
        List of Passage objects with stable IDs, locations, and offsets.
    """
    if config is None:
        config = ChunkingConfig()

    if raw_doc.format == "markdown" and config.prefer_heading_boundaries:
        return _chunk_by_blocks(
            raw_doc, config, boundary_type="heading", location_prefix="§",
        )
    elif config.prefer_paragraph_boundaries:
        return _chunk_by_blocks(
            raw_doc, config, boundary_type="paragraph", location_prefix="¶",
        )
    else:
        return _chunk_sized(raw_doc, config)


def _chunk_by_blocks(
    raw_doc: RawDocument,
    config: ChunkingConfig,
    boundary_type: str,
    location_prefix: str,
) -> list[Passage]:
    """Chunk by structural boundaries.

    Consecutive RawDocument units sharing the same location are combined
    into a single logical block before chunking.
    """
    passages: list[Passage] = []
    chunk_index = 0

    # Group consecutive units with the same location into logical blocks
    blocks: list[dict[str, Any]] = []
    current_block = None

    for unit in raw_doc.units:
        text = unit["text"]
        if not text:
            continue

        loc = unit.get("location", f"{location_prefix}unknown")

        if current_block is None:
            current_block = {
                "text": text,
                "location": loc,
                "start_offset": unit["start_offset"],
                "end_offset": unit["end_offset"],
            }
        elif loc == current_block["location"]:
            # Same separator the handlers use in full_text, so block text
            # stays consistent with offsets and words never run together.
            current_block["text"] += "\n\n" + text
            current_block["end_offset"] = unit["end_offset"]
        else:
            blocks.append(current_block)
            current_block = {
                "text": text,
                "location": loc,
                "start_offset": unit["start_offset"],
                "end_offset": unit["end_offset"],
            }

    if current_block is not None:
        blocks.append(current_block)

    for block in blocks:
        text = block["text"]
        if not text:
            continue

        location = _make_location(block["location"], chunk_index)
        pid = f"{raw_doc.document_id}:{location}"

        if _fits_in_one_chunk(text, config):
            p = Passage(
                id=pid,
                document_id=raw_doc.document_id,
                text=text,
                location=location,
                start_offset=block["start_offset"],
                end_offset=block["end_offset"],
            )
            # Merge with previous passage if below min size.
            # First passage can always merge (no previous location conflict).
            # Subsequent passages require same location to avoid merging across
            # section boundaries.
            can_merge = passages and len(passages[-1].text) < config.min_chunk_size
            if not can_merge:
                passages.append(p)
                chunk_index += 1
            elif len(passages) == 1:
                # First passage: always allow merge (prevents tiny first chunk)
                prev = passages[-1]
                prev.text = prev.text + " " + p.text
                prev.end_offset = p.end_offset
            elif passages[-1].location == p.location:
                # Same location: allow merge
                prev = passages[-1]
                prev.text = prev.text + " " + p.text
                prev.end_offset = p.end_offset
            else:
                # Different location AND not first: can't merge
                passages.append(p)
                chunk_index += 1
        else:
            sub_chunks = _sized_chunks(
                text, config,
                base_location=block["location"],
                chunk_index_start=chunk_index,
                doc_id=raw_doc.document_id,
                start_offset=block["start_offset"],
            )
            # Process all sub-chunks in order
            for sc in sub_chunks:
                if passages and len(passages[-1].text) < config.min_chunk_size:
                    prev = passages[-1]
                    prev.text = prev.text + " " + sc.text
                    prev.end_offset = sc.end_offset
                else:
                    passages.append(sc)
                    chunk_index += 1

    return passages


def _chunk_sized(raw_doc: RawDocument, config: ChunkingConfig) -> list[Passage]:
    """Chunk the entire document text into sized passages with overlap."""
    full_text = raw_doc.full_text
    if not full_text:
        return []

    passages: list[Passage] = []
    chunk_index = 0
    pos = 0
    doc_length = len(full_text)

    while pos < doc_length:
        remaining = doc_length - pos
        if remaining <= config.max_chunk_size:
            end = doc_length
        else:
            end = min(pos + config.target_length, doc_length)
            if (end - pos) > config.max_chunk_size:
                end = min(pos + config.max_chunk_size, doc_length)

        if end < doc_length and end > pos:
            split_at = _find_split_point(full_text, pos, end, config.target_length)
            if split_at > pos:
                end = split_at
        elif end == doc_length and remaining > config.max_chunk_size:
            # At the tail and remaining text exceeds max_chunk_size:
            # force a split at max_chunk_size boundary
            force_end = min(pos + config.max_chunk_size, doc_length)
            split_at = _find_split_point(full_text, pos, force_end, config.max_chunk_size)
            if split_at > pos:
                end = split_at
            else:
                end = force_end

        chunk_text = full_text[pos:end]

        if len(chunk_text) < config.min_chunk_size and chunk_index > 0:
            prev = passages[-1]
            prev.text = prev.text + " " + chunk_text
            prev.end_offset = pos + len(chunk_text)
            pos = max(pos + 1, end - config.overlap)
            chunk_index += 1
            if end >= doc_length:
                break
            continue

        location = f"¶chunk:{chunk_index}"
        pid = f"{raw_doc.document_id}:{location}"
        passages.append(Passage(
            id=pid,
            document_id=raw_doc.document_id,
            text=chunk_text,
            location=location,
            start_offset=pos,
            end_offset=min(pos + len(chunk_text), doc_length),
        ))

        if end >= doc_length:
            break

        pos = max(pos + 1, end - config.overlap)
        chunk_index += 1

    return passages


def _sized_chunks(
    text: str,
    config: ChunkingConfig,
    base_location: str,
    chunk_index_start: int,
    doc_id: str,
    start_offset: int,
) -> list[Passage]:
    """Split a single text block into sized chunks with overlap."""
    chunks: list[Passage] = []
    pos = 0
    text_len = len(text)
    chunk_index = chunk_index_start

    while pos < text_len:
        remaining = text_len - pos

        if remaining <= config.max_chunk_size:
            end = text_len
        else:
            end = min(pos + config.target_length, text_len)
            if (end - pos) > config.max_chunk_size:
                end = min(pos + config.max_chunk_size, text_len)

        if end < text_len and end > pos:
            split_at = _find_split_point(text, pos, end, config.target_length)
            if split_at > pos:
                end = split_at
        elif end == text_len and remaining > config.max_chunk_size:
            force_end = min(pos + config.max_chunk_size, text_len)
            split_at = _find_split_point(text, pos, force_end, config.max_chunk_size)
            if split_at > pos:
                end = split_at
            else:
                end = force_end

        chunk_text = text[pos:end]
        abs_start = start_offset + pos
        abs_end = start_offset + min(pos + len(chunk_text), text_len)

        if len(chunk_text) < config.min_chunk_size and chunks:
            prev = chunks[-1]
            prev.text = prev.text + " " + chunk_text
            prev.end_offset = abs_end
            pos = max(pos + 1, end - config.overlap)
            chunk_index += 1
            if end >= text_len:
                break
            continue

        location = _make_location(base_location, chunk_index)
        pid = f"{doc_id}:{location}"
        chunks.append(Passage(
            id=pid,
            document_id=doc_id,
            text=chunk_text,
            location=location,
            start_offset=abs_start,
            end_offset=abs_end,
        ))

        if end >= text_len:
            break

        pos = max(pos + 1, end - config.overlap)
        chunk_index += 1

    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fits_in_one_chunk(text: str, config: ChunkingConfig) -> bool:
    """Check if text fits in a single chunk without splitting."""
    return len(text) <= config.max_chunk_size


def _find_split_point(text: str, start: int, target_end: int, target_length: int) -> int:
    """Find a good split point near target_end, preferring whitespace."""
    text_len = len(text)
    search_end = min(target_end + 50, text_len)
    search_start = max(start, target_end - 50)

    for i in range(target_end - 1, search_start, -1):
        if text[i] in (" ", "\t", "\n", "\r"):
            return i + 1

    for i in range(target_end, search_end):
        if text[i] in (" ", "\t", "\n", "\r"):
            return i + 1

    return target_end


def _make_location(base_location: str, chunk_index: int) -> str:
    """Create a passage location string from a base location and chunk index."""
    clean = base_location.strip()
    if not clean or clean == "unknown":
        return f"¶chunk:{chunk_index}"
    return f"{clean}:chunk:{chunk_index}"
