"""Structured output schema and parsing — Story 4.3.

Defines the parsing function that converts an LLM's raw output into a
GeneratedAnswer. The parsing is based on the output format instructions
in the prompt (Story 4.2).

The parsing approach: request JSON output from the LLM; parse the JSON
into the GeneratedAnswer model. If parsing fails, return None and let
the caller handle it as a generation failure.

Citation.excerpt is NOT populated here — it is left for the citation
resolver (Story 4.6 / app.citation) to populate from the store.

Do NOT implement:
- Epic 4.4 (abstention logic) — deferred
- Epic 4.5 (partial/conflicting evidence handling) — deferred
- Epic 5 (conversation) — deferred
- Epic 6 (citation UI) — deferred
- Epic 7 (API integration) — deferred
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.generation.interface import (
    Citation,
    EvidenceQuality,
    GeneratedAnswer,
)


def parse_llm_output(raw_text: str) -> GeneratedAnswer | None:
    """Parse an LLM's raw output into a GeneratedAnswer.

    The LLM is instructed (via the prompt's output format section) to
    produce JSON with the following structure:

    {
        "answer_text": "...",
        "citations": [
            {
                "passage_id": "...",
                "document_name": "...",
                "location": "..."
            }
        ],
        "evidence_quality": "sufficient" | "partial" | "insufficient" | "conflicting",
        "evidence_quality_narrative": "...",
        "is_abstention": false
    }

    Args:
        raw_text: The raw text output from the LLM.

    Returns:
        A GeneratedAnswer if parsing succeeded, None if parsing failed.
        On failure, the caller should treat this as a generation error
        (not fabricate an answer).
    """
    if not raw_text or not raw_text.strip():
        return None

    text = raw_text.strip()

    # Try to extract a JSON block from the response.
    # The LLM may wrap JSON in markdown code fences or return raw JSON.
    json_str = _extract_json_block(text)
    if json_str is None:
        return None

    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    # --- Extract fields with validation ---

    answer_text = _get_nonempty_str(data, "answer_text")
    if answer_text is None:
        return None

    evidence_quality_str = _get_nonempty_str(data, "evidence_quality")
    if evidence_quality_str is None:
        return None
    if not EvidenceQuality.is_valid(evidence_quality_str):
        return None
    evidence_quality = EvidenceQuality(evidence_quality_str)

    evidence_narrative = _get_nonempty_str(data, "evidence_quality_narrative")
    if evidence_narrative is None:
        return None

    is_abstention = _get_bool(data, "is_abstention", default=False)

    # --- Citations ---
    raw_citations = data.get("citations", [])
    if not isinstance(raw_citations, list):
        raw_citations = []

    citations: list[Citation] = []
    for cit_data in raw_citations:
        if not isinstance(cit_data, dict):
            continue
        pid = _get_nonempty_str(cit_data, "passage_id")
        doc_name = _get_nonempty_str(cit_data, "document_name")
        location = _get_nonempty_str(cit_data, "location")
        if pid is None or doc_name is None or location is None:
            continue
        citations.append(
            Citation(
                passage_id=pid,
                document_name=doc_name,
                location=location,
                excerpt="",  # populated later by the citation resolver
            )
        )

    return GeneratedAnswer(
        answer_text=answer_text,
        citations=citations,
        evidence_quality=evidence_quality,
        evidence_quality_narrative=evidence_narrative,
        is_abstention=is_abstention,
        generation_latency_seconds=0.0,  # set by the caller after timing the LLM call
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def _extract_json_block(text: str) -> str | None:
    """Try to extract a JSON string from LLM output.

    Handles:
    - Raw JSON: '{"key": "value"}'
    - Markdown-fenced JSON: '```json\n{...}\n```' or '```\n{...}\n```'

    Returns the JSON string if found, None otherwise.
    """
    text = text.strip()

    # Try markdown fences first
    m = _JSON_BLOCK_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        if _looks_like_json(candidate):
            return candidate

    # Try raw JSON (the whole text is JSON)
    if _looks_like_json(text):
        return text

    # Try to find a JSON object starting with { and ending with }
    brace_start = text.find("{")
    if brace_start == -1:
        return None
    brace_end = text.rfind("}")
    if brace_end == -1 or brace_end <= brace_start:
        return None
    candidate = text[brace_start : brace_end + 1]
    if _looks_like_json(candidate):
        return candidate

    return None


def extract_json_block(text: str) -> str | None:
    """Public form of the JSON-object extraction used for LLM output (e.g. by the judge)."""
    return _extract_json_block(text)


def _looks_like_json(s: str) -> bool:
    """Quick check: does *s* start with '{' and end with '}'?"""
    return s.startswith("{") and s.endswith("}")


def _get_nonempty_str(data: dict[str, Any], key: str) -> str | None:
    """Get a non-empty string from *data[key]*, or None."""
    val = data.get(key)
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def _get_bool(data: dict[str, Any], key: str, default: bool = False) -> bool:
    """Get a boolean from *data[key]*, with a default fallback.

    Accepts: bool, or int 0/1, or str "true"/"false"/"1"/"0".
    """
    val = data.get(key)
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val != 0
    if isinstance(val, str):
        low = val.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
    return default
