"""Deterministic stand-ins for the LLM and embedding model in integration tests."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from app.generation.interface import LLMResponse

DOCS = Path(__file__).resolve().parents[2] / "evaluation" / "dataset" / "documents"


class HashEmbedding:
    """Deterministic bag-of-words embedding (no model download)."""

    dimension = 64

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class ScriptedLLM:
    """Returns a response built from the prompt by a test-supplied function."""

    def __init__(self, respond):
        self._respond = respond
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(raw_text=self._respond(prompt))


_PASSAGE_RE = re.compile(
    r"- \*\*Passage ID:\*\* (?P<id>[^\n]+)\n- \*\*Text:\*\*\n(?P<text>.*?)(?=\n### Passage |\n\n## )",
    re.DOTALL,
)


def passages_in(prompt: str) -> list[tuple[str, str]]:
    """(passage_id, text) pairs in the order they appear in the prompt."""
    return [(m["id"].strip(), m["text"]) for m in _PASSAGE_RE.finditer(prompt)]


def passage_ids_in(prompt: str) -> list[str]:
    return [pid for pid, _ in passages_in(prompt)]


def answer_json(citations: list[str], quality: str = "sufficient", text: str = "AES-256.") -> str:
    return json.dumps({
        "answer_text": text,
        "citations": [
            {"passage_id": pid, "document_name": "model-supplied", "location": "model-supplied"}
            for pid in citations
        ],
        "evidence_quality": quality,
        "evidence_quality_narrative": "narrative",
        "is_abstention": quality == "insufficient",
    })


class CiteLLM(ScriptedLLM):
    """Cites the first retrieved passage containing the scripted phrase for each turn.

    With no matching passage it answers "insufficient" with no citations.
    """

    def __init__(self, phrases: list[str]):
        self._phrases = list(phrases)
        super().__init__(self._respond_to)

    def _respond_to(self, prompt: str) -> str:
        phrase = self._phrases.pop(0)
        hits = [pid for pid, text in passages_in(prompt) if phrase.lower() in text.lower()]
        if not hits:
            return answer_json([], quality="insufficient", text="Not in the documents.")
        return answer_json([hits[0]], text=f"Answer citing '{phrase}'.")
