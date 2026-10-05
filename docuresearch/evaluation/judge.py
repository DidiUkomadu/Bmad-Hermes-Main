"""LLM-as-judge for answer faithfulness and citation support (methodology §2.3–2.5).

For each answer, a judge model breaks the answer into factual claims and checks
each one against the passages retrieved for that question:

- **Faithfulness (§2.4):** an answer is faithful only if every factual claim is
  supported by a retrieved passage. Unsupported claims are listed individually.
- **Hallucination rate (§2.5):** an inference counts as unsupported unless the
  answer labels it as an inference. Reported per answer and per claim.
- **False absences:** "the documents do not say X" is checked separately. It
  is fine when true; when a retrieved passage does say X it counts as an
  unsupported claim (the system wrongly declared the evidence missing).
- **Citation support (§2.3):** the fraction of the answer's citations whose
  passage actually supports at least one claim, independent of the gold list.

The judge only sees the question, the answer, and the retrieved passages. It is
told to ignore outside knowledge: a claim that is true in the world but absent
from the passages is unsupported.

Caveat: a model judging its own answers can be lenient. Use a different (ideally
stronger) judge model via DOCURESEARCH_JUDGE_MODEL where possible, and spot-check
the listed unsupported claims by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.generation.interface import LLMInterface
from app.generation.schema import extract_json_block

_MAX_PASSAGE_CHARS = 2000

_INSTRUCTIONS = """\
You are a strict fact-checker for a document question-answering system.

Decide whether the ANSWER below is supported by the PASSAGES, which are the only
evidence the system had. Ignore your own knowledge: a claim that is true in the
real world but not stated in (or directly implied by) the passages is UNSUPPORTED.

Steps:
1. Split the answer into atomic factual claims about the subject matter.
   Statements that the passages do NOT contain something ("the documents do not
   say X") are not claims: list them under "absence_statements" instead, and
   mark each "accurate" if no passage contains that information, or not
   accurate if some passage does contain it. Also leave out descriptions of
   evidence quality and restatements of the question.
2. For each claim decide:
   - "supported": stated in, or directly implied by, at least one passage.
     List the IDs of the passages that support it.
   - "labelled_inference": not stated, but the answer explicitly presents it as
     an inference or assumption (e.g. "this suggests", "likely").
   - "unsupported": anything else, including claims attributed to a document
     that does not contain them.
3. For each CITED passage ID, decide whether that passage supports at least one
   of the answer's claims.

Respond with a single JSON object and nothing else:
{
  "claims": [
    {"claim": "...", "verdict": "supported" | "labelled_inference" | "unsupported",
     "passage_ids": ["..."], "reason": "short reason"}
  ],
  "absence_statements": [
    {"statement": "...", "accurate": true, "passage_ids": ["... if not accurate"]}
  ],
  "citations": [{"passage_id": "...", "supports_a_claim": true}]
}"""


@dataclass
class JudgeVerdict:
    """The judge's assessment of one answer."""

    faithful: bool | None
    claims_total: int = 0
    claims_supported: int = 0
    unsupported_claims: list[str] = field(default_factory=list)
    citation_support: float | None = None  # None when the answer has no citations
    error: str | None = None


def build_judge_prompt(
    question: str,
    answer: str,
    passages: list[dict[str, Any]],
    cited_ids: list[str],
) -> str:
    """The judge prompt: instructions, question, answer, and the retrieved passages."""
    parts = [_INSTRUCTIONS, f"QUESTION:\n{question}", f"ANSWER:\n{answer}"]
    cited = set(cited_ids)
    lines = ["PASSAGES (the system's retrieved evidence):"]
    for p in passages:
        marker = " [CITED]" if p["passage_id"] in cited else ""
        text = p.get("text", "")[:_MAX_PASSAGE_CHARS]
        lines.append(f"--- Passage ID: {p['passage_id']}{marker}\n{text}")
    if not passages:
        lines.append("(none)")
    parts.append("\n".join(lines))
    parts.append(f"CITED PASSAGE IDS: {json.dumps(cited_ids)}")
    return "\n\n".join(parts)


def parse_judge_output(raw: str, passage_ids: set[str], cited_ids: list[str]) -> JudgeVerdict:
    """Turn the judge's JSON into a verdict; malformed output becomes an error verdict."""
    block = extract_json_block(raw or "")
    try:
        data = json.loads(block) if block else None
    except ValueError:
        data = None
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        return JudgeVerdict(faithful=None, error="Judge output was not valid JSON")

    total = supported = 0
    unsupported: list[str] = []
    for item in data["claims"]:
        if not isinstance(item, dict) or not str(item.get("claim", "")).strip():
            continue
        total += 1
        verdict = str(item.get("verdict", "")).strip().lower()
        ids = {str(i) for i in item.get("passage_ids") or []}
        # A "supported" verdict must point at a real retrieved passage.
        if verdict == "supported" and ids & passage_ids:
            supported += 1
        elif verdict == "labelled_inference":
            supported += 1
        else:
            reason = str(item.get("reason", "")).strip()
            unsupported.append(f"{item['claim']}" + (f" ({reason})" if reason else ""))

    # "The documents do not say X" is fine when true; when a passage does say X,
    # it is a false abstention and counts as an unsupported claim.
    for item in data.get("absence_statements") or []:
        if not isinstance(item, dict) or not str(item.get("statement", "")).strip():
            continue
        if item.get("accurate") is False:
            total += 1
            ids = ", ".join(str(i) for i in item.get("passage_ids") or [])
            unsupported.append(
                f"False absence: {item['statement']}" + (f" (contradicted by {ids})" if ids else "")
            )

    citation_support = None
    if cited_ids:
        flags = {
            str(c.get("passage_id")): bool(c.get("supports_a_claim"))
            for c in data.get("citations") or []
            if isinstance(c, dict)
        }
        citation_support = sum(1 for pid in cited_ids if flags.get(pid)) / len(cited_ids)

    return JudgeVerdict(
        faithful=not unsupported,
        claims_total=total,
        claims_supported=supported,
        unsupported_claims=unsupported,
        citation_support=citation_support,
    )


class FaithfulnessJudge:
    """Judges answers with an LLM (any LLMInterface)."""

    def __init__(self, llm: LLMInterface) -> None:
        self._llm = llm

    def judge(
        self,
        question: str,
        answer: str,
        passages: list[dict[str, Any]],
        cited_ids: list[str],
    ) -> JudgeVerdict:
        prompt = build_judge_prompt(question, answer, passages, cited_ids)
        response = self._llm.generate(prompt)
        if not response.raw_text:
            return JudgeVerdict(faithful=None, error=response.parse_error or "No judge output")
        return parse_judge_output(
            response.raw_text, {p["passage_id"] for p in passages}, cited_ids
        )
