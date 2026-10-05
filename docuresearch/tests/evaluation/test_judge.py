"""Tests for the faithfulness judge (methodology §2.3–2.5)."""

from __future__ import annotations

import json

from app.generation.interface import LLMResponse
from evaluation.judge import FaithfulnessJudge, build_judge_prompt, parse_judge_output
from evaluation.runner import EvaluationRunner, SystemInterface
from evaluation.schema import EvaluationQuestion, GoldPassage, QuestionType

PASSAGES = [
    {"passage_id": "p1", "text": "Keys must be rotated every 90 days."},
    {"passage_id": "p2", "text": "The protocol uses AES-256."},
]


def _judge_json(claims, citations=()):
    return json.dumps({"claims": list(claims), "citations": list(citations)})


def _claim(text, verdict, ids=(), reason=""):
    return {"claim": text, "verdict": verdict, "passage_ids": list(ids), "reason": reason}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_all_claims_supported_is_faithful():
    raw = _judge_json(
        [_claim("Keys rotate every 90 days", "supported", ["p1"]),
         _claim("It uses AES-256", "supported", ["p2"])],
        [{"passage_id": "p1", "supports_a_claim": True}],
    )
    v = parse_judge_output(raw, {"p1", "p2"}, ["p1"])
    assert v.faithful is True
    assert (v.claims_total, v.claims_supported) == (2, 2)
    assert v.unsupported_claims == []
    assert v.citation_support == 1.0


def test_unsupported_claim_is_listed_with_reason():
    raw = _judge_json([
        _claim("Keys rotate every 90 days", "supported", ["p1"]),
        _claim("Keys are 4096-bit RSA", "unsupported", [], "no passage mentions RSA"),
    ])
    v = parse_judge_output(raw, {"p1", "p2"}, [])
    assert v.faithful is False
    assert (v.claims_total, v.claims_supported) == (2, 1)
    assert v.unsupported_claims == ["Keys are 4096-bit RSA (no passage mentions RSA)"]


def test_supported_verdict_needs_a_real_retrieved_passage():
    raw = _judge_json([_claim("Something", "supported", ["not-retrieved"])])
    v = parse_judge_output(raw, {"p1"}, [])
    assert v.faithful is False


def test_labelled_inference_is_not_a_hallucination():
    raw = _judge_json([_claim("This suggests keys are short-lived", "labelled_inference")])
    assert parse_judge_output(raw, {"p1"}, []).faithful is True


def test_citation_support_counts_each_citation():
    raw = _judge_json(
        [_claim("Keys rotate every 90 days", "supported", ["p1"])],
        [{"passage_id": "p1", "supports_a_claim": True},
         {"passage_id": "p2", "supports_a_claim": False}],
    )
    assert parse_judge_output(raw, {"p1", "p2"}, ["p1", "p2"]).citation_support == 0.5


def test_no_citations_means_no_citation_support_score():
    raw = _judge_json([_claim("x", "supported", ["p1"])])
    assert parse_judge_output(raw, {"p1"}, []).citation_support is None


def test_answer_without_factual_claims_is_faithful():
    v = parse_judge_output(_judge_json([]), {"p1"}, [])
    assert v.faithful is True and v.claims_total == 0


def test_fenced_json_is_accepted():
    raw = "```json\n" + _judge_json([_claim("x", "supported", ["p1"])]) + "\n```"
    assert parse_judge_output(raw, {"p1"}, []).faithful is True


def test_malformed_output_is_an_error_not_a_verdict():
    v = parse_judge_output("I think it is fine.", {"p1"}, [])
    assert v.faithful is None
    assert v.error


def test_prompt_contains_question_answer_and_marked_passages():
    prompt = build_judge_prompt("How often?", "Every 90 days.", PASSAGES, ["p1"])
    assert "QUESTION:\nHow often?" in prompt
    assert "ANSWER:\nEvery 90 days." in prompt
    assert "--- Passage ID: p1 [CITED]\nKeys must be rotated every 90 days." in prompt
    assert "--- Passage ID: p2\nThe protocol uses AES-256." in prompt
    assert "Ignore your own knowledge" in prompt


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


class _LLM:
    def __init__(self, raw):
        self.raw = raw
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return LLMResponse(raw_text=self.raw)


class _System(SystemInterface):
    def __init__(self, generated):
        self.generated = generated

    def retrieve(self, query, scope_doc_id=None):
        return PASSAGES

    def generate(self, query, passages, scope_doc_id=None, conversation_history=None):
        return self.generated

    def resolve_passage(self, passage_id):
        return None


QUESTION = EvaluationQuestion(
    id="q-x", text="How often are keys rotated?", type=QuestionType.SINGLE_SOURCE,
    gold_answer="Every 90 days.",
    gold_passages=[GoldPassage("p1", "d", "d.pdf", "page:1", "90 days")],
)

ANSWER = {
    "answer": "Every 90 days, using 4096-bit RSA keys.",
    "citations": [{"passage_id": "p1"}],
    "evidence_quality": "sufficient",
    "evidence_narrative": "",
    "abstention": False,
}


def test_runner_records_judge_verdict(tmp_path):
    llm = _LLM(_judge_json(
        [_claim("Every 90 days", "supported", ["p1"]),
         _claim("4096-bit RSA keys", "unsupported", [], "not in passages")],
        [{"passage_id": "p1", "supports_a_claim": True}],
    ))
    runner = EvaluationRunner(results_dir=tmp_path, judge=FaithfulnessJudge(llm))
    result = runner._run_one(QUESTION, _System(ANSWER))

    assert result.answer_faithful is False
    assert (result.claims_total, result.claims_supported) == (2, 1)
    assert result.unsupported_claims == ["4096-bit RSA keys (not in passages)"]
    assert result.citation_support == 1.0
    assert "Keys must be rotated every 90 days." in llm.prompts[0]

    report = runner.report([result])
    assert "Answer Faithfulness:  0.0% (0/1)" in report
    assert "Hallucination Rate:   100.0% (1/1) of answers" in report
    assert "Unsupported Claims:   50.0% (1/2 claims)" in report
    assert "UNSUPPORTED: 4096-bit RSA keys (not in passages)" in report


def test_runner_does_not_judge_failed_generation(tmp_path):
    llm = _LLM(_judge_json([]))
    failed = {**ANSWER, "answer": "GENERATION FAILED: boom", "evidence_quality": None}
    runner = EvaluationRunner(results_dir=tmp_path, judge=FaithfulnessJudge(llm))
    result = runner._run_one(QUESTION, _System(failed))
    assert result.answer_faithful is None
    assert llm.prompts == []


def test_runner_without_judge_reports_na(tmp_path):
    result = EvaluationRunner(results_dir=tmp_path)._run_one(QUESTION, _System(ANSWER))
    assert result.answer_faithful is None
    assert result.claims_total is None


def test_judge_error_is_recorded(tmp_path):
    runner = EvaluationRunner(results_dir=tmp_path, judge=FaithfulnessJudge(_LLM("not json")))
    result = runner._run_one(QUESTION, _System(ANSWER))
    assert result.answer_faithful is None
    assert result.judge_error
    assert "Judge errors:         1 answers not judged" in runner.report([result])


def test_failed_generation_is_not_scored_for_abstention(tmp_path):
    failed = {**ANSWER, "answer": "GENERATION FAILED: 429", "evidence_quality": None,
              "abstention": False}
    result = EvaluationRunner(results_dir=tmp_path)._run_one(QUESTION, _System(failed))
    assert result.abstention_correct is None


def test_accurate_absence_statement_is_not_a_hallucination():
    raw = json.dumps({"claims": [], "absence_statements": [
        {"statement": "The passages do not recommend any Python libraries", "accurate": True},
    ]})
    v = parse_judge_output(raw, {"p1"}, [])
    assert v.faithful is True
    assert v.claims_total == 0


def test_false_absence_statement_is_unsupported():
    raw = json.dumps({"claims": [], "absence_statements": [
        {"statement": "The passages do not say how often keys rotate", "accurate": False,
         "passage_ids": ["p1"]},
    ]})
    v = parse_judge_output(raw, {"p1"}, [])
    assert v.faithful is False
    assert v.unsupported_claims == [
        "False absence: The passages do not say how often keys rotate (contradicted by p1)"
    ]


def test_prompt_asks_for_absence_statements():
    assert "absence_statements" in build_judge_prompt("q", "a", PASSAGES, [])
