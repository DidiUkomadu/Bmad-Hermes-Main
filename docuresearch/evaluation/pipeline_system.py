"""Evaluate the real ResearchPipeline against the evaluation dataset.

Provides ``PipelineSystem`` (a SystemInterface adapter over
``app.pipeline.ResearchPipeline``) and a CLI that ingests the dataset
documents into a fresh temporary store and runs every question through
real hybrid retrieval and the configured LLM.

Usage (from the docuresearch/ directory, with DOCURESEARCH_LLM_* set):
    python -m evaluation.pipeline_system --label baseline
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from app.config import Settings
from app.generation.openai_compat import OpenAICompatibleLLM
from app.generation.prompt import RetrievedContext
from app.main import _build_pipeline
from app.pipeline import GenerationError, ResearchPipeline
from app.store.schema import get_connection, get_passage, set_db_path
from evaluation.judge import FaithfulnessJudge
from evaluation.runner import DATASET_DIR, EvaluationRunner, SystemInterface

GENERATION_FAILED_PREFIX = "GENERATION FAILED:"


class PipelineSystem(SystemInterface):
    """SystemInterface backed by the production ResearchPipeline."""

    def __init__(self, pipeline: ResearchPipeline, db_path: str) -> None:
        self._pipeline = pipeline
        self._db_path = db_path
        self._last_context: RetrievedContext | None = None

    def retrieve(self, query: str, scope_doc_id: str | None = None) -> list[dict[str, Any]]:
        self._last_context = self._pipeline.retrieve(query, document_id=scope_doc_id)
        return [
            {
                "passage_id": p.passage_id,
                "score": p.score,
                "document_name": p.document_name,
                "location": p.location,
                "text": p.text,
            }
            for p in self._last_context.passages
        ]

    def generate(
        self,
        query: str,
        passages: list[dict[str, Any]],
        scope_doc_id: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        context = self._last_context
        if context is None or context.query != query:
            context = self._pipeline.retrieve(query, document_id=scope_doc_id)
        try:
            answer = self._pipeline.generate(context)
        except GenerationError as exc:
            return {
                "answer": f"{GENERATION_FAILED_PREFIX} {exc}",
                "citations": [],
                "evidence_quality": None,
                "evidence_narrative": "",
                "abstention": False,
            }
        return {
            "answer": answer.answer_text,
            "citations": [
                {
                    "passage_id": c.passage_id,
                    "document_name": c.document_name,
                    "location": c.location,
                    "excerpt": c.excerpt,
                }
                for c in answer.citations
            ],
            "evidence_quality": answer.evidence_quality.value,
            "evidence_narrative": answer.evidence_quality_narrative,
            "abstention": answer.is_abstention,
        }

    def resolve_passage(self, passage_id: str) -> dict[str, Any] | None:
        set_db_path(self._db_path)
        conn = get_connection()
        try:
            p = get_passage(conn, passage_id)
        finally:
            conn.close()
        if p is None:
            return None
        return {
            "passage_id": p.id,
            "document_id": p.document_id,
            "location": p.location,
            "text": p.text,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default="pipeline", help="Label for the results file")
    parser.add_argument("--no-save", action="store_true", help="Do not write a results file")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the faithfulness judge (halves model requests; faithfulness shows N/A)",
    )
    parser.add_argument(
        "--questions",
        help="Comma-separated question IDs to run (default: all), e.g. q-001,q-002",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    settings = Settings.load()
    if not settings.llm.configured:
        print(
            "LLM backend not configured: set DOCURESEARCH_LLM_BASE_URL and "
            "DOCURESEARCH_LLM_MODEL (and DOCURESEARCH_LLM_API_KEY) in the environment "
            "or docuresearch/.env. See .env.example.",
            file=sys.stderr,
        )
        return 2
    print(f"Model: {settings.llm.model} at {settings.llm.base_url}")
    judge_llm = None
    if not args.no_judge:
        judge_model = os.environ.get("DOCURESEARCH_JUDGE_MODEL") or settings.llm.model
        judge_llm = OpenAICompatibleLLM(
            settings.llm.base_url, judge_model, settings.llm.api_key,
            settings.llm.timeout_seconds,
        )
        note = "" if judge_model != settings.llm.model else " (same model as answers)"
        print(f"Judge: {judge_model}{note}")

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "eval.db"
        pipeline, llm = _build_pipeline(dataclasses.replace(settings, db_path=db_path))
        db_path = str(db_path)
        for doc in sorted((DATASET_DIR / "documents").iterdir()):
            if doc.is_file():
                info = pipeline.ingest(doc)
                print(f"Ingested {info['document_name']}: {info['passage_count']} passages")

        runner = EvaluationRunner(
            judge=FaithfulnessJudge(judge_llm) if judge_llm else None
        )
        questions = runner.load_questions()
        if args.questions:
            wanted = {q.strip() for q in args.questions.split(",") if q.strip()}
            unknown = wanted - {q.id for q in questions}
            if unknown:
                print(f"Unknown question IDs: {', '.join(sorted(unknown))}", file=sys.stderr)
                return 2
            questions = [q for q in questions if q.id in wanted]
        try:
            results = runner.run(questions, system=PipelineSystem(pipeline, db_path))
        finally:
            llm.close()
            if judge_llm:
                judge_llm.close()

    print(runner.report(results))
    failures = [r for r in results if r.system_answer.startswith(GENERATION_FAILED_PREFIX)]
    if failures:
        print(
            f"WARNING: {len(failures)}/{len(results)} questions failed generation "
            f"({', '.join(r.question_id for r in failures)}); metrics above are unreliable."
        )
    if judge_llm is None:
        print("Faithfulness not judged (--no-judge): reported as N/A.")
    if not args.no_save:
        print(f"Results saved to {runner.save_results(results, label=args.label)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
