"""Evaluation runner (Story 0.4).

Loads the evaluation dataset, submits questions to a system interface,
collects answers and citations, compares against gold data, and computes
the metrics defined in the evaluation methodology.

The runner is structured to work against:
- A full system (API or generation pipeline) — when available.
- Individual components (e.g. retrieval only) — during incremental development.
- A mock system — for testing the runner itself.

Usage:
    from evaluation.runner import EvaluationRunner
    from evaluation.schema import EvaluationQuestion

    runner = EvaluationRunner()
    questions = runner.load_questions()
    results = runner.run(questions, system=mx.MockSystem())
    runner.report(results)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from app.store.schema import (
    get_connection,
    set_db_path,
)
from evaluation.schema import (
    EvaluationQuestion,
    EvaluationResult,
    EvidenceQuality,
    QuestionType,
)

logger = logging.getLogger(__name__)

# Default paths
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
QUESTIONS_FILE = DATASET_DIR / "questions.json"
RESULTS_DIR = DATASET_DIR / "results"


class SystemInterface:
    """Abstract interface for the system under evaluation.

    Subclass or implement this to connect the runner to the actual system
    (generation pipeline, API, or a component being tested).

    The runner calls these methods in order for each question:
        1. retrieve       — get passages for the query
        2. generate       — produce an answer from retrieved passages
        3. resolve_passage — get passage text by ID (for citation verification)
    """

    def retrieve(
        self, query: str, scope_doc_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return a ranked list of passages for the query.

        Each entry: {"passage_id": str, "score": float, "document_id": str}
        """
        raise NotImplementedError

    def generate(
        self,
        query: str,
        passages: list[dict[str, Any]],
        scope_doc_id: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate an answer from retrieved passages.

        Returns: {
            "answer": str,
            "citations": [{"passage_id": str, "document_name": str,
                           "location": str, "excerpt": str}],
            "evidence_quality": str,       # "sufficient" | "partial" | ...
            "evidence_narrative": str,
            "abstention": bool,
        }
        """
        raise NotImplementedError

    def resolve_passage(self, passage_id: str) -> dict[str, Any] | None:
        """Resolve a passage ID to its text and metadata.

        Returns: {"passage_id": str, "document_id": str, "document_name": str,
                  "location": str, "text": str} or None if not found.
        """
        raise NotImplementedError


class StoreBackedSystem(SystemInterface):
    """A system interface backed by the SQLite content store.

    This is usable as soon as the store is populated — it can retrieve
    passages and resolve citations, even before the generation layer exists.
    """

    # Placeholder answers are copied from retrieved passages.
    answers_are_extractive = True

    def __init__(self, db_path: str):
        set_db_path(db_path)
        self._conn = None

    def _get_conn(self):
        if self._conn is None:
            self._conn = get_connection()
        return self._conn

    def retrieve(self, query: str, scope_doc_id: str | None = None) -> list[dict[str, Any]]:
        """Brute-force cosine similarity retrieval (approved MVP default).

        For the MVP, we use simple term-frequency cosine similarity: split
        query and passage into terms, compute TF vectors, and rank by cosine
        similarity. No embeddings, no BM25 (those come in later waves).
        """
        from collections import Counter

        conn = self._get_conn()

        # Get all passages (or scoped subset)
        if scope_doc_id:
            rows = conn.execute(
                "SELECT id, document_id, text, location FROM passages WHERE document_id = ?",
                (scope_doc_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, document_id, text, location FROM passages"
            ).fetchall()

        if not rows:
            return []

        # Build query term vector
        query_terms = _tokenize(query)
        query_vec = Counter(query_terms)

        results: list[dict[str, Any]] = []
        for row in rows:
            passage_id = row["id"]
            doc_id = row["document_id"]
            text = row["text"]
            location = row["location"]

            passage_terms = _tokenize(text)
            passage_vec = Counter(passage_terms)

            # Cosine similarity
            similarity = _cosine_similarity(query_vec, passage_vec)
            if similarity > 0:
                results.append({
                    "passage_id": passage_id,
                    "score": similarity,
                    "document_id": doc_id,
                    "text": text,
                    "location": location,
                })

        # Sort by score descending
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def generate(
        self,
        query: str,
        passages: list[dict[str, Any]],
        scope_doc_id: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate an answer from retrieved passages.

        For the MVP (before the LLM integration in Epic 4), this returns
        a placeholder answer that demonstrates the output structure. The
        actual LLM-backed generation comes later.

        The placeholder:
        - Synthesizes a basic answer from the top passage text.
        - Cites the top passage.
        - Reports evidence quality as "sufficient" if passages were found.
        """
        if not passages:
            return {
                "answer": (
                    "The documents do not contain enough information to answer "
                    "this definitively."
                ),
                "citations": [],
                "evidence_quality": "insufficient",
                "evidence_narrative": (
                    "No relevant passages were found for this query in the "
                    "uploaded documents."
                ),
                "abstention": True,
            }

        # Take top passages (up to 3 for the placeholder)
        top_passages = passages[:3]

        # Build a simple answer from the passage texts
        answer_parts = []
        citations = []
        for p in top_passages:
            text = p.get("text", "")
            loc = p.get("location", "unknown")

            # Try to get document name from store
            doc_name = self._get_document_name(p.get("document_id", ""))

            citations.append({
                "passage_id": p["passage_id"],
                "document_name": doc_name,
                "location": loc,
                "excerpt": text[:200] + ("..." if len(text) > 200 else ""),
            })
            answer_parts.append(f"[{doc_name}, {loc}]: {text[:150]}")

        answer = "Based on the retrieved passages:\n\n" + "\n\n".join(answer_parts)

        return {
            "answer": answer,
            "citations": citations,
            "evidence_quality": "sufficient",
            "evidence_narrative": "The documents support this directly.",
            "abstention": False,
        }

    def resolve_passage(self, passage_id: str) -> dict[str, Any] | None:
        """Resolve a passage ID from the content store."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT p.id, p.document_id, p.text, p.location, d.name AS doc_name "
            "FROM passages p JOIN documents d ON p.document_id = d.id "
            "WHERE p.id = ?",
            (passage_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "passage_id": row["id"],
            "document_id": row["document_id"],
            "document_name": row["doc_name"],
            "location": row["location"],
            "text": row["text"],
        }

    def _get_document_name(self, doc_id: str) -> str:
        """Get the document name for a document ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT name FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        return row["name"] if row else doc_id


# ---------------------------------------------------------------------------
# Tokenization and similarity (for brute-force cosine retrieval)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenization, lowercased."""
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def _cosine_similarity(vec_a: dict[str, int], vec_b: dict[str, int]) -> float:
    """Compute cosine similarity between two term-frequency dicts."""
    import math

    # Dot product
    dot = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in vec_a)
    if dot == 0:
        return 0.0

    # Magnitudes
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


def _rate(flags: list[bool]) -> str:
    """Format a list of booleans as 'NN.N% (hits/total)'."""
    hits = sum(1 for f in flags if f)
    return f"{hits / len(flags):.1%} ({hits}/{len(flags)})"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class EvaluationRunner:
    """Runs the evaluation dataset against a system and computes metrics."""

    def __init__(
        self,
        questions_path: Path = QUESTIONS_FILE,
        results_dir: Path = RESULTS_DIR,
    ):
        self.questions_path = questions_path
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_questions(self) -> list[EvaluationQuestion]:
        """Load all questions from the dataset JSON file."""
        with open(self.questions_path) as f:
            data = json.load(f)

        questions_data = data.get("questions", [])
        questions = [EvaluationQuestion.from_dict(q) for q in questions_data]
        logger.info("Loaded %d questions from %s", len(questions), self.questions_path)
        return questions

    def run(
        self,
        questions: list[EvaluationQuestion] | None = None,
        system: SystemInterface | None = None,
        db_path: str | None = None,
    ) -> list[EvaluationResult]:
        """Run the evaluation dataset against the system.

        Args:
            questions: Questions to run. If None, loads from the default dataset.
            system: System interface to evaluate. If None, creates a
                StoreBackedSystem using the given db_path.
            db_path: Path to the SQLite database (only used if system is None).

        Returns:
            List of EvaluationResult, one per question.
        """
        if questions is None:
            questions = self.load_questions()

        if system is None:
            if db_path is None:
                raise ValueError(
                    "Either system or db_path must be provided. "
                    "When evaluating before the API exists, pass a db_path "
                    "to use the StoreBackedSystem."
                )
            system = StoreBackedSystem(db_path)

        results: list[EvaluationResult] = []

        for q in questions:
            result = self._run_one(q, system)
            results.append(result)
            logger.info("  %s", result.summary_line())

        return results

    def _run_one(
        self, question: EvaluationQuestion, system: SystemInterface
    ) -> EvaluationResult:
        """Run a single question and produce an EvaluationResult."""
        start_total = time.monotonic()

        # Step 1: Retrieve passages
        start_retrieval = time.monotonic()
        retrieved = system.retrieve(question.text)
        retrieval_latency = time.monotonic() - start_retrieval

        # Step 2: Generate answer
        start_gen = time.monotonic()
        gen_output = system.generate(
            query=question.text,
            passages=retrieved,
            scope_doc_id=None,
        )
        generation_latency = time.monotonic() - start_gen

        total_latency = time.monotonic() - start_total

        # Step 3: Compute metrics
        result = EvaluationResult(
            question_id=question.id,
            system_answer=gen_output.get("answer", ""),
            system_citations=gen_output.get("citations", []),
            system_evidence_quality=(
                EvidenceQuality(gen_output["evidence_quality"])
                if gen_output.get("evidence_quality") else None
            ),
            system_evidence_narrative=gen_output.get("evidence_narrative", ""),
            system_abstention=gen_output.get("abstention", False),
            retrieval_latency_s=retrieval_latency,
            generation_latency_s=generation_latency,
            total_latency_s=total_latency,
            run_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        # Compute retrieval precision and recall
        self._compute_retrieval_metrics(result, question, retrieved)

        # Compute citation correctness
        self._compute_citation_correctness(result, question)

        # Compute answer faithfulness
        self._compute_faithfulness(result, question, retrieved, system)

        # Compute abstention accuracy
        self._compute_abstention(result, question)

        # Compute conflict handling
        self._compute_conflict(result, question)

        return result

    def _compute_retrieval_metrics(
        self,
        result: EvaluationResult,
        question: EvaluationQuestion,
        retrieved: list[dict[str, Any]],
    ) -> None:
        """Compute retrieval precision and recall."""
        # Gold relevant passage IDs
        gold_ids = {gp.passage_id for gp in question.gold_passages}

        if not gold_ids:
            # Insufficient-evidence question: no gold passages, so precision
            # and recall are undefined. Ranked retrieval always returns its
            # top-N candidates, so "retrieved nothing" is not a meaningful
            # success signal; these questions are scored on abstention instead.
            result.retrieval_precision = None  # N/A
            result.retrieval_recall = None  # N/A
            return

        retrieved_ids = {r["passage_id"] for r in retrieved}

        # One entry per distinct gold passage; a gold passage is hit when it
        # or any of its equivalents (same content in another document) is
        # retrieved.
        accepted_by_gold = {
            gp.passage_id: gp.accepted_ids() for gp in question.gold_passages
        }
        all_accepted = set().union(*accepted_by_gold.values())

        # Precision: fraction of retrieved that are relevant
        if retrieved_ids:
            result.retrieval_precision = (
                len(retrieved_ids & all_accepted) / len(retrieved_ids)
            )
        else:
            result.retrieval_precision = 0.0

        # Recall: fraction of gold relevant that were retrieved
        hits = sum(1 for ids in accepted_by_gold.values() if ids & retrieved_ids)
        result.retrieval_recall = hits / len(accepted_by_gold)

    def _compute_citation_correctness(
        self,
        result: EvaluationResult,
        question: EvaluationQuestion,
    ) -> None:
        """Compute citation correctness: what fraction of citations are correct."""
        citations = result.system_citations
        if not citations:
            result.citation_correctness = None  # N/A
            return

        gold_ids = set().union(*(gp.accepted_ids() for gp in question.gold_passages))

        correct = 0
        for cit in citations:
            pid = cit.get("passage_id", "")
            # A citation is correct if it references a gold passage
            # (and the passage exists — checked by the system)
            if pid in gold_ids:
                correct += 1

        result.citation_correctness = correct / len(citations)

    def _compute_faithfulness(
        self,
        result: EvaluationResult,
        question: EvaluationQuestion,
        retrieved: list[dict[str, Any]],
        system: SystemInterface,
    ) -> None:
        """Determine if the answer is faithful to retrieved passages.

        Only systems that build answers verbatim from retrieved passages
        (``answers_are_extractive = True``, e.g. the StoreBackedSystem
        placeholder) are faithful by construction. For LLM-generated answers
        faithfulness is not yet measured and is recorded as None rather than
        assumed — the report then omits it instead of claiming 100%.
        """
        if getattr(system, "answers_are_extractive", False):
            result.answer_faithful = True
        else:
            result.answer_faithful = None

    def _compute_abstention(
        self,
        result: EvaluationResult,
        question: EvaluationQuestion,
    ) -> None:
        """Evaluate whether the abstention decision was correct."""
        if question.type == QuestionType.INSUFFICIENT_EVIDENCE:
            # Should abstain
            result.abstention_correct = result.system_abstention
        else:
            # Should NOT abstain (sufficient evidence exists)
            result.abstention_correct = not result.system_abstention

    def _compute_conflict(
        self,
        result: EvaluationResult,
        question: EvaluationQuestion,
    ) -> None:
        """Evaluate whether conflicting evidence was correctly handled."""
        if question.type != QuestionType.CONFLICTING_EVIDENCE:
            result.conflict_handled = None  # N/A
            return

        # Check if the answer mentions both conflicting sources
        answer = result.system_answer.lower()
        conf_sources = question.conflicting_sources

        if not conf_sources:
            result.conflict_handled = False
            return

        # Each conflicting source must be surfaced: either named in the answer
        # text, or cited via a passage from that source's document (optional
        # "document_name" field, the stored filename). The model only sees
        # stored document names, so citation-based matching is the robust path.
        cited_docs = {
            c.get("document_name", "").lower() for c in result.system_citations
        }
        all_mentioned = True
        for src in conf_sources:
            source_name = src.get("source", "").lower()
            doc_name = src.get("document_name", "").lower()
            named = bool(source_name) and source_name in answer
            cited = bool(doc_name) and doc_name in cited_docs
            if not (named or cited):
                all_mentioned = False
                break

        # Check evidence quality is "conflicting"
        eq_correct = (
            result.system_evidence_quality == EvidenceQuality.CONFLICTING
        )

        result.conflict_handled = all_mentioned and eq_correct

    def report(self, results: list[EvaluationResult]) -> str:
        """Produce a readable text summary of evaluation results."""
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("DocuResearch Evaluation Results")
        lines.append(f"Questions: {len(results)}")
        lines.append(f"Run timestamp: {results[0].run_timestamp if results else 'N/A'}")
        lines.append("=" * 70)
        lines.append("")

        # Per-question details
        lines.append("Per-question results:")
        lines.append("-" * 70)
        for r in results:
            lines.append(r.summary_line())
            lines.append(f"  Answer: {r.system_answer[:120]}...")
            lines.append(f"  Citations: {len(r.system_citations)}")
            lines.append(f"  Evidence: {r.system_evidence_quality}")
            lines.append("")

        # Aggregate metrics
        lines.append("=" * 70)
        lines.append("Aggregate Metrics")
        lines.append("=" * 70)

        # Retrieval precision and recall
        precisions = [r.retrieval_precision for r in results if r.retrieval_precision is not None]
        recalls = [r.retrieval_recall for r in results if r.retrieval_recall is not None]

        if precisions:
            lines.append(f"  Retrieval Precision:  mean={sum(precisions)/len(precisions):.3f}  "
                        f"min={min(precisions):.3f}  max={max(precisions):.3f}")
        if recalls:
            lines.append(f"  Retrieval Recall:     mean={sum(recalls)/len(recalls):.3f}  "
                        f"min={min(recalls):.3f}  max={max(recalls):.3f}")

        # Citation correctness
        cit_correctness = [
            r.citation_correctness for r in results if r.citation_correctness is not None
        ]
        if cit_correctness:
            mean = sum(cit_correctness) / len(cit_correctness)
            lines.append(f"  Citation Correctness: mean={mean:.3f}")

        # Faithfulness
        faithful = [r.answer_faithful for r in results if r.answer_faithful is not None]
        if faithful:
            lines.append(f"  Answer Faithfulness:  {_rate(faithful)}")

        # Abstention accuracy
        abstentions = [r.abstention_correct for r in results if r.abstention_correct is not None]
        if abstentions:
            lines.append(f"  Abstention Accuracy:  {_rate(abstentions)}")

        # Conflict handling
        conflicts = [r.conflict_handled for r in results if r.conflict_handled is not None]
        if conflicts:
            lines.append(f"  Conflict Handling:    {_rate(conflicts)}")

        # Latency
        latencies = [r.total_latency_s for r in results if r.total_latency_s > 0]
        if latencies:
            lines.append(f"  Response Latency:     mean={sum(latencies)/len(latencies):.3f}s  "
                        f"min={min(latencies):.3f}s  max={max(latencies):.3f}s")

        lines.append("")
        return "\n".join(lines)

    def save_results(self, results: list[EvaluationResult], label: str = "run") -> Path:
        """Save results to a timestamped JSON file."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{label}_{timestamp}.json"
        path = self.results_dir / filename

        data = {
            "label": label,
            "timestamp": timestamp,
            "run_timestamp": results[0].run_timestamp if results else "",
            "question_count": len(results),
            "results": [r.to_dict() for r in results],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Results saved to %s", path)
        return path
