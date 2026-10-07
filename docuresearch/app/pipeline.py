"""End-to-end research pipeline: ingest → retrieve → generate → verify.

Composes the independently tested components into the single flow the API
and the evaluation runner both use:

    query
      → hybrid retrieval over the content store (scoped, ranked)
      → RetrievedContext
      → prompt → LLM → parse
      → resolve_citations          (drop citations not in retrieved context)
      → enforce_citation_grounding (withhold claims with no valid citation)
      → finalize_generation        (abstention / evidence-quality rules)
      → GeneratedAnswer

Retrieval indexes are rebuilt from the store on every query. Embeddings
are persisted at ingest time, so a rebuild only re-tokenizes for BM25 and
embeds the query; this keeps results consistent with deletions without
any cache invalidation, and is fast enough for the MVP corpus size.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from app.chunking.strategy import ChunkingConfig
from app.generation.citation import resolve_citations
from app.generation.interface import (
    EvidenceQuality,
    GeneratedAnswer,
    GenerationConfig,
    LLMInterface,
)
from app.generation.post_process import enforce_citation_grounding, finalize_generation
from app.generation.prompt import (
    ConversationTurn,
    RetrievedContext,
    RetrievedPassage,
    build_prompt,
)
from app.generation.schema import parse_llm_output
from app.ingestion.orchestrator import ingest_document, ingest_document_bytes
from app.retrieval.hybrid import HybridRetriever, ScopedHybridRetriever
from app.retrieval.interface import DocumentScope, EmbeddingModel, HybridWeighting
from app.retrieval.keyword import BM25RetrieverImpl
from app.retrieval.ranking import RankingLayer
from app.retrieval.semantic import SemanticRetrieverImpl, SentenceTransformerEmbeddingModel
from app.store.schema import (
    create_schema,
    get_connection,
    list_documents,
    list_passages_for_document,
    set_db_path,
)

logger = logging.getLogger(__name__)

NO_PASSAGES_ANSWER_TEXT = (
    "The documents do not contain enough information to answer this: "
    "no passages were retrieved for the question."
)


class GenerationError(RuntimeError):
    """The LLM call failed or returned output that could not be parsed.

    Raised instead of fabricating an answer; callers decide how to surface it.
    """


@dataclass(frozen=True)
class PipelineResult:
    """A final answer together with the retrieval context it was grounded in."""

    answer: GeneratedAnswer
    context: RetrievedContext


class ResearchPipeline:
    """The DocuResearch question-answering pipeline over a SQLite store.

    Args:
        db_path: Path to the SQLite content store.
        llm: Any LLMInterface implementation.
        embedding_model: Used at ingest time and for query embedding. Defaults
            to the sentence-transformers model (loaded lazily on first use).
        weighting: Semantic/keyword blend for hybrid retrieval.
        max_candidates: Candidate set size passed to generation.
        generation_config: Prompt construction options.
        chunking_config: Chunking options used at ingest time.
    """

    def __init__(
        self,
        db_path: str,
        llm: LLMInterface,
        embedding_model: EmbeddingModel | None = None,
        weighting: HybridWeighting | None = None,
        max_candidates: int = 20,
        generation_config: GenerationConfig | None = None,
        chunking_config: ChunkingConfig | None = None,
    ) -> None:
        self._db_path = db_path
        self._llm = llm
        self._embedding_model = embedding_model or SentenceTransformerEmbeddingModel()
        self._weighting = weighting or HybridWeighting()
        self._ranking = RankingLayer(max_candidates)
        # By default the prompt shows every retrieved candidate: a passage that
        # counts as "retrieved" but is cut from the prompt can't be used, and
        # the model wrongly concludes the documents lack the answer.
        self._generation_config = generation_config or GenerationConfig(
            max_passages_in_prompt=max_candidates
        )
        self._chunking_config = chunking_config

        set_db_path(db_path)
        conn = get_connection()
        create_schema(conn)
        conn.close()

    @property
    def db_path(self) -> str:
        """Path to the SQLite content store this pipeline reads and writes."""
        return self._db_path

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, source: str | Path, owner_id: str | None = None) -> dict[str, Any]:
        """Ingest a document from disk, embedding its passages."""
        return ingest_document(
            source,
            db_path=self._db_path,
            chunking_config=self._chunking_config,
            embedding_model=self._embedding_model,
            owner_id=owner_id,
        )

    def ingest_bytes(
        self, data: bytes, filename: str, owner_id: str | None = None
    ) -> dict[str, Any]:
        """Ingest an uploaded document as *owner_id*'s, embedding its passages."""
        return ingest_document_bytes(
            data,
            filename,
            db_path=self._db_path,
            chunking_config=self._chunking_config,
            embedding_model=self._embedding_model,
            owner_id=owner_id,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        document_id: str | None = None,
        conversation_history: list[ConversationTurn] | None = None,
        retrieval_query: str | None = None,
        owner_id: str | None = None,
        before_model_call: Callable[[], None] | None = None,
    ) -> PipelineResult:
        """Answer *query*, optionally restricted to one document.

        *retrieval_query*, when given, is searched instead of *query* (e.g. a
        follow-up expanded with conversation context); the prompt always
        carries the user's *query* verbatim. *owner_id* restricts the search
        to that user's documents (see ``retrieve``). *before_model_call* is
        passed to ``generate``.

        Raises:
            GenerationError: if generation fails (see ``generate``).
            Anything *before_model_call* raises (e.g. ``QuestionLimitReached``)
            propagates unchanged; the model is not called.
        """
        context = self.retrieve(
            query, document_id, conversation_history, retrieval_query, owner_id=owner_id
        )
        answer = self.generate(context, before_model_call=before_model_call)
        return PipelineResult(answer=answer, context=context)

    def retrieve(
        self,
        query: str,
        document_id: str | None = None,
        conversation_history: list[ConversationTurn] | None = None,
        retrieval_query: str | None = None,
        owner_id: str | None = None,
    ) -> RetrievedContext:
        """Run scoped hybrid retrieval and package the result for generation.

        Only the documents of *owner_id* are loaded into the search index, so
        another user's passages can never be retrieved, cited, or shown to the
        model (Epic 8 isolation). ``None`` searches every document and is for
        trusted internal callers (the evaluation runner), never the API.
        """
        start = time.monotonic()
        set_db_path(self._db_path)
        conn = get_connection()
        try:
            doc_names = {d.id: d.name for d in list_documents(conn, owner_id=owner_id)}
            passages = {
                p.id: p
                for doc_id in doc_names
                for p in list_passages_for_document(conn, doc_id)
            }
        finally:
            conn.close()

        missing = sum(1 for p in passages.values() if p.embedding is None)
        if missing:
            logger.warning(
                "%d of %d passages have no embedding and are keyword-searchable only",
                missing, len(passages),
            )

        if document_id:
            scope = DocumentScope(mode="specific", document_id=document_id)
        else:
            scope = DocumentScope(mode="all")

        retriever = ScopedHybridRetriever(
            HybridRetriever(
                SemanticRetrieverImpl(self._embedding_model),
                BM25RetrieverImpl(),
                self._weighting,
            )
        )
        retriever.index_passages(
            [
                {
                    "id": p.id,
                    "text": p.text,
                    "embedding": p.embedding,
                    "document_id": p.document_id,
                }
                for p in passages.values()
            ],
            scope,
        )
        results = retriever.search(retrieval_query or query, top_n=self._ranking.max_candidates)
        ranked = self._ranking.rank(results, latency_seconds=time.monotonic() - start)

        return RetrievedContext(
            query=query,
            scope={"mode": scope.mode, "document_id": scope.document_id},
            passages=[
                RetrievedPassage(
                    passage_id=r.passage_id,
                    document_name=doc_names[passages[r.passage_id].document_id],
                    location=passages[r.passage_id].location,
                    text=passages[r.passage_id].text,
                    score=r.score,
                )
                for r in ranked.results
            ],
            conversation_history=conversation_history,
            retrieval_latency_seconds=ranked.retrieval_latency_seconds,
        )

    def generate(
        self,
        context: RetrievedContext,
        before_model_call: Callable[[], None] | None = None,
    ) -> GeneratedAnswer:
        """Generate and verify an answer for an already-retrieved context.

        *before_model_call*, when given, is called exactly once, right before
        the model is called; it is not called when no passages were retrieved
        (no model call is made then). It may raise to stop the call, e.g. when
        a daily question limit is reached (Story 9.1); the exception
        propagates unchanged.

        Raises:
            GenerationError: if the LLM call fails or its output cannot be
                parsed. No answer is fabricated in that case.
        """
        if not context.passages:
            # Nothing to ground an answer in — abstain without calling the LLM.
            return GeneratedAnswer(
                answer_text=NO_PASSAGES_ANSWER_TEXT,
                citations=[],
                evidence_quality=EvidenceQuality.INSUFFICIENT,
                evidence_quality_narrative="No passages were retrieved for this query.",
                is_abstention=True,
                generation_latency_seconds=0.0,
            )

        if before_model_call is not None:
            before_model_call()

        prompt = build_prompt(context, self._generation_config)
        start = time.monotonic()
        response = self._llm.generate(prompt)
        latency = time.monotonic() - start

        parsed = response.structured
        if parsed is None and response.raw_text:
            parsed = parse_llm_output(response.raw_text)
        if parsed is None:
            raise GenerationError(response.parse_error or "LLM returned no parsable answer")

        answer = replace(parsed, generation_latency_seconds=latency)
        answer = resolve_citations(answer, context)
        answer = enforce_citation_grounding(answer)
        return finalize_generation(answer, context)
