"""FastAPI application — Story 7.6.

Run locally (from the docuresearch/ directory):
    uvicorn app.main:app --reload

then open http://localhost:8000 for the UI (Epic 6) or /docs for the API.

Every endpoint except health, sign-in/registration, and the UI itself requires
a signed-in user, and each user only ever sees their own documents and
conversations (Epic 8).

The app starts without an LLM configured: document upload, listing, removal,
and citation lookup work; question endpoints return 503 until
DOCURESEARCH_LLM_BASE_URL and DOCURESEARCH_LLM_MODEL are set.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import auth, citations, documents, query
from app.api.deps import AppState
from app.api.errors import register_error_handlers
from app.config import Settings
from app.generation.interface import LLMResponse
from app.generation.openai_compat import OpenAICompatibleLLM
from app.pipeline import ResearchPipeline
from app.retrieval.interface import HybridWeighting
from app.retrieval.semantic import SentenceTransformerEmbeddingModel

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"
STATIC_DIR = Path(__file__).resolve().parent / "static"

_LLM_NOT_CONFIGURED = (
    "Set DOCURESEARCH_LLM_BASE_URL and DOCURESEARCH_LLM_MODEL (and DOCURESEARCH_LLM_API_KEY "
    "if your provider needs one), or the [llm] section of config.toml, then restart."
)


class _UnconfiguredLLM:
    """Placeholder backend; query endpoints refuse with 503 before reaching it."""

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(raw_text="", parse_error=_LLM_NOT_CONFIGURED)


def _build_pipeline(settings: Settings) -> tuple[ResearchPipeline, OpenAICompatibleLLM | None]:
    llm = None
    if settings.llm.configured:
        llm = OpenAICompatibleLLM(
            base_url=settings.llm.base_url,
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            timeout_seconds=settings.llm.timeout_seconds,
        )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline = ResearchPipeline(
        str(settings.db_path),
        llm or _UnconfiguredLLM(),
        embedding_model=SentenceTransformerEmbeddingModel(settings.embedding_model),
        weighting=HybridWeighting(
            semantic_weight=settings.semantic_weight,
            keyword_weight=settings.keyword_weight,
        ),
        max_candidates=settings.max_candidates,
    )
    return pipeline, llm


def create_app(
    settings: Settings | None = None,
    pipeline: ResearchPipeline | None = None,
) -> FastAPI:
    """Build the application.

    Args:
        settings: Runtime settings; loaded from config/env when None.
        pipeline: Pre-built pipeline (tests inject one with fake LLM and
            embeddings). When None, one is built from *settings* at startup.
    """
    settings = settings or Settings.load()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned_llm = None
        if pipeline is None:
            built, owned_llm = _build_pipeline(settings)
            reason = None if owned_llm else _LLM_NOT_CONFIGURED
            if reason:
                logger.warning("LLM backend not configured; question endpoints disabled.")
            app.state.docuresearch = AppState(settings, built, reason)
        else:
            app.state.docuresearch = AppState(settings, pipeline)
        try:
            yield
        finally:
            if owned_llm is not None:
                owned_llm.close()

    app = FastAPI(
        title="DocuResearch",
        description="Answers questions about your technical documents, with citations.",
        version="0.1.0",
        lifespan=lifespan,
    )
    register_error_handlers(app)
    for module in (auth, documents, query, citations):
        app.include_router(module.router, prefix=API_PREFIX)

    @app.get(f"{API_PREFIX}/health", tags=["health"])
    def health() -> dict[str, object]:
        state: AppState = app.state.docuresearch
        return {
            "status": "ok",
            "llm_configured": state.llm_unavailable_reason is None,
            "registration_open": state.settings.allow_registration,
        }

    # Citation UI (Epic 6): plain HTML + JS, no build step (implementation plan §14.8).
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
