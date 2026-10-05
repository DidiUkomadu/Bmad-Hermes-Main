"""Shared test configuration.

Tests never use the network. Hugging Face is put in offline mode before any
model library is imported, so a cached embedding model loads locally and an
uncached one fails fast instead of hanging on DNS or downloads.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture(scope="session")
def real_embedding_model():
    """The real sentence-transformers model, loaded once from the local cache.

    Skips (rather than downloading) when the model has never been downloaded:
    run the app once, or `python -c "from sentence_transformers import
    SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`, to cache it.
    """
    from app.retrieval.semantic import SentenceTransformerEmbeddingModel

    model = SentenceTransformerEmbeddingModel()
    try:
        _ = model.dimension
    except Exception as exc:  # noqa: BLE001 - any load failure means "not available"
        pytest.skip(f"embedding model not cached locally ({type(exc).__name__})")
    return model
