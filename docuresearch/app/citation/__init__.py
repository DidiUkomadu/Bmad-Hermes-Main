"""Backend citation resolution: passage ID → stored passage text and metadata."""

from app.citation.resolver import ResolvedPassage, resolve_passage

__all__ = ["ResolvedPassage", "resolve_passage"]
