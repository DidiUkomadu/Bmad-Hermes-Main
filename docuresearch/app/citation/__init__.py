"""Backend citation resolution: passage ID → stored passage text and metadata."""

from app.citation.resolver import ResolvedPassage, format_location, resolve_passage

__all__ = ["ResolvedPassage", "format_location", "resolve_passage"]
