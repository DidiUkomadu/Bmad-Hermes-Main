"""Conversation package — Story 5.1 (history storage); Story 5.2 builds on it."""

from app.conversation.session import (
    DEFAULT_MAX_TURNS,
    Session,
    SessionNotFoundError,
    add_turn,
    create_session,
    delete_session,
    get_history,
    get_session,
    prune_history,
    set_document_scope,
)
from app.generation.prompt import ConversationTurn

__all__ = [
    "DEFAULT_MAX_TURNS",
    "ConversationTurn",
    "Session",
    "SessionNotFoundError",
    "add_turn",
    "create_session",
    "delete_session",
    "get_history",
    "get_session",
    "prune_history",
    "set_document_scope",
]
