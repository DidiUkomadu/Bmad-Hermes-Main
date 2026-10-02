"""Conversation package — Story 5.1 (history storage) and 5.2 (follow-up queries)."""

from app.conversation.followup import ConversationAnswer, ask
from app.conversation.session import (
    DEFAULT_MAX_TURNS,
    Session,
    SessionNotFoundError,
    SessionSummary,
    add_turn,
    create_session,
    delete_session,
    get_history,
    get_session,
    list_sessions,
    prune_history,
    set_document_scope,
)
from app.generation.prompt import ConversationTurn

__all__ = [
    "DEFAULT_MAX_TURNS",
    "ConversationAnswer",
    "ConversationTurn",
    "Session",
    "SessionNotFoundError",
    "SessionSummary",
    "add_turn",
    "ask",
    "create_session",
    "delete_session",
    "get_history",
    "get_session",
    "list_sessions",
    "prune_history",
    "set_document_scope",
]
