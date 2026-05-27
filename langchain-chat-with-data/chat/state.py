"""AppState — mutable runtime state for one chat app instance."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AppState:
    """Holds the live chain, retriever, LLM, corpus label, and per-session history.

    A single AppState instance is created per running app (nomic / minilm) so
    the two apps remain fully isolated even when running simultaneously.
    """

    chain: Callable | None     = None   # (question, session_id) -> generator[str]
    retriever: object | None   = None   # surfaced separately to show source docs
    llm: object | None         = None   # reused across chain rebuilds
    corpus: str                = "None loaded"
    _sessions: dict            = field(default_factory=dict)  # session_id -> list

    def get_session_history(self, session_id: str) -> list:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        return self._sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        self._sessions[session_id] = []
