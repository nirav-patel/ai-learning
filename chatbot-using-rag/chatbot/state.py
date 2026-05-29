"""state.py — AppState: mutable runtime state for one chatbot instance.

Holds the active RAGPipeline, retriever, LLM, WeaviateStore, corpus label,
and per-session chat history.

One AppState is created per running app so multiple instances remain isolated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .storage import WeaviateStore


@dataclass
class AppState:
    chain: "RAGPipeline | None"    = None   # type: ignore[name-defined]
    retriever: object | None       = None   # exposed separately for source-doc display
    llm: object | None             = None   # reused across chain rebuilds
    corpus: str                    = "None loaded"
    vector_store: WeaviateStore    = field(default_factory=WeaviateStore)
    _sessions: dict                = field(default_factory=dict)

    def get_session_history(self, session_id: str) -> list:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        return self._sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        self._sessions[session_id] = []

    def build_chain(self, retriever, config) -> None:
        """Wire a retriever into a new RAGPipeline, lazily creating the LLM.

        If TRULENS_PROD_ENABLED=true, the pipeline's lcel_chain is wrapped with
        TruChain so every chat turn is automatically recorded to the TruLens DB.

        Args:
            retriever: A configured LangChain retriever.
            config:    AppConfig used to create the LLM if not yet initialised.
        """
        from .providers.llm import make_llm
        from .pipeline import RAGPipeline

        if self.llm is None:
            self.llm = make_llm(config)

        self.retriever = retriever
        self.chain = RAGPipeline(retriever, self.llm, self.get_session_history)

        # Optionally activate TruLens live PROD tracing (TRULENS_PROD_ENABLED=true)
        try:
            from .eval.prod_tracing import is_enabled, wrap_chain
            if is_enabled():
                wrap_chain(self.chain.lcel_chain, config)
        except Exception:
            pass  # tracing is optional — never break the chatbot
