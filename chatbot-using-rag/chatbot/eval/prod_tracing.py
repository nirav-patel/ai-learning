"""chatbot/eval/prod_tracing.py — TruLens live PROD tracing for the RAG pipeline.

When enabled (TRULENS_PROD_ENABLED=true), every chat turn is automatically
recorded to the TruLens SQLite DB with full RAG Triad scoring.  The dashboard
at http://localhost:8501 then shows live quality metrics over real user traffic.

Architecture
────────────
  RAGPipeline.lcel_chain  →  wrapped with TruChain
  pipeline.stream()       →  calls lcel_chain.stream() inside a TruChain context

The TruChain wrapper is transparent to the Gradio UI — it intercepts the same
LCEL Runnable the chatbot already uses and records every invocation as a span.

How to enable
─────────────
  1. Add to .env:
       TRULENS_PROD_ENABLED=true

  2. Optionally override the DB path (defaults to chatbot/eval/trulens.db):
       TRULENS_DB_URL=sqlite:////absolute/path/to/prod.db

  3. Run the chatbot normally:
       python main.py

  4. View results in the browser (no re-run needed):
       python -m chatbot.eval.run_evaluation --dashboard-only
       → http://localhost:8501

Note on feedback scoring in PROD
─────────────────────────────────
  By default, PROD tracing records spans and inputs/outputs but does NOT run
  feedback scoring on every turn (too expensive for real-time use).  Set
  TRULENS_PROD_FEEDBACK=true to enable async Bedrock feedback scoring per turn.
  This adds ~5–10s latency per turn for the background scorer.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

_tru_chain = None   # module-level singleton; initialised once
_tru_session = None


def is_enabled() -> bool:
    """Return True if TRULENS_PROD_ENABLED=true in the environment."""
    return os.getenv("TRULENS_PROD_ENABLED", "false").lower() == "true"


def _feedback_enabled() -> bool:
    return os.getenv("TRULENS_PROD_FEEDBACK", "false").lower() == "true"


def wrap_chain(lcel_chain: "Runnable", config) -> "Runnable":
    """Wrap *lcel_chain* with TruChain for live PROD tracing.

    Returns the original chain unchanged when TRULENS_PROD_ENABLED is false,
    so the chatbot works identically with tracing off.

    When enabled:
      - Every invocation is recorded to the TruLens SQLite DB.
      - If TRULENS_PROD_FEEDBACK=true, RAG Triad metrics are scored async via
        Bedrock after each turn (adds background latency).
      - The chain itself is returned unchanged — TruChain is a transparent wrapper.

    Args:
        lcel_chain: The RAGPipeline.lcel_chain Runnable to wrap.
        config:     AppConfig (used for app_version label and Bedrock creds).

    Returns:
        The original lcel_chain (TruChain instruments it in-place).
    """
    global _tru_chain, _tru_session

    if not is_enabled():
        return lcel_chain

    try:
        import nest_asyncio
        nest_asyncio.apply()

        from trulens.apps.langchain import TruChain
        from trulens.core import TruSession

        db_path = Path(__file__).parent / "trulens.db"
        db_url = os.getenv("TRULENS_DB_URL", f"sqlite:///{db_path}")

        if _tru_session is None:
            _tru_session = TruSession(database_url=db_url)
            logger.info("TruLens PROD tracing enabled — DB: %s", db_url)

        app_version = os.getenv("APP_VERSION", "prod")

        feedbacks = []
        if _feedback_enabled():
            try:
                from chatbot.eval.run_evaluation import _build_feedbacks
                feedbacks = _build_feedbacks(config, lcel_chain)
                logger.info("TruLens PROD feedback scoring enabled (async Bedrock judge)")
            except Exception as exc:
                logger.warning("Could not build TruLens feedbacks: %s — tracing without scoring", exc)

        _tru_chain = TruChain(
            lcel_chain,
            app_name="RAGChatbot-PROD",
            app_version=app_version,
            feedbacks=feedbacks,
        )

        logger.info(
            "TruLens PROD tracing active — app_name=RAGChatbot-PROD version=%s feedback=%s",
            app_version,
            _feedback_enabled(),
        )

    except ImportError:
        logger.warning(
            "TRULENS_PROD_ENABLED=true but trulens packages not installed. "
            "Run: pip install -e '.[eval]'"
        )

    return lcel_chain


def get_tru_chain():
    """Return the TruChain wrapper (None if tracing is not active)."""
    return _tru_chain
