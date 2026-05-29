"""chatbot/eval/prod_tracing.py — TruLens live PROD tracing for the RAG pipeline.

When enabled (TRULENS_PROD_ENABLED=true), every chat turn is automatically
recorded to the TruLens SQLite DB. The dashboard at http://localhost:8501
then shows traces over real user traffic.

How to enable
─────────────
  1. Add to .env:  TRULENS_PROD_ENABLED=true
  2. Run the chatbot normally:  python main.py
  3. View results:  python -m chatbot.eval.run_evaluation --dashboard-only
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

_tru_chain = None
_tru_session = None


def is_enabled() -> bool:
    """Return True if TRULENS_PROD_ENABLED=true in the environment."""
    return os.getenv("TRULENS_PROD_ENABLED", "false").lower() == "true"


def wrap_chain(lcel_chain: "Runnable", config) -> "Runnable":
    """Wrap *lcel_chain* with TruChain for live PROD tracing.

    Returns the original chain unchanged when TRULENS_PROD_ENABLED is false.
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

        _tru_chain = TruChain(
            lcel_chain,
            app_name="RAGChatbot-PROD",
            app_version=app_version,
        )

        logger.info("TruLens PROD tracing active — version=%s", app_version)

    except ImportError:
        logger.warning(
            "TRULENS_PROD_ENABLED=true but trulens not installed. "
            "Run: pip install -e '.[eval]'"
        )

    return lcel_chain


def get_tru_chain():
    """Return the TruChain wrapper (None if tracing is not active)."""
    return _tru_chain
