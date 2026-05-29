"""infrastructure/observability.py — Phoenix tracing for the RAG pipeline.

Calling setup_observability(config) once at app startup auto-instruments every
LangChain component (LLM calls, retriever calls, chain steps) without any
manual span creation.

Phoenix runs an embedded local server — no Docker, no external account needed.
Dashboard: http://localhost:<phoenix_port>

Set PHOENIX_ENABLED=false in .env to disable tracing (e.g. in CI).

Phoenix 16.x setup (3-step):
  1. px.launch_app()              — start the Phoenix UI server
  2. phoenix.otel.register()      — wire up the OpenTelemetry tracer provider
  3. LangChainInstrumentor()      — auto-instrument all LangChain components
"""
from __future__ import annotations

import io
import logging
import sys
from contextlib import redirect_stdout

logger = logging.getLogger(__name__)

_PROJECT_NAME = "rag-chatbot"


def setup_observability(config: AppConfig) -> None:
    """Start Phoenix and instrument LangChain. No-op if phoenix_enabled=False."""
    if not config.phoenix_enabled:
        logger.info("Phoenix observability disabled (PHOENIX_ENABLED=false).")
        return

    try:
        import phoenix as px
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor
    except ImportError as exc:
        logger.warning(
            "Phoenix not installed — observability skipped. "
            "Run: pip install arize-phoenix openinference-instrumentation-langchain\n"
            "Error: %s", exc,
        )
        return

    try:
        # Suppress Phoenix's startup print() noise (URLs, deprecation banners).
        with redirect_stdout(io.StringIO()):
            # Step 1: start the Phoenix UI server
            px.launch_app(port=config.phoenix_port)

            # Step 2: register the OTel tracer provider pointing at Phoenix's OTLP endpoint
            tracer_provider = register(
                project_name=_PROJECT_NAME,
                endpoint=f"http://localhost:{config.phoenix_port}/v1/traces",
                verbose=False,
            )

            # Step 3: instrument LangChain with the wired-up tracer provider
            LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

        logger.info(
            "Phoenix tracing active — project '%s' → http://localhost:%d",
            _PROJECT_NAME,
            config.phoenix_port,
        )
    except Exception as exc:
        logger.warning("Phoenix failed to start — observability skipped: %s", exc)
