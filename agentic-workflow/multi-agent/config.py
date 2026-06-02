"""
Central configuration loaded from environment variables.

All settings can be overridden via a .env file in the project root.
See .env.example for the full reference.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """Return env var value or raise a clear error at startup."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "See .env.example for setup instructions."
        )
    return value


def _optional(key: str, default: str) -> str:
    return os.getenv(key, default)


# ── LLM ───────────────────────────────────────────────────────────────────────

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
"""OpenAI API key — required at runtime, validated lazily so tests can run."""

AGENT_MODEL: str = _optional("AGENT_MODEL", "openai:gpt-4o-mini")
"""aisuite model string used by all text-based agents."""

IMAGE_MODEL: str = _optional("IMAGE_MODEL", "dall-e-3")
"""OpenAI image model used by the Graphic Designer agent."""

IMAGE_SIZE: str = _optional("IMAGE_SIZE", "1024x1024")
"""Dimensions for generated images."""

IMAGE_QUALITY: str = _optional("IMAGE_QUALITY", "standard")
"""Quality level for image generation: standard | hd (dall-e-3 only)."""

# ── Web Search ─────────────────────────────────────────────────────────────────

TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
"""Tavily search API key — required for MarketResearchAgent at runtime."""

TAVILY_BASE_URL: str | None = os.getenv("DLAI_TAVILY_BASE_URL")
"""Optional custom Tavily base URL (e.g. DLAI sandbox)."""

TAVILY_MAX_RETRIES: int = int(_optional("TAVILY_MAX_RETRIES", "3"))
"""How many times to retry a failed Tavily search."""

TAVILY_RETRY_DELAY: float = float(_optional("TAVILY_RETRY_DELAY", "1.5"))
"""Initial delay (seconds) between Tavily retries — multiplied by attempt number."""

# ── Pipeline / Output ─────────────────────────────────────────────────────────

OUTPUT_DIR: Path = Path(_optional("OUTPUT_DIR", "./output"))
"""Directory where generated images and markdown reports are saved."""

# ── Agent Loop ────────────────────────────────────────────────────────────────

MAX_AGENT_ITERATIONS: int = int(_optional("MAX_AGENT_ITERATIONS", "10"))
"""Safety cap on the number of LLM + tool iterations per agent run."""
