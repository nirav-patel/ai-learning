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


# ── AWS / Bedrock ─────────────────────────────────────────────────────────────

AWS_REGION: str = _optional("AWS_REGION", "us-east-1")
"""AWS region where Bedrock models are enabled (e.g. us-east-1, eu-west-1)."""

AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
"""AWS access key — optional if using an IAM role or AWS profile."""

AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
"""AWS secret key — optional if using an IAM role or AWS profile."""

AWS_PROFILE: str | None = os.getenv("AWS_PROFILE")
"""Optional AWS named profile (from ~/.aws/credentials). Used when explicit keys are absent."""

# ── LLM ───────────────────────────────────────────────────────────────────────

AGENT_MODEL: str = _optional(
    "AGENT_MODEL", "us.anthropic.claude-sonnet-4-6"
)
"""Bedrock model ID used by all text-based agents (bare ID, no ``aws:`` prefix).

Example: ``us.anthropic.claude-sonnet-4-6`` or ``anthropic.claude-3-haiku-20240307-v1:0``.
"""

COPYWRITER_MODEL: str = _optional(
    "COPYWRITER_MODEL", "us.anthropic.claude-sonnet-4-6"
)
"""Bedrock model ID used by the CopywriterAgent for multimodal (vision) calls.

Uses the bare Bedrock model ID (without the ``aws:`` prefix) because the Anthropic
SDK for Bedrock accepts the ID directly.
"""

# ── Image generation ──────────────────────────────────────────────────────────

IMAGE_MODEL: str = _optional("IMAGE_MODEL", "amazon.titan-image-generator-v2:0")
"""Bedrock image model ID used by the Graphic Designer agent (Amazon Titan)."""

IMAGE_SIZE: str = _optional("IMAGE_SIZE", "1024x1024")
"""Dimensions for generated images in ``WxH`` format (e.g. ``512x512``, ``1024x1024``)."""

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
