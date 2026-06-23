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


def _optional(key: str, default: str) -> str:
    # os.getenv returns "" when a key is set to empty in .env; fall back to default.
    return os.getenv(key) or default


# ── AWS / Bedrock ─────────────────────────────────────────────────────────────

AWS_REGION: str = _optional("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_PROFILE: str | None = os.getenv("AWS_PROFILE")

# ── LLM ───────────────────────────────────────────────────────────────────────

AGENT_MODEL: str = _optional("AGENT_MODEL", "us.anthropic.claude-sonnet-4-6")
TRIAGE_MODEL: str = _optional("TRIAGE_MODEL", AGENT_MODEL)
EMBED_MODEL: str = _optional("EMBED_MODEL", "amazon.titan-embed-text-v2:0")

# ── Memory ────────────────────────────────────────────────────────────────────

MEMORY_DB_DIR: Path = Path(_optional("MEMORY_DB_DIR", "./memory_db"))
USER_ID: str = _optional("USER_ID", "john")

# ── Agent loop ────────────────────────────────────────────────────────────────

MAX_AGENT_ITERATIONS: int = int(_optional("MAX_AGENT_ITERATIONS", "10"))
