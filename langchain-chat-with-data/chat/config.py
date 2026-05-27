"""AppConfig — all tunable parameters for a chat app instance."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class AppConfig:
    # ── Embedding ─────────────────────────────────────────────────────────────
    embed_model_name: str = "nomic-ai/nomic-embed-text-v1.5"

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_model_id: str  = "us.anthropic.claude-sonnet-4-6"
    aws_region: str    = field(default_factory=lambda: os.getenv("AWS_REGION", "us-west-1"))

    # ── Vector store ──────────────────────────────────────────────────────────
    # Absolute paths are resolved in the runner files using __file__
    persist_dir: str     = "nomic_chroma_db"
    source_docs_dir: str = "docs"

    # ── Chunking ──────────────────────────────────────────────────────────────
    # chunk_size is measured in tokens (tiktoken cl100k_base BPE).
    # Must not exceed the embedding model's maximum sequence length:
    #   nomic-embed-text-v1.5 → 8192 tokens  → 512 is a safe default
    #   all-MiniLM-L6-v2      →  256 tokens  → use 256
    chunk_size: int        = 512
    chunk_overlap: int     = 64          # ~12.5 % preserves cross-boundary context
    tiktoken_encoding: str = "cl100k_base"  # BPE encoding; no OpenAI call is made

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retriever_k: int       = 5
    retriever_fetch_k: int = 4 * retriever_k   # candidate pool before MMR re-ranking (4 × k)
    mmr_lambda: float      = 0.7   # 1.0 = pure similarity, 0.0 = pure diversity

    # ── UI ────────────────────────────────────────────────────────────────────
    port: int = 7860
