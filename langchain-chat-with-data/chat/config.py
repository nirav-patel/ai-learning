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
    # source_docs_dir: directory scanned for PDFs on first run
    # weaviate_index_name: collection name inside Weaviate (PascalCase required)
    # weaviate_persist_dir: on-disk path for embedded Weaviate persistence
    source_docs_dir: str      = "docs"
    weaviate_index_name: str  = "LangchainDocs"
    weaviate_persist_dir: str = field(
        default_factory=lambda: os.path.join(os.path.dirname(__file__), "weaviate_data")
    )

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
    # hybrid_alpha: blend between vector and keyword search
    #   1.0 = pure semantic (vector),  0.0 = pure keyword (BM25),  0.5 = balanced
    hybrid_alpha: float    = 0.75  # slightly favour semantic over keyword

    # ── UI ────────────────────────────────────────────────────────────────────
    port: int = 7860
