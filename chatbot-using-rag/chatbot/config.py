"""AppConfig — all tunable parameters for the RAG chatbot."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# Resolve project root (one level above the chatbot/ package)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class AppConfig:
    # ── LLM provider ──────────────────────────────────────────────────────────
    # Supported: "bedrock" | "openai" | "ollama"
    llm_provider: str  = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "bedrock"))
    llm_model_id: str  = "us.anthropic.claude-sonnet-4-6"
    aws_region: str    = field(default_factory=lambda: os.getenv("AWS_REGION", "us-west-2"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    # ── Embedding provider ────────────────────────────────────────────────────
    # Supported: "huggingface" | "openai" | "ollama"
    embed_provider: str   = field(default_factory=lambda: os.getenv("EMBED_PROVIDER", "huggingface"))
    embed_model_name: str = "nomic-ai/nomic-embed-text-v1.5"

    # ── Vector store ──────────────────────────────────────────────────────────
    # data_sources_dir: directory scanned for PDFs on first run
    # weaviate_index_name: collection name inside Weaviate (PascalCase required)
    # weaviate_persist_dir: on-disk path for embedded Weaviate data
    data_sources_dir: str    = field(
        default_factory=lambda: os.getenv("DATA_SOURCES_DIR", os.path.join(_PROJECT_ROOT, "data-sources"))
    )
    weaviate_index_name: str = "RagDocs"
    weaviate_persist_dir: str = field(
        default_factory=lambda: os.getenv("WEAVIATE_PERSIST_DIR", os.path.join(_PROJECT_ROOT, "vector-store"))
    )

    # ── Chunking ──────────────────────────────────────────────────────────────
    # chunk_size is in tokens (tiktoken cl100k_base BPE).
    # Max sequence lengths:
    #   nomic-embed-text-v1.5 → 8192 tokens  (512 is a safe practical default)
    #   all-MiniLM-L6-v2      →  256 tokens
    chunk_size: int        = 512
    chunk_overlap: int     = 64
    tiktoken_encoding: str = "cl100k_base"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retriever_k: int   = 5
    # hybrid_alpha: 1.0 = pure vector | 0.0 = pure BM25 | 0.5 = balanced
    hybrid_alpha: float = 0.75

    # ── Observability ─────────────────────────────────────────────────────────
    phoenix_enabled: bool = field(
        default_factory=lambda: os.getenv("PHOENIX_ENABLED", "true").lower() == "true"
    )
    phoenix_port: int = field(
        default_factory=lambda: int(os.getenv("PHOENIX_PORT", "6006"))
    )

    # ── UI ────────────────────────────────────────────────────────────────────
    port: int = 7860
