"""nomic_app.py — RAG chat UI backed by nomic-embed-text-v1.5 (port 7860).

Run:
    cd langchain-chat-with-data
    uv run python nomic_app.py
"""
import os

from chat.config    import AppConfig
from chat.ui import run_app

config = AppConfig(
    embed_model_name  = "nomic-ai/nomic-embed-text-v1.5",
    persist_dir       = os.path.join(os.path.dirname(__file__), "nomic_chroma_db"),
    source_docs_dir   = os.path.join(os.path.dirname(__file__), "docs"),
    chunk_size        = 512,    # nomic supports up to 8192 tokens — 512 is practical
    chunk_overlap     = 64,
    port              = 6060,
)

if __name__ == "__main__":
    run_app(config)
