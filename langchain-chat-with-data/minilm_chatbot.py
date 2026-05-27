"""minilm_app.py — RAG chat UI backed by all-MiniLM-L6-v2 (port 7861).

NOTE: all-MiniLM-L6-v2 has a hard maximum of 256 tokens per sequence.
      chunk_size is set to 256 to match this limit exactly.
      Using a larger chunk_size would silently truncate chunks at embedding
      time, degrading retrieval quality for documents with long paragraphs.

Run:
    cd langchain-chat-with-data
    uv run python minilm_app.py
"""
import os

from chat.config    import AppConfig
from chat.ui import run_app

config = AppConfig(
    embed_model_name  = "all-MiniLM-L6-v2",
    persist_dir       = os.path.join(os.path.dirname(__file__), "sample_chroma_db"),
    source_docs_dir   = os.path.join(os.path.dirname(__file__), "docs"),
    chunk_size        = 256,    # hard max for all-MiniLM-L6-v2
    chunk_overlap     = 32,
    port              = 7070,
)

if __name__ == "__main__":
    run_app(config)
