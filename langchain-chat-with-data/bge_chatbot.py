"""bge_app.py — RAG chat UI backed by BAAI/bge-base-en-v1.5 (port 7861).

Run:
    cd langchain-chat-with-data
    uv run python bge_chatbot.py
"""
import os

from chat.config import AppConfig
from chat.ui import run_app


config = AppConfig(
    embed_model_name="BAAI/bge-base-en-v1.5",
    weaviate_persist_dir=os.path.join(os.path.dirname(__file__), "bge_weaviate_db"),
    weaviate_index_name="BGELangchainDocs",
    source_docs_dir=os.path.join(os.path.dirname(__file__), "docs"),
    chunk_size=512,
    chunk_overlap=64,
    port=7070,
)


if __name__ == "__main__":
    run_app(config)