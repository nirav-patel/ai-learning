"""Embedding model factory.

nomic-embed-text-v1.5 requires:
  - trust_remote_code=True     (custom BERT-2048 architecture)
  - batch_size=8               (nomic pads batches to max seq len; default 32
                                causes ~16 GiB attention buffers on large PDFs)
  - task prefixes              ("search_document: " for indexing,
                                "search_query: "    for query-time lookup)

all-MiniLM-L6-v2 needs no special configuration (no prefixes, no custom code).
"""
from __future__ import annotations

from .config import AppConfig


def make_embeddings(config: AppConfig):
    """Return an embeddings object appropriate for the configured model."""
    from langchain_huggingface import HuggingFaceEmbeddings

    if "nomic" in config.embed_model_name.lower():

        class NomicEmbeddings(HuggingFaceEmbeddings):
            """HuggingFaceEmbeddings with nomic-embed-text-v1.5 task prefixes.

            The model was trained with mandatory task-specific prefixes:
              - "search_document: " when embedding corpus chunks for the vector store
              - "search_query: "    when embedding the user question at retrieval time
            Using the wrong or no prefix measurably degrades retrieval quality.
            """

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return super().embed_documents(
                    ["search_document: " + t for t in texts]
                )

            def embed_query(self, text: str) -> list[float]:
                return super().embed_query("search_query: " + text)

        return NomicEmbeddings(
            model_name=config.embed_model_name,
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"batch_size": 8},
        )

    # Plain HuggingFace model (e.g. all-MiniLM-L6-v2) — no special handling needed
    return HuggingFaceEmbeddings(model_name=config.embed_model_name)
