"""storage.py — WeaviateStore: manages the embedded Weaviate vector store.

Merges the former weaviate_manager.py (client lifecycle) and vectorstore.py
(retriever setup, chain initialisation) into a single cohesive class.

For production, replace ``connect_to_embedded()`` with
``connect_to_weaviate_cloud()`` or ``connect_to_local()``.
"""
from __future__ import annotations

import atexit
import logging
import os

import weaviate
from weaviate.exceptions import WeaviateStartUpError

from .config import AppConfig

logger = logging.getLogger(__name__)


class WeaviateStore:
    """Owns the full lifecycle of the embedded Weaviate vector store.

    Responsibilities:
      - Client connection (lazy, cached, cleaned up on exit)
      - Collection initialisation from a PDF directory (first run)
      - Fast-path loading when the collection already exists
      - Adding documents to an existing collection
      - Replacing the entire collection with new documents
      - Returning a configured hybrid (vector + BM25) retriever
    """

    def __init__(self) -> None:
        self._client: weaviate.WeaviateClient | None = None
        self._persist_dir: str | None = None
        atexit.register(self._close)

    # ── Public interface ──────────────────────────────────────────────────────

    def initialise(self, config: AppConfig, embeddings) -> object | None:
        """Bring the vector store to a ready state and return a retriever.

        Fast path — collection already exists → load and return retriever.
        First run — scan data_sources_dir for PDFs, embed, ingest, return retriever.

        Returns:
            A configured LangChain retriever, or None if no documents are available.
        """
        from langchain_weaviate import WeaviateVectorStore
        from .ingestion import DocumentLoader

        client = self._get_client(config)

        # Fast path: collection already has documents
        try:
            vectordb = WeaviateVectorStore(
                client=client,
                index_name=config.weaviate_index_name,
                text_key="text",
                embedding=embeddings,
            )
            if vectordb.similarity_search("test", k=1):
                count = self._chunk_count(client, config.weaviate_index_name)
                logger.info("Vector store ready — %d chunks in '%s'.", count, config.weaviate_index_name)
                return self._as_retriever(vectordb, config)
        except Exception as exc:
            logger.warning("Could not load existing collection — %s", exc)

        # First run: build from PDFs in data_sources_dir
        logger.info("No existing data. Scanning '%s' for PDFs …", config.data_sources_dir)
        loader = DocumentLoader(config)
        chunks, pdf_paths = loader.load_directory()

        if not chunks:
            logger.warning("No PDFs found — upload one via the UI to get started.")
            return None

        logger.info("Embedding %d chunks from %d PDF(s) …", len(chunks), len(pdf_paths))
        vectordb = WeaviateVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            index_name=config.weaviate_index_name,
        )
        logger.info("Ingested %d chunks into '%s'.", len(chunks), config.weaviate_index_name)
        return self._as_retriever(vectordb, config)

    def replace_documents(self, chunks: list, config: AppConfig, embeddings) -> object:
        """Replace the entire collection with new chunks and return a retriever."""
        from langchain_weaviate import WeaviateVectorStore

        client = self._get_client(config)
        try:
            client.collections.delete(config.weaviate_index_name)
        except Exception:
            pass

        vectordb = WeaviateVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            index_name=config.weaviate_index_name,
        )
        logger.info("Replaced collection '%s' with %d chunks.", config.weaviate_index_name, len(chunks))
        return self._as_retriever(vectordb, config)

    def add_documents(self, chunks: list, config: AppConfig, embeddings) -> tuple[object, int]:
        """Add chunks to the existing collection and return (retriever, total_count)."""
        from langchain_weaviate import WeaviateVectorStore

        client = self._get_client(config)
        vectordb = WeaviateVectorStore(
            client=client,
            index_name=config.weaviate_index_name,
            text_key="text",
            embedding=embeddings,
        )
        vectordb.add_documents(chunks)
        total = self._chunk_count(client, config.weaviate_index_name)
        logger.info("Added %d chunks. Total: %d.", len(chunks), total)
        return self._as_retriever(vectordb, config), total

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_client(self, config: AppConfig) -> weaviate.WeaviateClient:
        """Return (or lazily create) the embedded Weaviate client."""
        persist_dir = os.path.abspath(config.weaviate_persist_dir)
        if (
            self._client is None
            or not self._client.is_connected()
            or self._persist_dir != persist_dir
        ):
            if self._client is not None and self._client.is_connected():
                self._client.close()
            try:
                self._client = weaviate.connect_to_embedded(
                    persistence_data_path=persist_dir,
                    environment_variables={"LOG_LEVEL": "error", "DISABLE_TELEMETRY": "true"},
                )
            except WeaviateStartUpError as exc:
                if "already listening on ports" not in str(exc):
                    raise
                self._client = weaviate.connect_to_local(port=8079, grpc_port=50050)
            self._persist_dir = persist_dir
        return self._client

    @staticmethod
    def _as_retriever(vectordb, config: AppConfig):
        """Return a hybrid (vector + BM25) LangChain retriever.

        alpha: 1.0 = pure vector | 0.0 = pure BM25 | 0.75 = slightly favour semantic
        """
        return vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.retriever_k, "alpha": config.hybrid_alpha},
        )

    @staticmethod
    def _chunk_count(client: weaviate.WeaviateClient, index_name: str) -> int:
        try:
            return client.collections.get(index_name).aggregate.over_all().total_count
        except Exception:
            return 0

    def _close(self) -> None:
        if self._client is None:
            return
        try:
            if self._client.is_connected():
                self._client.close()
        except Exception:
            pass
