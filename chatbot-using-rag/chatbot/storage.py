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
import shutil
from typing import Any, Protocol

import weaviate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableSerializable
from weaviate.exceptions import WeaviateStartUpError

from .config import AppConfig

logger = logging.getLogger(__name__)


class VectorStoreBackend(Protocol):
    """Contract implemented by all retriever backends.

    The app and eval flows both depend on this minimal surface area, so all
    backends expose the same three operations.
    """

    def initialise(self, config: AppConfig, embeddings) -> object | None:
        ...

    def replace_documents(self, chunks: list, config: AppConfig, embeddings) -> object:
        ...

    def add_documents(self, chunks: list, config: AppConfig, embeddings) -> tuple[object, int]:
        ...


def make_vector_store(config: AppConfig) -> VectorStoreBackend:
    """Return the configured vector-store backend.

    Supported backends
    ------------------
    - weaviate_langchain
    - llamaindex_sentence_window
    """
    backend = config.retrieval_backend.lower().strip()
    if backend == "weaviate_langchain":
        return WeaviateStore()
    if backend == "llamaindex_sentence_window":
        return LlamaIndexSentenceWindowStore()
    raise ValueError(
        f"Unknown retrieval backend '{config.retrieval_backend}'. "
        "Supported: weaviate_langchain | llamaindex_sentence_window"
    )


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


class LlamaIndexSentenceWindowRetriever(RunnableSerializable[str, list[Document]]):
    """Adapter exposing a LangChain-like retriever interface over LlamaIndex.

    This class is also an LCEL Runnable, so it can be composed via the `|`
    operator in the pipeline.
    """

    def __init__(
        self,
        index,
        similarity_top_k: int,
        rerank_enabled: bool = False,
        rerank_model: str = "BAAI/bge-reranker-base",
        rerank_top_n: int = 3,
    ) -> None:
        from llama_index.core.postprocessor import MetadataReplacementPostProcessor

        self._retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        self._window_postprocessor = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )
        self._reranker = None

        if rerank_enabled:
            try:
                import importlib

                sentence_transformer_rerank = None
                for module_name in (
                    "llama_index.core.postprocessor",
                    "llama_index.postprocessor.sbert_rerank",
                ):
                    try:
                        reranker_mod = importlib.import_module(module_name)
                        sentence_transformer_rerank = getattr(
                            reranker_mod,
                            "SentenceTransformerRerank",
                        )
                        break
                    except Exception:
                        continue

                if sentence_transformer_rerank is None:
                    raise ImportError("SentenceTransformerRerank class not found")

                self._reranker = sentence_transformer_rerank(
                    model=rerank_model,
                    top_n=rerank_top_n,
                )
                logger.info(
                    "Sentence-window reranker enabled — model=%s, top_n=%d",
                    rerank_model,
                    rerank_top_n,
                )
            except Exception as exc:
                logger.warning(
                    "Sentence-window reranker requested but unavailable (%s). "
                    "Install 'llama-index-postprocessor-sbert' and restart.",
                    exc,
                )

    def invoke(self, query: str, config=None, **kwargs) -> list[Document]:
        from llama_index.core import QueryBundle

        nodes = self._retriever.retrieve(query)
        nodes = self._window_postprocessor.postprocess_nodes(
            nodes,
            query_bundle=QueryBundle(query_str=query),
        )
        if self._reranker is not None:
            nodes = self._reranker.postprocess_nodes(
                nodes,
                query_bundle=QueryBundle(query_str=query),
            )

        docs: list[Document] = []
        for node_with_score in nodes:
            node = node_with_score.node
            metadata = dict(node.metadata or {})
            metadata.pop("window", None)
            metadata.pop("original_text", None)
            docs.append(Document(page_content=node.get_content(), metadata=metadata))
        return docs

    def __call__(self, query: str) -> list[Document]:
        return self.invoke(query)


class LlamaIndexSentenceWindowStore:
    """Vector-store backend using LlamaIndex sentence-window retrieval.

    Persistence is local-file based. This backend intentionally mirrors the
    WeaviateStore public methods so the rest of the app can stay unchanged.
    """

    def __init__(self) -> None:
        self._index = None
        self._persist_dir: str | None = None

    def initialise(self, config: AppConfig, _embeddings) -> object | None:
        """Load existing index or build one from PDFs in data_sources_dir."""
        from .ingestion import DocumentLoader

        persist_dir = os.path.abspath(config.llamaindex_persist_dir)
        self._persist_dir = persist_dir
        self._index = self._load_index_if_exists(config)

        if self._index is not None:
            count = self._node_count(self._index)
            logger.info("LlamaIndex ready — %d nodes loaded from '%s'.", count, persist_dir)
            return self._as_retriever(self._index, config)

        logger.info("No LlamaIndex data found. Scanning '%s' for PDFs ...", config.data_sources_dir)
        loader = DocumentLoader(config)
        chunks, pdf_paths = loader.load_directory()
        if not chunks:
            logger.warning("No PDFs found — upload one via the UI to get started.")
            return None

        self._index = self._build_index(chunks, config)
        self._persist(self._index, config)
        logger.info(
            "Built LlamaIndex sentence-window index from %d chunk(s), %d PDF(s).",
            len(chunks),
            len(pdf_paths),
        )
        return self._as_retriever(self._index, config)

    def replace_documents(self, chunks: list, config: AppConfig, _embeddings) -> object:
        """Replace the entire persisted index with newly provided documents."""
        self._clear_persist_dir(config)
        self._index = self._build_index(chunks, config)
        self._persist(self._index, config)
        logger.info("Replaced LlamaIndex index with %d chunk(s).", len(chunks))
        return self._as_retriever(self._index, config)

    def add_documents(self, chunks: list, config: AppConfig, _embeddings) -> tuple[object, int]:
        """Append documents as sentence-window nodes into the existing index."""
        if self._index is None:
            self._index = self._load_index_if_exists(config)

        if self._index is None:
            self._index = self._build_index(chunks, config)
        else:
            self._index.insert_nodes(self._to_sentence_window_nodes(chunks, config))

        self._persist(self._index, config)
        total = self._node_count(self._index)
        logger.info("Added %d chunk(s). LlamaIndex now has %d nodes.", len(chunks), total)
        return self._as_retriever(self._index, config), total

    def _build_index(self, chunks: list, config: AppConfig):
        from llama_index.core import StorageContext, VectorStoreIndex

        nodes = self._to_sentence_window_nodes(chunks, config)
        embed_model = self._make_llama_embed_model(config)
        storage_context = StorageContext.from_defaults()
        return VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
        )

    def _to_sentence_window_nodes(self, chunks: list, config: AppConfig) -> list:
        from llama_index.core import Document as LlamaDocument
        from llama_index.core.node_parser import SentenceWindowNodeParser

        documents = [
            LlamaDocument(text=doc.page_content, metadata=dict(doc.metadata or {}))
            for doc in chunks
        ]

        parser = SentenceWindowNodeParser.from_defaults(
            window_size=config.sentence_window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        return parser.get_nodes_from_documents(documents)

    def _load_index_if_exists(self, config: AppConfig):
        from llama_index.core import StorageContext, load_index_from_storage

        persist_dir = os.path.abspath(config.llamaindex_persist_dir)
        if not os.path.isdir(persist_dir):
            return None

        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            return load_index_from_storage(
                storage_context,
                embed_model=self._make_llama_embed_model(config),
            )
        except Exception as exc:
            logger.warning("Could not load existing LlamaIndex index — %s", exc)
            return None

    @staticmethod
    def _persist(index, config: AppConfig) -> None:
        persist_dir = os.path.abspath(config.llamaindex_persist_dir)
        os.makedirs(persist_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=persist_dir)

    @staticmethod
    def _clear_persist_dir(config: AppConfig) -> None:
        persist_dir = os.path.abspath(config.llamaindex_persist_dir)
        if os.path.isdir(persist_dir):
            shutil.rmtree(persist_dir, ignore_errors=True)

    @staticmethod
    def _node_count(index) -> int:
        try:
            return len(index.docstore.docs)
        except Exception:
            return 0

    @staticmethod
    def _as_retriever(index, config: AppConfig) -> LlamaIndexSentenceWindowRetriever:
        return LlamaIndexSentenceWindowRetriever(
            index,
            similarity_top_k=config.retriever_k,
            rerank_enabled=config.sentence_window_rerank_enabled,
            rerank_model=config.sentence_window_rerank_model,
            rerank_top_n=config.sentence_window_rerank_top_n,
        )

    @staticmethod
    def _make_llama_embed_model(config: AppConfig):
        provider = config.embed_provider.lower().strip()

        if provider == "huggingface":
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            lower_name = config.embed_model_name.lower()
            if "nomic" in lower_name:
                return HuggingFaceEmbedding(
                    model_name=config.embed_model_name,
                    trust_remote_code=True,
                    text_instruction="search_document: ",
                    query_instruction="search_query: ",
                )
            return HuggingFaceEmbedding(model_name=config.embed_model_name)

        if provider == "openai":
            from llama_index.embeddings.openai import OpenAIEmbedding

            return OpenAIEmbedding(model=config.embed_model_name)

        if provider == "ollama":
            from llama_index.embeddings.ollama import OllamaEmbedding

            return OllamaEmbedding(
                model_name=config.embed_model_name,
                base_url=config.ollama_base_url,
            )

        raise ValueError(
            f"Unsupported embed provider '{config.embed_provider}' for LlamaIndex backend. "
            "Supported: huggingface | openai | ollama"
        )
