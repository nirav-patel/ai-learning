"""Vector store helpers — MMR retriever setup, chain wiring, and auto-initialisation.

Vector store backend: Weaviate (embedded, runs in-process — no Docker needed).
The embedded instance stores data in a local temp directory and is cleaned up
when the process exits.  For production, swap weaviate.connect_to_embedded()
with weaviate.connect_to_weaviate_cloud() or weaviate.connect_to_local().
"""
from __future__ import annotations

import os

import weaviate
from weaviate.exceptions import WeaviateStartUpError

from .config import AppConfig
from .state  import AppState

# Module-level Weaviate client — shared across calls, closed on process exit.
_weaviate_client: weaviate.WeaviateClient | None = None
_weaviate_persist_dir: str | None = None


def _get_client(config: AppConfig) -> weaviate.WeaviateClient:
    """Return (or lazily create) the embedded Weaviate client."""
    global _weaviate_client, _weaviate_persist_dir
    persist_dir = os.path.abspath(config.weaviate_persist_dir)
    if (
        _weaviate_client is None
        or not _weaviate_client.is_connected()
        or _weaviate_persist_dir != persist_dir
    ):
        if _weaviate_client is not None and _weaviate_client.is_connected():
            _weaviate_client.close()
        try:
            _weaviate_client = weaviate.connect_to_embedded(
                persistence_data_path=persist_dir
            )
        except WeaviateStartUpError as exc:
            # Another embedded instance may already be serving default ports.
            if "already listening on ports" not in str(exc):
                raise
            _weaviate_client = weaviate.connect_to_local(port=8079, grpc_port=50050)
        _weaviate_persist_dir = persist_dir
    return _weaviate_client


def as_mmr_retriever(vectordb, config: AppConfig):
    """Return a hybrid retriever (vector + BM25 keyword search).

    Weaviate's hybrid search runs both a dense vector search and a sparse BM25
    keyword search, then merges results using Reciprocal Rank Fusion (RRF).
    This handles queries that are either semantically rich OR keyword-specific
    better than either method alone.

    alpha controls the blend:
      1.0 = pure vector (semantic)   0.0 = pure BM25 (keyword)   0.5 = balanced
    """
    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k":     config.retriever_k,
            "alpha": config.hybrid_alpha,
        },
    )


def wire_chain(retriever, state: AppState, config: AppConfig) -> None:
    """Attach *retriever* to *state*, (re-)building the LLM and chain as needed."""
    from .llm_factory import make_llm
    from .rag_chain    import build_rag_chain

    if state.llm is None:
        state.llm = make_llm(config)

    state.retriever = retriever
    state.chain     = build_rag_chain(retriever, state.llm, state.get_session_history)


def initialise(config: AppConfig, state: AppState) -> None:
    """Bring the vector store to a ready state, then wire the chain.

    Fast path  — Weaviate collection already has chunks → load and wire.
    First run  — scan source_docs_dir for *.pdf, embed, ingest, then wire.
    """
    from langchain_weaviate import WeaviateVectorStore

    from .embeddings import make_embeddings
    from .pdf_utils  import split_pdfs_from_dir

    embeddings = make_embeddings(config)
    client     = _get_client(config)

    # ── Fast path: collection already populated ───────────────────────────────
    try:
        vectordb = WeaviateVectorStore(
            client=client,
            index_name=config.weaviate_index_name,
            text_key="text",
            embedding=embeddings,
        )
        # Weaviate returns an empty list if the collection doesn't exist yet
        sample = vectordb.similarity_search("test", k=1)
        if sample:
            count = client.collections.get(config.weaviate_index_name).aggregate.over_all().total_count
            retriever = as_mmr_retriever(vectordb, config)
            wire_chain(retriever, state, config)
            state.corpus = config.weaviate_index_name
            print(f"[vectorstore] Ready — {count} chunks loaded from '{config.weaviate_index_name}'.")
            return
    except Exception as exc:  # noqa: BLE001
        print(f"[vectorstore] Could not load existing collection — {exc}")

    # ── First run: build from PDFs ────────────────────────────────────────────
    print(f"[vectorstore] No data found.  Scanning '{config.source_docs_dir}' for PDFs …")
    splits, pdf_paths = split_pdfs_from_dir(config.source_docs_dir, config)

    if not splits:
        print("[vectorstore] No PDF files found — upload one via the UI to get started.")
        return

    print(f"[vectorstore] Embedding {len(splits)} chunks from {len(pdf_paths)} PDF(s) …")
    vectordb = WeaviateVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        client=client,
        index_name=config.weaviate_index_name,
    )
    retriever = as_mmr_retriever(vectordb, config)
    wire_chain(retriever, state, config)
    names        = [os.path.basename(p) for p in pdf_paths]
    state.corpus = ", ".join(names)
    print(f"[vectorstore] Ingested {len(splits)} chunks into '{config.weaviate_index_name}'.") 
