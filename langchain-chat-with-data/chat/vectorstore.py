"""Vector store helpers — MMR retriever setup, chain wiring, and auto-initialisation."""
from __future__ import annotations

import os

from .config import AppConfig
from .state  import AppState


def as_mmr_retriever(vectordb, config: AppConfig):
    """Wrap a Chroma collection as an MMR retriever.

    MMR (Maximum Marginal Relevance) selects *k* documents that are similar to
    the query while being dissimilar to each other, reducing redundancy in the
    retrieved context passed to the LLM.
    """
    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":           config.retriever_k,
            "fetch_k":     config.retriever_fetch_k,   # candidate pool before re-ranking
            "lambda_mult": config.mmr_lambda,           # 1.0 = similarity, 0.0 = diversity
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

    Fast path  — persist_dir already exists → load and wire.
    First run  — scan source_docs_dir for *.pdf, embed, persist, then wire.
    """
    from langchain_chroma import Chroma

    from .embeddings import make_embeddings
    from .pdf_utils  import split_pdfs_from_dir

    embeddings = make_embeddings(config)

    # ── Fast path: existing ChromaDB ─────────────────────────────────────────
    if os.path.isdir(config.persist_dir) and os.listdir(config.persist_dir):
        print(f"[vectorstore] Loading existing DB from '{config.persist_dir}' …")
        try:
            vectordb = Chroma(
                persist_directory=config.persist_dir,
                embedding_function=embeddings,
            )
            count = vectordb._collection.count()
            if count == 0:
                print(f"[vectorstore] Existing DB at '{config.persist_dir}' is empty — rebuilding from PDFs …")
                # Fall through to first-run path below
            else:
                retriever = as_mmr_retriever(vectordb, config)
                wire_chain(retriever, state, config)
                state.corpus = os.path.basename(config.persist_dir)
                print(f"[vectorstore] Ready — {count} chunks loaded.")
                return
        except Exception as exc:
            print(
                f"[vectorstore] WARNING: Could not load existing DB — {exc}\n"
                "This usually means the persisted embeddings were created with a "
                "different model or dimensionality.  Delete the folder and re-upload "
                "your PDFs to rebuild the index."
            )
            return

    # ── First run: build from PDFs ────────────────────────────────────────────
    print(f"[vectorstore] No existing DB at '{config.persist_dir}'.  Scanning '{config.source_docs_dir}' …")
    splits, pdf_paths = split_pdfs_from_dir(config.source_docs_dir, config)

    if not splits:
        print("[vectorstore] No PDF files found — upload one via the UI to get started.")
        return

    print(f"[vectorstore] Embedding {len(splits)} chunks from {len(pdf_paths)} PDF(s) …")
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=config.persist_dir,
    )
    retriever = as_mmr_retriever(vectordb, config)
    wire_chain(retriever, state, config)
    names        = [os.path.basename(p) for p in pdf_paths]
    state.corpus = ", ".join(names)
    print(f"[vectorstore] DB persisted with {vectordb._collection.count()} chunks.")
