"""app.py — Gradio UI: event handlers, demo builder, and app launcher.

All event handlers are closures capturing (config, state) so multiple
independent instances can run in the same process on different ports.
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import os
import sys
import threading

logger = logging.getLogger(__name__)

# ── Python 3.14 asyncio compatibility patch ───────────────────────────────────

_gradio_loop_local = threading.local()


def _get_or_create_thread_loop() -> asyncio.AbstractEventLoop:
    loop = getattr(_gradio_loop_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _gradio_loop_local.loop = loop
    asyncio.set_event_loop(loop)
    return loop


def _patch_gradio_asyncio() -> None:
    """Work around Gradio creating throwaway event loops on Python 3.14."""
    if sys.version_info[:2] < (3, 14):
        return
    import gradio.queueing as gradio_queueing
    import gradio.utils as gradio_utils

    def safe_get_lock() -> asyncio.Lock:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            _get_or_create_thread_loop()
        return asyncio.Lock()

    def safe_get_stop_event() -> asyncio.Event:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            _get_or_create_thread_loop()
        return asyncio.Event()

    gradio_utils.safe_get_lock = safe_get_lock
    gradio_utils.safe_get_stop_event = safe_get_stop_event
    gradio_queueing.safe_get_lock = safe_get_lock


atexit.register(lambda: getattr(_gradio_loop_local, "loop", None) and not _gradio_loop_local.loop.is_closed() and _gradio_loop_local.loop.close())

import gradio as gr  # noqa: E402

_patch_gradio_asyncio()

from .config import AppConfig  # noqa: E402
from .state  import AppState   # noqa: E402


# ── UI helpers ────────────────────────────────────────────────────────────────

def _source_table(docs: list) -> str:
    """Format retrieved documents as a Markdown table."""
    if not docs:
        return "No sources retrieved."
    rows = ["| # | Source | Page |", "|---|--------|------|"]
    for i, doc in enumerate(docs, 1):
        src  = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "—")
        rows.append(f"| {i} | {src} | {page} |")
    return "\n".join(rows)


# ── Event handlers ────────────────────────────────────────────────────────────

def _make_upload_handler(config: AppConfig, state: AppState):
    """Replace corpus: split → embed → persist → rebuild chain."""
    from .ingestion          import DocumentLoader
    from .providers.embeddings import make_embeddings

    def _handle(files, session_id: str) -> tuple[str, list]:
        if not files:
            return "No files uploaded.", []

        embeddings = make_embeddings(config)
        loader     = DocumentLoader(config)
        all_chunks, names = [], []

        for fobj in files:
            path  = fobj.name if hasattr(fobj, "name") else str(fobj)
            fname = os.path.basename(path)
            logger.info("Upload: processing '%s' …", fname)
            chunks = loader.load_pdf(path)
            logger.info("Upload: '%s' → %d chunks", fname, len(chunks))
            all_chunks.extend(chunks)
            names.append(fname)

        if not all_chunks:
            return "No text could be extracted from the uploaded file(s).", []

        retriever = state.vector_store.replace_documents(all_chunks, config, embeddings)
        state.build_chain(retriever, config)
        state.corpus = ", ".join(names)

        return (
            f"Indexed **{len(all_chunks)} chunks** from {len(names)} file(s): {state.corpus}",
            [],
        )

    return _handle


def _make_add_handler(config: AppConfig, state: AppState):
    """Add PDFs to the existing corpus without clearing it."""
    from .ingestion            import DocumentLoader
    from .providers.embeddings import make_embeddings

    def _handle(files) -> str:
        if not files:
            return "No files selected."
        if state.retriever is None:
            return "No active vector store. Upload a PDF first."

        embeddings = make_embeddings(config)
        loader     = DocumentLoader(config)
        all_chunks, names = [], []

        for fobj in files:
            path  = fobj.name if hasattr(fobj, "name") else str(fobj)
            fname = os.path.basename(path)
            logger.info("Add: processing '%s' …", fname)
            all_chunks.extend(loader.load_pdf(path))
            names.append(fname)

        if not all_chunks:
            return "No text could be extracted."

        retriever, total = state.vector_store.add_documents(all_chunks, config, embeddings)
        state.build_chain(retriever, config)
        state.corpus = state.corpus + ", " + ", ".join(names)

        return (
            f"Added **{len(all_chunks)} chunks** from {len(names)} file(s). "
            f"DB now contains **{total} chunks** total."
        )

    return _handle


def _make_chat_handler(state: AppState):
    """Stream the RAG answer and return source documents."""

    def _handle(message: str, history: list, session_id: str):
        if state.chain is None:
            yield history + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": "Please upload a PDF first."},
            ], "No active chain."
            return

        if not message.strip():
            yield history, ""
            return

        history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": ""},
        ]
        accumulated = ""

        for chunk in state.chain.stream(message, session_id):
            accumulated += chunk
            history[-1]["content"] = accumulated
            yield history, ""

        try:
            docs    = state.retriever.invoke(message)
            sources = _source_table(docs)
        except Exception:
            sources = "Could not retrieve sources."

        yield history, sources

    return _handle


def _make_clear_handler(state: AppState):
    def _handle(session_id: str) -> tuple[list, str]:
        state.clear_session(session_id)
        return [], ""
    return _handle


# ── Demo builder ──────────────────────────────────────────────────────────────

def build_demo(config: AppConfig, state: AppState) -> gr.Blocks:
    """Construct and return the Gradio Blocks UI (does not launch it)."""
    model_short    = config.embed_model_name.split("/")[-1]
    title          = f"RAG Chatbot — {model_short} (port {config.port})"
    phoenix_link   = f"[Phoenix traces](http://localhost:{config.phoenix_port})" if config.phoenix_enabled else "Disabled"
    backend = config.retrieval_backend.lower().strip()
    rerank_label = "disabled"
    if backend == "llamaindex_sentence_window" and config.sentence_window_rerank_enabled:
        rerank_label = (
            f"enabled ({config.sentence_window_rerank_model}, top_n={config.sentence_window_rerank_top_n})"
        )

    if backend == "llamaindex_sentence_window":
        retrieval_label = (
            "LlamaIndex Sentence Window, "
            f"top-k={config.retriever_k}, window_size={config.sentence_window_size}"
        )
        retrieval_steps_label = f"top-{config.retriever_k} sentence-window retrieval"
        backend_hints = """
### Active retrieval mode
- Backend: **llamaindex_sentence_window**
- Best for local precision and context continuity around matching sentences
- Optional reranker improves precision for ambiguous / multi-topic questions

### Tuning guidance
- If answers miss nearby details: increase `SENTENCE_WINDOW_SIZE` (try 4-6)
- If answers include too much context: decrease `SENTENCE_WINDOW_SIZE` (try 2)
- If results are noisy: reduce `RETRIEVER_K` and enable sentence-window reranking
"""
        vector_store_label = (
            f"LlamaIndex local persistence — {config.llamaindex_persist_dir}"
        )
    else:
        retrieval_label = (
            f"Hybrid (vector + BM25), top-k={config.retriever_k}, alpha={config.hybrid_alpha}"
        )
        retrieval_steps_label = f"top-{config.retriever_k} chunks via hybrid search (vector + BM25)"
        backend_hints = """
### Active retrieval mode
- Backend: **weaviate_langchain**
- Best for broad recall and mixed semantic + keyword matching
- Great default when documents use varied terminology

### Tuning guidance
- If exact keyword matches are weak: lower `HYBRID_ALPHA` toward 0.5
- If semantic matches are weak: raise `HYBRID_ALPHA` toward 0.85
- If answers miss relevant sections: increase `RETRIEVER_K`
"""
        vector_store_label = (
            f"Weaviate (embedded) — {config.weaviate_persist_dir} / {config.weaviate_index_name}"
        )

    upload_handler = _make_upload_handler(config, state)
    add_handler    = _make_add_handler(config, state)
    chat_handler   = _make_chat_handler(state)
    clear_handler  = _make_clear_handler(state)

    def _refresh_history(session_id: str) -> str:
        history = state.get_session_history(session_id)
        if not history:
            return "*No messages yet. Start chatting in the Conversation tab.*"
        lines = []
        for msg in history:
            role    = "**You**" if msg.type == "human" else "**Assistant**"
            content = msg.content[:400] + ("…" if len(msg.content) > 400 else "")
            lines.append(f"{role}: {content}")
        return "\n\n---\n\n".join(lines)

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"## {title}")
        session_id = gr.State(lambda: __import__("uuid").uuid4().hex)

        with gr.Tabs():

            # ── Conversation ──────────────────────────────────────────────────
            with gr.Tab("💬 Conversation"):
                chatbot = gr.Chatbot(
                    label="", height=500, show_label=False, layout="bubble",
                    avatar_images=(None, "https://api.dicebear.com/9.x/bottts-neutral/svg?seed=copilot"),
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask a question about your documents …",
                        label="", lines=1, scale=5, show_label=False, submit_btn=False,
                    )
                    send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=90)
                sources_md = gr.Markdown(value="*Retrieved sources will appear here after each answer.*")

            # ── Chat History ──────────────────────────────────────────────────
            with gr.Tab("📜 Chat History"):
                history_md  = gr.Markdown(value="*No messages yet.*")
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")

            # ── Database ──────────────────────────────────────────────────────
            with gr.Tab("🗄 Database"):
                gr.Markdown("### Current corpus")
                corpus_info = gr.Textbox(value=state.corpus, label="Loaded corpus", interactive=False)

                gr.Markdown("### Replace corpus\n*Replaces the current corpus and clears conversation history.*")
                upload_btn    = gr.UploadButton("📄 Load & index", file_types=[".pdf"], file_count="multiple", variant="primary")
                upload_status = gr.Markdown()

                gr.Markdown("### Add to existing corpus")
                add_btn    = gr.UploadButton("➕ Add to DB", file_types=[".pdf"], file_count="multiple", variant="secondary")
                add_status = gr.Markdown()

            # ── Configure ────────────────────────────────────────────────────
            with gr.Tab("⚙️ Configure"):
                gr.Markdown("### Session")
                session_display = gr.Textbox(label="Session ID", interactive=False,
                                             info="Each browser session has its own isolated conversation history.")
                clear_btn    = gr.Button("🗑 Clear conversation", variant="stop", size="sm")
                clear_status = gr.Textbox(label="", interactive=False, show_label=False)

                gr.Markdown(f"""
### Chain details
| Component | Value |
|---|---|
| LLM | `{config.llm_provider}` — `{config.llm_model_id}` |
| Embeddings | `{config.embed_provider}` — `{config.embed_model_name}` |
| Backend | `{config.retrieval_backend}` |
| Sentence-window reranker | {rerank_label} |
| Vector store | {vector_store_label} |
| PDF loader | PyMuPDF — preserves tables and multi-column layouts |
| Chunking | tiktoken `{config.tiktoken_encoding}` — {config.chunk_size} tokens / {config.chunk_overlap} overlap |
| Retrieval | {retrieval_label} |
| History | Per-session `list[HumanMessage | AIMessage]` |
| Observability | {phoenix_link} |

### How it works
1. Your question is condensed with chat history into a standalone question
2. The standalone question retrieves {retrieval_steps_label}
3. The LLM streams an answer using those chunks as context
4. The Q&A pair is appended to your session history for the next turn

{backend_hints}
""")

        # ── Wire events ───────────────────────────────────────────────────────
        for trigger in (send_btn.click, msg_box.submit):
            trigger(
                fn=chat_handler, inputs=[msg_box, chatbot, session_id], outputs=[chatbot, sources_md],
            ).then(fn=lambda: "", outputs=[msg_box])

        refresh_btn.click(fn=_refresh_history, inputs=[session_id], outputs=[history_md])

        upload_btn.click(
            fn=upload_handler, inputs=[upload_btn, session_id], outputs=[upload_status, chatbot],
        ).then(fn=lambda: state.corpus, outputs=[corpus_info])

        add_btn.click(
            fn=add_handler, inputs=[add_btn], outputs=[add_status],
        ).then(fn=lambda: state.corpus, outputs=[corpus_info])

        clear_btn.click(
            fn=clear_handler, inputs=[session_id], outputs=[chatbot, sources_md],
        ).then(fn=lambda: "Conversation cleared.", outputs=[clear_status])

        demo.load(fn=lambda sid: sid, inputs=[session_id], outputs=[session_display])

    return demo


# ── App runner ────────────────────────────────────────────────────────────────

def run_app(config: AppConfig) -> None:
    """Initialise the vector store and launch the Gradio app."""
    from .infrastructure.logging       import configure_logging
    from .infrastructure.observability import setup_observability
    from .providers.embeddings         import make_embeddings
    from .storage                      import make_vector_store

    configure_logging()

    if sys.version_info[:2] >= (3, 14):
        logger.warning(
            "Python 3.14 detected. "
            "This stack is most stable on 3.11–3.13 and may emit deprecation warnings."
        )

    setup_observability(config)

    state      = AppState()
    state.vector_store = make_vector_store(config)
    embeddings = make_embeddings(config)

    logger.info(
        "Initialising — llm=%s, embed=%s, port=%d",
        config.llm_provider, config.embed_model_name, config.port,
    )
    retriever = state.vector_store.initialise(config, embeddings)
    if retriever:
        state.build_chain(retriever, config)
        state.corpus = config.weaviate_index_name

    demo = build_demo(config, state)
    demo.launch(server_port=config.port, share=False, theme=gr.themes.Soft())
