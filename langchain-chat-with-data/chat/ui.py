"""Gradio UI — event handlers, demo builder, and app runner.

All event handlers are returned as closures that capture (config, state) so
multiple independent Gradio instances (nomic / minilm) can run in the same
Python process on different ports without any shared global mutable state.
"""
from __future__ import annotations

import os
import tempfile

import gradio as gr

from .config     import AppConfig
from .state      import AppState


# ── UI helpers ────────────────────────────────────────────────────────────────

def _source_table(docs: list) -> str:
    """Format a list of retrieved documents as a Markdown table."""
    if not docs:
        return "No sources retrieved."
    rows = ["| # | Source | Page |", "|---|--------|------|"]
    for i, doc in enumerate(docs, 1):
        src  = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "—")
        rows.append(f"| {i} | {src} | {page} |")
    return "\n".join(rows)


# ── Event handlers (closures) ─────────────────────────────────────────────────

def _make_upload_handler(config: AppConfig, state: AppState):
    """Handle PDF upload: split → embed → persist → wire chain."""
    from langchain_chroma import Chroma
    from .embeddings  import make_embeddings
    from .pdf_utils   import split_pdf
    from .vectorstore import as_mmr_retriever, wire_chain

    def _handle(files, session_id: str) -> tuple[str, list]:
        """Upload one or more PDFs and add them to the vector store.

        Returns (status_text, updated_chatbot_list).
        """
        if not files:
            return "No files uploaded.", []

        embeddings = make_embeddings(config)
        all_splits = []
        names      = []

        for fobj in files:
            path  = fobj.name if hasattr(fobj, "name") else str(fobj)
            fname = os.path.basename(path)
            print(f"[upload] Processing '{fname}' …")
            splits = split_pdf(path, config)
            print(f"[upload]   → {len(splits)} chunks")
            all_splits.extend(splits)
            names.append(fname)

        if not all_splits:
            return "No text could be extracted from the uploaded file(s).", []

        print(f"[upload] Embedding {len(all_splits)} chunks …")
        vectordb = Chroma.from_documents(
            documents=all_splits,
            embedding=embeddings,
            persist_directory=config.persist_dir,
        )
        retriever    = as_mmr_retriever(vectordb, config)
        wire_chain(retriever, state, config)
        state.corpus = ", ".join(names)

        msg = (
            f"Indexed **{len(all_splits)} chunks** from "
            f"{len(names)} file(s): {state.corpus}"
        )
        return msg, []

    return _handle


def _make_add_handler(config: AppConfig, state: AppState):
    """Handle 'Add to DB' — embed new PDFs into an *existing* collection."""
    from langchain_chroma import Chroma
    from .embeddings  import make_embeddings
    from .pdf_utils   import split_pdf
    from .vectorstore import as_mmr_retriever, wire_chain

    def _handle(files) -> str:
        if not files:
            return "No files selected."
        if state.retriever is None:
            return "No active vector store.  Upload a PDF first."

        embeddings = make_embeddings(config)
        all_splits = []
        names      = []

        for fobj in files:
            path   = fobj.name if hasattr(fobj, "name") else str(fobj)
            fname  = os.path.basename(path)
            print(f"[add] Processing '{fname}' …")
            splits = split_pdf(path, config)
            all_splits.extend(splits)
            names.append(fname)

        if not all_splits:
            return "No text could be extracted."

        # Load existing collection and add new chunks
        vectordb = Chroma(
            persist_directory=config.persist_dir,
            embedding_function=embeddings,
        )
        vectordb.add_documents(all_splits)
        retriever    = as_mmr_retriever(vectordb, config)
        wire_chain(retriever, state, config)
        state.corpus = state.corpus + ", " + ", ".join(names)

        return (
            f"Added **{len(all_splits)} chunks** from {len(names)} file(s). "
            f"DB now contains **{vectordb._collection.count()} chunks** total."
        )

    return _handle


def _make_chat_handler(state: AppState):
    """Handle user message — stream the RAG answer."""

    def _handle(message: str, history: list, session_id: str):
        """
        Args:
            message:    User's typed question.
            history:    Gradio-managed chat history (list of message dicts).
            session_id: Gradio state value used to look up per-session LangChain memory.

        Yields partial (history, sources) tuples for streaming updates.
        """
        if state.chain is None:
            history = history + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": "Please upload a PDF first."},
            ]
            yield history, "No active chain."
            return

        if not message.strip():
            yield history, ""
            return

        history     = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": ""},
        ]
        accumulated = ""

        for chunk in state.chain(message, session_id):
            accumulated += chunk
            history[-1]["content"] = accumulated
            yield history, ""

        # After streaming completes, fetch the retrieved docs for the source table.
        # We re-invoke the retriever on the question to get the same docs.
        try:
            docs    = state.retriever.invoke(message)
            sources = _source_table(docs)
        except Exception:
            sources = "Could not retrieve sources."

        yield history, sources

    return _handle


def _make_clear_handler(state: AppState):
    """Clear the chat history for the current session."""

    def _handle(session_id: str) -> tuple[list, str]:
        state.clear_session(session_id)
        return [], ""

    return _handle


# ── Demo builder ──────────────────────────────────────────────────────────────

def build_demo(config: AppConfig, state: AppState) -> gr.Blocks:
    """Construct and return the Gradio Blocks UI.

    The demo is NOT launched here — call .launch() on the returned object,
    or use run_app() below which does that for you.
    """
    model_short = config.embed_model_name.split("/")[-1]
    title       = f"Chat with your PDFs — {model_short} (port {config.port})"

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

            # ── Tab 1: Conversation ───────────────────────────────────────────
            with gr.Tab("💬 Conversation"):
                chatbot = gr.Chatbot(
                    label="", height=500, show_label=False,
                    layout="bubble",
                    avatar_images=(None, "https://api.dicebear.com/9.x/bottts-neutral/svg?seed=copilot"),
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask a question about your documents …",
                        label="",
                        lines=1,
                        scale=5,
                        show_label=False,
                        submit_btn=False,
                    )
                    send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=90)
                sources_md = gr.Markdown(
                    value="*Retrieved sources will appear here after each answer.*"
                )

            # ── Tab 2: Chat History ───────────────────────────────────────────
            with gr.Tab("📜 Chat History"):
                history_md  = gr.Markdown(value="*No messages yet.*")
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")

            # ── Tab 3: Database ───────────────────────────────────────────────
            with gr.Tab("🗄 Database"):
                gr.Markdown("### Current corpus")
                corpus_info = gr.Textbox(
                    value=state.corpus,
                    label="Loaded corpus",
                    interactive=False,
                )
                gr.Markdown(
                    "### Replace corpus\n"
                    "*Replaces the current corpus and clears conversation history.*"
                )
                upload_btn    = gr.UploadButton(
                    "📄 Load & index", file_types=[".pdf"],
                    file_count="multiple", variant="primary",
                )
                upload_status = gr.Markdown()

                gr.Markdown("### Add to existing corpus")
                add_btn    = gr.UploadButton(
                    "➕ Add to DB", file_types=[".pdf"],
                    file_count="multiple", variant="secondary",
                )
                add_status = gr.Markdown()

            # ── Tab 4: Configure ──────────────────────────────────────────────
            with gr.Tab("⚙️ Configure"):
                gr.Markdown("### Session")
                session_display = gr.Textbox(
                    label="Session ID",
                    interactive=False,
                    info="Each browser session has its own isolated conversation history.",
                )
                clear_btn    = gr.Button("🗑 Clear conversation", variant="stop", size="sm")
                clear_status = gr.Textbox(label="", interactive=False, show_label=False)

                gr.Markdown(f"""
### Chain details
| Component | Value |
|---|---|
| LLM | AWS Bedrock `{config.llm_model_id}` |
| Embeddings | `{config.embed_model_name}` |
| Vector store | ChromaDB — `{config.persist_dir}` |
| PDF loader | PyMuPDF — preserves tables, multi-column layouts |
| Chunking | tiktoken `{config.tiktoken_encoding}` — {config.chunk_size} tokens / {config.chunk_overlap} overlap |
| Retrieval | MMR top-k={config.retriever_k}, fetch\\_k={config.retriever_fetch_k}, λ={config.mmr_lambda} |
| History | Per-session `list[HumanMessage | AIMessage]` |

### How it works
1. Your question is condensed with chat history into a standalone question
2. The standalone question retrieves the top-{config.retriever_k} document chunks (MMR)
3. The LLM streams an answer using those chunks as context
4. The Q&A pair is appended to your session history for the next turn
""")

        # ── Wire events ───────────────────────────────────────────────────────
        send_btn.click(
            fn=chat_handler,
            inputs=[msg_box, chatbot, session_id],
            outputs=[chatbot, sources_md],
        ).then(fn=lambda: "", outputs=[msg_box])

        msg_box.submit(
            fn=chat_handler,
            inputs=[msg_box, chatbot, session_id],
            outputs=[chatbot, sources_md],
        ).then(fn=lambda: "", outputs=[msg_box])

        refresh_btn.click(
            fn=_refresh_history,
            inputs=[session_id],
            outputs=[history_md],
        )

        upload_btn.click(
            fn=upload_handler,
            inputs=[upload_btn, session_id],
            outputs=[upload_status, chatbot],
        ).then(
            fn=lambda: state.corpus,
            outputs=[corpus_info],
        )

        add_btn.click(
            fn=add_handler,
            inputs=[add_btn],
            outputs=[add_status],
        ).then(
            fn=lambda: state.corpus,
            outputs=[corpus_info],
        )

        clear_btn.click(
            fn=clear_handler,
            inputs=[session_id],
            outputs=[chatbot, sources_md],
        ).then(fn=lambda: "Conversation cleared.", outputs=[clear_status])

        # Populate session ID display when the Configure tab is visited
        demo.load(fn=lambda sid: sid, inputs=[session_id], outputs=[session_display])

    return demo


# ── App runner ────────────────────────────────────────────────────────────────

def run_app(config: AppConfig) -> None:
    """Initialise the vector store and launch the Gradio app."""
    from .state      import AppState
    from .vectorstore import initialise

    state = AppState()

    print(f"[run_app] Initialising — model={config.embed_model_name}, port={config.port}")
    initialise(config, state)

    demo = build_demo(config, state)
    demo.launch(server_port=config.port, share=False, theme=gr.themes.Soft())
