"""
Chat UI v2 — Modern LCEL + Gradio
====================================
Replaces the deprecated ConversationalRetrievalChain + Memory with:
  • RunnableWithMessageHistory  (langchain_core) — manages per-session history
  • InMemoryChatMessageHistory  (langchain_core) — stores messages in RAM
  • Pure LCEL pipeline          (langchain_core) — composable, streamable

IMPROVEMENTS OVER l6_chat_ui.py (v1)
──────────────────────────────────────
  ✅ Streaming responses (tokens arrive progressively)
  ✅ No deprecated memory classes
  ✅ Per-session isolated history (multi-user safe)
  ✅ Sources shown inline with each answer
  ✅ Tabbed UI — Conversation / Chat History / Database / Configure
  ✅ Error handling that won't crash the UI

HOW THE CHAIN WORKS
────────────────────
  Input:  {"input": user_question}   +   session_id (for history lookup)

  Step 1 — Condenser (only if history exists):
    history + question → standalone question (LLM call)
    e.g. "what about fathers?" → "what is the paternity leave policy?"

  Step 2 — Retriever:
    standalone_question → [relevant chunks]

  Step 3 — Answerer:
    context + history + original question → streamed answer (LLM call)

  History is stored per session_id in InMemoryChatMessageHistory.
  RunnableWithMessageHistory injects + updates history automatically.

RUN
───
    cd langchain-chat-with-data
    uv run python l6_chat_ui.py
    # open http://localhost:7860
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import uuid
import warnings

import boto3
import certifi

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

sys.path.append("..")
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# ── LangSmith ─────────────────────────────────────────────────────────────────
_ls_key = os.getenv("LANGCHAIN_API_KEY", "")
if _ls_key:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault(
        "LANGCHAIN_ENDPOINT",
        os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
    )
    os.environ.setdefault("LANGCHAIN_PROJECT", "langchain-chat-with-data-ui")

# ── Configuration ─────────────────────────────────────────────────────────────
# nomic-embed-text-v1.5: 8192-token context window, 768-dim vectors, MTEB ~62
# Significantly outperforms all-MiniLM-L6-v2 (256 tokens, 384-dim, MTEB ~56)
# Requires trust_remote_code=True and task-specific prefixes (see _make_embeddings)
EMBED_MODEL_NAME  = "nomic-ai/nomic-embed-text-v1.5"
LLM_MODEL_ID      = "us.anthropic.claude-sonnet-4-6"
AWS_REGION        = os.getenv("AWS_REGION", "us-west-1")
# Separate persist dir from the old sample_chroma_db (built with all-MiniLM-L6-v2, 384-dim).
# nomic_chroma_db is built with nomic-embed-text-v1.5 (768-dim) and persisted on first run.
DEFAULT_PERSIST   = os.path.join(os.path.dirname(__file__), "nomic_chroma_db")
# Source PDFs to index at startup if nomic_chroma_db does not yet exist.
# All *.pdf files found directly inside this directory are loaded automatically.
SOURCE_DOCS_DIR   = os.path.join(os.path.dirname(__file__), "docs")
# 512 tokens is well within nomic's 8192-token limit and keeps tables / paragraphs intact.
# nomic uses a BERT-style tokenizer (~10-20% more tokens than cl100k_base for the same
# text), so 512 cl100k tokens ≈ 580 nomic tokens at most — safe headroom before 8192.
CHUNK_SIZE        = 512   # tokens (cl100k_base / BPE)
CHUNK_OVERLAP     = 64    # tokens ← 12.5% overlap preserves cross-boundary context
# cl100k_base is the underlying BPE encoding used by GPT-4 / text-embedding-3 models.
# It is a good approximation for splitting; no OpenAI call is ever made.
TIKTOKEN_ENCODING = "cl100k_base"
RETRIEVER_K       = 3
# MMR retrieval: fetches fetch_k candidates then re-ranks for relevance + diversity.
# Prevents adjacent overlap-chunks from all landing in the same top-k result set.
# lambda_mult=0.7 leans toward relevance; lower values increase diversity.
RETRIEVER_FETCH_K = RETRIEVER_K * 4   # candidate pool before MMR re-ranking
MMR_LAMBDA        = 0.7               # 1.0 = pure similarity, 0.0 = pure diversity

# ── Global app state ───────────────────────────────────────────────────────────
_state: dict = {
    "chain":     None,   # callable(question, session_id) -> generator[str]
    "retriever": None,   # used separately to surface source docs
    "llm":       None,
    "corpus":    "None loaded",
}


# ══════════════════════════════════════════════════════════════════════════════
# Session history — manual per-session storage (no deprecated classes)
# ══════════════════════════════════════════════════════════════════════════════

from langchain_core.messages import AIMessage, HumanMessage

_sessions: dict[str, list] = {}  # session_id -> list of HumanMessage/AIMessage


def get_session_history(session_id: str) -> list:
    if session_id not in _sessions:
        _sessions[session_id] = []
    return _sessions[session_id]


# ══════════════════════════════════════════════════════════════════════════════
# LCEL chain
# ══════════════════════════════════════════════════════════════════════════════

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

_CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the conversation history and a follow-up question, rephrase the "
        "follow-up as a self-contained standalone question. "
        "If there is no history, return the question unchanged. "
        "Return ONLY the rephrased question — do not answer it.",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

_QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the question using ONLY the context "
        "provided below. If the answer is not in the context, say exactly: "
        "'I don't know based on the available documents.'\n\n"
        "Context:\n{context}",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def build_rag_chain(retriever, llm):
    """
    Pure LCEL pipeline — no deprecated classes.

    Returns a callable: invoke(question, session_id) -> generator[str]
    History is stored as LangChain HumanMessage/AIMessage objects per session.
    """
    condense_chain = _CONDENSE_PROMPT | llm | StrOutputParser()

    def _run(question: str, session_id: str):
        history = get_session_history(session_id)

        # Step 1: condense follow-up question if history exists
        if history:
            standalone = condense_chain.invoke({"input": question, "chat_history": history})
        else:
            standalone = question

        # Step 2: retrieve context
        docs = retriever.invoke(standalone)
        context = "\n\n---\n\n".join(d.page_content for d in docs)

        # Step 3: stream answer
        answer_chunks = []
        for chunk in (_QA_PROMPT | llm | StrOutputParser()).stream(
            {"input": question, "chat_history": history, "context": context}
        ):
            answer_chunks.append(chunk)
            yield chunk

        # Persist turn to history
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content="".join(answer_chunks)))

    return _run


# ── Client factories ───────────────────────────────────────────────────────────

def _make_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    class NomicEmbeddings(HuggingFaceEmbeddings):
        """HuggingFaceEmbeddings with nomic-embed-text-v1.5 task prefixes.

        nomic-embed-text-v1.5 was trained with mandatory task prefixes:
          - "search_document: " when embedding chunks for the vector store
          - "search_query: "    when embedding the user's question at retrieval time
        Using the wrong or no prefix measurably degrades retrieval quality.
        """

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return super().embed_documents(["search_document: " + t for t in texts])

        def embed_query(self, text: str) -> list[float]:
            return super().embed_query("search_query: " + text)

    return NomicEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"trust_remote_code": True},
        # nomic-bert-2048 pads all sequences in a batch to the longest one.
        # Large PDFs produce many 512-token chunks; the default batch_size=32
        # causes the attention matrix to exceed available RAM (15+ GiB).
        # batch_size=8 keeps peak memory well under 2 GiB per forward pass.
        encode_kwargs={"batch_size": 8},
    )


def _make_llm():
    from langchain_aws import ChatBedrock
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION, verify=certifi.where())
    return ChatBedrock(
        client=client,
        model_id=LLM_MODEL_ID,
        model_kwargs={"temperature": 0.0},
    )


def _as_mmr_retriever(vectordb):
    """Return an MMR retriever with the configured k / fetch_k / lambda."""
    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":           RETRIEVER_K,
            "fetch_k":     RETRIEVER_FETCH_K,
            "lambda_mult": MMR_LAMBDA,
        },
    )


def _wire_chain(retriever) -> None:
    """Attach a retriever to the global chain — shared by all load paths.

    Initialises the LLM on first call, then (re-)builds the RAG chain so that
    both startup loading and PDF uploads always use the same pipeline.
    """
    if _state["llm"] is None:
        _state["llm"] = _make_llm()
    _state["retriever"] = retriever
    _state["chain"]     = build_rag_chain(retriever, _state["llm"])


def _split_pdf(path: str):
    """Load and split a single PDF into token-sized chunks.

    Used by both the startup auto-index and the Gradio upload handler so that
    the exact same pipeline (loader + splitter settings) is always applied.
    """
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pages    = PyMuPDFLoader(path).load()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TIKTOKEN_ENCODING,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(pages)


def _initialise(persist_dir: str = DEFAULT_PERSIST) -> str:
    """Load or build the ChromaDB at startup.

    Fast path (subsequent runs): if nomic_chroma_db already exists on disk,
    load it directly — no re-indexing needed.

    First-run path: if the directory does not exist, scan SOURCE_DOCS_DIR for
    *.pdf files, index them all with the current pipeline, and persist the DB
    so the next run is instant.
    """
    from langchain_chroma import Chroma

    embeddings = _make_embeddings()

    # ── Fast path: DB already built ───────────────────────────────────────────
    if os.path.exists(persist_dir):
        try:
            vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            n = vectordb._collection.count()
            if n > 0:
                _wire_chain(_as_mmr_retriever(vectordb))
                _state["corpus"] = f"{os.path.basename(persist_dir)}/ ({n} chunks)"
                return f"✓ Loaded {_state['corpus']}"
        except Exception as e:
            msg = str(e)
            if "dimension" in msg.lower() or "embedding" in msg.lower():
                print(f"⚠ Dimension mismatch in '{persist_dir}' — rebuilding from source docs.")
            else:
                return f"⚠ Could not load ChromaDB: {e}"

    # ── First-run path: build from PDFs in SOURCE_DOCS_DIR ────────────────────
    pdf_files = [
        os.path.join(SOURCE_DOCS_DIR, f)
        for f in os.listdir(SOURCE_DOCS_DIR)
        if f.lower().endswith(".pdf")
    ] if os.path.isdir(SOURCE_DOCS_DIR) else []

    if not pdf_files:
        return (
            f"⚠ No PDFs found in '{SOURCE_DOCS_DIR}' and '{persist_dir}' does not exist. "
            "Upload a PDF in the Database tab to get started."
        )

    print(f"Building index from {len(pdf_files)} PDF(s) in '{SOURCE_DOCS_DIR}' …")
    all_splits = []
    for path in pdf_files:
        fname = os.path.basename(path)
        print(f"  → {fname}")
        all_splits.extend(_split_pdf(path))

    vectordb = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    n = vectordb._collection.count()
    _wire_chain(_as_mmr_retriever(vectordb))
    names   = ", ".join(os.path.basename(p) for p in pdf_files)
    _state["corpus"] = f"{names} ({n} chunks)"
    return f"✓ Built and persisted '{persist_dir}' from {len(pdf_files)} PDF(s) — {n} chunks"


# ══════════════════════════════════════════════════════════════════════════════
# Gradio event handlers
# ══════════════════════════════════════════════════════════════════════════════

def respond(message: str, chat_history: list, session_id: str):
    """Streaming generator — yields one updated chat_history per token.

    chat_history format: [{"role": "user"|"assistant", "content": str}, ...]
    """
    message = message.strip()
    if not message:
        yield chat_history, "", ""
        return

    if _state["chain"] is None:
        yield chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠ No documents loaded. Please upload a PDF in the **Database** tab."},
        ], "", ""
        return

    # Retrieve source docs for display (fast; retriever is not the bottleneck)
    try:
        src_docs = _state["retriever"].invoke(message)
    except Exception:
        src_docs = []

    sources_md = ""
    if src_docs:
        lines = []
        for d in src_docs:
            src     = d.metadata.get("source", "?").split("/")[-1]
            pg      = d.metadata.get("page", "?")
            snippet = d.page_content.strip().replace("\n", " ")[:110]
            lines.append(f"**[{src} | p{pg}]** {snippet}…")
        sources_md = "**Retrieved sources:**\n\n" + "\n\n".join(lines)

    # Append placeholder turn; update assistant content as tokens stream in
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "▌"},
    ]
    yield chat_history, "", sources_md

    full_answer = ""
    try:
        for chunk in _state["chain"](message, session_id):
            full_answer += chunk
            chat_history[-1]["content"] = full_answer + "◌"
            yield chat_history, "", sources_md
    except Exception as e:
        full_answer = f"⚠ Error: {e}"

    chat_history[-1]["content"] = full_answer
    yield chat_history, "", sources_md


def upload_pdf(file_obj, session_id: str):
    """Build a new in-memory vector store from an uploaded PDF.

    Uses the same _split_pdf pipeline (PyMuPDFLoader + tiktoken encoder) as the
    startup auto-index and wires the result into the chain via _wire_chain so
    all load paths are consistent.
    """
    if file_obj is None:
        return "No file selected.", _state["corpus"], []

    from langchain_chroma import Chroma

    try:
        splits = _split_pdf(file_obj)

        with tempfile.TemporaryDirectory() as tmp:
            vectordb = Chroma.from_documents(
                documents=splits,
                embedding=_make_embeddings(),
                persist_directory=tmp,
            )
            _wire_chain(_as_mmr_retriever(vectordb))

        fname            = os.path.basename(file_obj)
        _state["corpus"] = f"{fname} ({len(splits)} chunks)"

        # Clear this session's history since the corpus changed
        _sessions[session_id] = []

        return f"✓ Loaded '{fname}' — {len(splits)} chunks", _state["corpus"], []
    except Exception as e:
        return f"⚠ Error loading PDF: {e}", _state["corpus"], []


def refresh_history(session_id: str) -> str:
    msgs = get_session_history(session_id)
    if not msgs:
        return "*No messages yet. Start chatting in the Conversation tab.*"
    lines = []
    for msg in msgs:
        role = "🧑 **You**" if msg.type == "human" else "🤖 **Bot**"
        lines.append(f"{role}\n\n{msg.content}\n\n---")
    return "\n".join(lines)


def reset_session(session_id: str):
    _sessions[session_id] = []
    return [], "", "*History cleared.*", f"✓ Session {session_id[:8]}\u2026 cleared"


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

import gradio as gr

print("\nInitialising …")
startup_msg = _initialise()
print(startup_msg)

with gr.Blocks(
    title="ChatWithYourData_Bot",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css="footer { display: none !important; }",
) as demo:

    # Per-session state — unique ID per browser connection
    session_id_state = gr.State(value=lambda: str(uuid.uuid4()))

    gr.Markdown(
        "## 🤖 ChatWithYourData_Bot\n"
        "*Powered by AWS Bedrock Claude · HuggingFace Embeddings · LangChain LCEL*"
    )

    with gr.Tabs():

        # ── Tab 1: Conversation ────────────────────────────────────────────────
        with gr.Tab("💬 Conversation"):
            chatbot = gr.Chatbot(
                label="",
                height=460,
                show_label=False,
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Enter text here…",
                    label="",
                    lines=1,
                    scale=5,
                    show_label=False,
                    submit_btn=False,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=90)

            sources_box = gr.Markdown(
                value="*Retrieved sources will appear here after each answer.*"
            )

        # ── Tab 2: Chat History ────────────────────────────────────────────────
        with gr.Tab("📜 Chat History"):
            history_md  = gr.Markdown(value="*No messages yet.*")
            refresh_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")

        # ── Tab 3: Database ────────────────────────────────────────────────────
        with gr.Tab("🗄 Database"):
            gr.Markdown("### Current corpus")
            corpus_info = gr.Textbox(
                value=_state["corpus"],
                label="Loaded corpus",
                interactive=False,
            )
            gr.Markdown(
                "### Upload a new PDF\n"
                "*Uploading replaces the current corpus and clears your conversation history.*"
            )
            pdf_upload    = gr.File(label="Select PDF", file_types=[".pdf"])
            upload_btn    = gr.Button("Load PDF", variant="primary")
            upload_status = gr.Textbox(label="Status", interactive=False)

        # ── Tab 4: Configure ───────────────────────────────────────────────────
        with gr.Tab("⚙️ Configure"):
            gr.Markdown("### Session")
            session_display = gr.Textbox(
                label="Session ID",
                interactive=False,
                info="Each browser session has its own isolated conversation history.",
            )
            clear_btn    = gr.Button("🗑 Clear conversation", variant="stop", size="sm")
            clear_status = gr.Textbox(label="", interactive=False, show_label=False)

            gr.Markdown("""
### Chain details
| Component | Value |
|---|---|
| LLM | AWS Bedrock Claude Sonnet 4.6 |
| Embeddings | HuggingFace `nomic-embed-text-v1.5` (fully local, 8192-token context) |
| Vector store | ChromaDB |
| PDF loader | PyMuPDF — preserves tables, multi-column layouts |
| Chunking | `RecursiveCharacterTextSplitter.from_tiktoken_encoder` — `cl100k_base`, 512 tokens / 64 overlap |
| History | Plain `list[HumanMessage \\| AIMessage]` — one per session |
| Chain | LCEL pipeline with manual history management |
| Retrieval | MMR top-`k=3`, `fetch_k=12`, `lambda=0.7` — relevant + diverse chunks |

### How it works
1. Your question is condensed with chat history into a standalone question  
2. The standalone question retrieves the top-3 document chunks  
3. The LLM streams an answer using those chunks as context  
4. The Q&A pair is appended to your session history for the next turn  
""")

    # ── Wire events ────────────────────────────────────────────────────────────

    send_inputs  = [msg_box, chatbot, session_id_state]
    send_outputs = [chatbot, msg_box, sources_box]

    send_btn.click(fn=respond, inputs=send_inputs, outputs=send_outputs)
    msg_box.submit(fn=respond, inputs=send_inputs, outputs=send_outputs)

    refresh_btn.click(fn=refresh_history, inputs=[session_id_state], outputs=[history_md])

    upload_btn.click(
        fn=upload_pdf,
        inputs=[pdf_upload, session_id_state],
        outputs=[upload_status, corpus_info, chatbot],
    )

    # Show session ID once on page load
    demo.load(fn=lambda sid: sid, inputs=[session_id_state], outputs=[session_display])

    clear_btn.click(
        fn=reset_session,
        inputs=[session_id_state],
        outputs=[chatbot, sources_box, history_md, clear_status],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
