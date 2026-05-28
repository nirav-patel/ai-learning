"""
Conversational Chat with Memory
=================================
L6 of LangChain: Chat with Your Data.
Builds on l5_question_answering.py by adding conversational memory.

KEY UPGRADE OVER l5
───────────────────
  l5 used RetrievalQA — completely stateless:
    turn 1: "what is the sick leave policy?"       → good answer
    turn 2: "what happens if they run out?"         → confused / no context

  l6 uses ConversationalRetrievalChain — stateful:
    The chain first condenses (question + history) into a standalone question,
    then retrieves relevant chunks, then answers.
    Follow-up questions referencing prior context now work correctly.

HOW ConversationalRetrievalChain WORKS INTERNALLY
───────────────────────────────────────────────────
  Input: new question + stored chat_history
      ↓
  Step 1 — Question Condenser (LLM call):
    Combines history + new question → standalone question
    e.g. "what happens if they run out?" + history about sick leave
         → "what happens when sick leave days are exhausted?"
      ↓
  Step 2 — Retriever:
    Retrieves relevant chunks using the standalone question
      ↓
  Step 3 — QA Chain (LLM call):
    Answers the standalone question using the retrieved context
      ↓
  Output: answer + updated memory

MEMORY TYPES COVERED
─────────────────────
  1. ConversationBufferMemory        — keeps ALL messages verbatim (simplest)
  2. ConversationBufferWindowMemory  — keeps only last k exchanges (token-safe)
  3. ConversationSummaryMemory       — LLM summarises old messages (most scalable)

  Trade-offs:
    Buffer    : full fidelity, grows unboundedly → context window risk
    Window    : bounded size, may forget distant context
    Summary   : bounded size, preserves gist, costs extra LLM calls to summarise

RUN
───
    uv run python l6_chat.py              # section-by-section demo
    uv run python l6_chat_ui.py           # Gradio web chat interface
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

import boto3
import certifi

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress LangChainDeprecationWarning
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

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
    os.environ.setdefault("LANGCHAIN_PROJECT", "langchain-chat-with-data")
    print(f"  LangSmith tracing ENABLED  (project: {os.environ['LANGCHAIN_PROJECT']})")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("  LangSmith tracing DISABLED (set LANGCHAIN_API_KEY in .env to enable)")

# ── Configuration ─────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_ID     = "us.anthropic.claude-sonnet-4-6"
AWS_REGION       = os.getenv("AWS_REGION", "us-west-2")
PERSIST_DIR      = "sample_chroma_db"


# ── Factories ──────────────────────────────────────────────────────────────────

def _make_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


def _make_llm():
    from langchain_aws import ChatBedrock
    client = boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        verify=certifi.where(),
    )
    return ChatBedrock(
        client=client,
        model_id=LLM_MODEL_ID,
        model_kwargs={"temperature": 0.0},
    )


def build_chain(memory, retriever, llm, combine_docs_prompt=None):
    """
    Build a ConversationalRetrievalChain with the given memory type.

    When return_source_documents=True the chain has two outputs (answer +
    source_documents), so output_key must be set explicitly so memory knows
    which output to store as the assistant turn.
    """
    kwargs: dict = dict(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
        verbose=False,
    )
    if combine_docs_prompt:
        kwargs["combine_docs_chain_kwargs"] = {"prompt": combine_docs_prompt}

    return ConversationalRetrievalChain.from_llm(**kwargs)


def _ask(chain, question: str, show_sources: bool = True) -> str:
    result = chain.invoke({"question": question})
    answer = result.get("answer", "")
    print(f"\n  Q: {question!r}")
    print(f"  A: {answer}")
    if show_sources:
        for d in result.get("source_documents", []):
            src = d.metadata.get("source", "?").split("/")[-1]
            pg  = d.metadata.get("page", "?")
            print(f"     ↳ [{src} | p{pg}]")
    return answer


# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SETUP")
print("=" * 60)

import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
)

embeddings = _make_embeddings()

if os.path.exists(PERSIST_DIR):
    print(f"\n  Reloading ChromaDB from '{PERSIST_DIR}' …")
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
else:
    print(f"\n  Building ChromaDB (run l3 first for best results) …")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    pages: list = []
    for pdf_path in sorted(glob.glob("docs/*.pdf")):
        pages.extend(PyMuPDFLoader(pdf_path).load())
    splits = splitter.split_documents(pages)
    vectordb = Chroma.from_documents(
        documents=splits, embedding=embeddings, persist_directory=PERSIST_DIR
    )

print(f"  Vector store chunks : {vectordb._collection.count()}")
print(f"\n  Initialising LLM …")
llm = _make_llm()
print(f"  Model               : {LLM_MODEL_ID}")

retriever = vectordb.as_retriever(search_kwargs={"k": 3})


# ══════════════════════════════════════════════════════════════════════════════
# 1. WHY l5 FAILED — and how l6 fixes it
#
#    Reproduce the exact 2-question sequence from l5 limitations demo.
#    This time the follow-up references prior context correctly.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. ConversationBufferMemory — fixing l5 stateless limitation")
print("=" * 60)
print("""
  ConversationBufferMemory stores the full message history verbatim.
  Every turn is kept in memory and injected into the condenser prompt.
  Best for: short conversations where full history fits the context window.
""")

memory_buf = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)
qa_buf = build_chain(memory_buf, retriever, llm)

_ask(qa_buf, "What is an embedding model and how does it work?")
_ask(qa_buf, "How does it differ from traditional word2vec embeddings?")        # ← follow-up now works
_ask(qa_buf, "What are the best open source embedding models available today?")

# Inspect the stored memory
turns = memory_buf.chat_memory.messages
print(f"\n  Memory contains {len(turns)} messages ({len(turns)//2} full turns)")
for i, msg in enumerate(turns):
    role = "Human" if i % 2 == 0 else "AI"
    print(f"  [{role}] {str(msg.content)[:80]!r}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ConversationBufferWindowMemory (k=2)
#
#    Keeps only the last k human+AI exchange pairs.
#    Older context is silently dropped — prevents unbounded growth.
#    Best for: long conversations where distant context is less relevant.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. ConversationBufferWindowMemory  (k=2 — last 2 turns only)")
print("=" * 60)
print("""
  Only the most recent k=2 exchanges are kept.
  Turn 1 is forgotten once turn 3 is answered.
  Best for: long sessions where only recent context matters.
""")

memory_win = ConversationBufferWindowMemory(
    k=2,
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)
qa_win = build_chain(memory_win, retriever, llm)

_ask(qa_win, "What is a vector database and how does it enable semantic search?", show_sources=False)
_ask(qa_win, "What makes it different from a traditional relational database?", show_sources=False)
_ask(qa_win, "Which open source vector databases would you recommend?", show_sources=False)

turns_win = memory_win.chat_memory.messages
print(f"\n  Memory window: {len(turns_win)} messages stored (max = k×2 = {2*2})")
print("  (Earliest turn was dropped once the window was full)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ConversationSummaryMemory
#
#    Instead of storing raw messages, an LLM continuously summarises the
#    conversation into a running summary.  The summary replaces the full
#    history in the prompt — keeping token usage bounded even over many turns.
#
#    Cost: 1 extra LLM call per turn to update the summary.
#    Best for: very long conversations; when context gist > verbatim history.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. ConversationSummaryMemory  (LLM-summarised history)")
print("=" * 60)
print("""
  History is collapsed into a running summary by the LLM after each turn.
  The prompt always receives the summary rather than the raw messages.
  Token usage stays bounded regardless of conversation length.
""")

memory_sum = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)
qa_sum = build_chain(memory_sum, retriever, llm)

_ask(qa_sum, "What is RAG and how does it work?", show_sources=False)
_ask(qa_sum, "What are the main components needed to build one?", show_sources=False)
_ask(qa_sum, "How should I choose the right chunk size for my documents?", show_sources=False)

print(f"\n  Summary memory buffer:\n  {memory_sum.buffer!r}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Custom Prompts
#
#    ConversationalRetrievalChain has TWO internal prompts you can override:
#
#    a) condense_question_prompt — rewrites (history + new_question) into a
#       standalone question used for retrieval.
#       Inputs: {chat_history}, {question}
#
#    b) combine_docs_chain prompt — the final QA prompt.
#       Inputs: {context}, {question}
#       Passed via combine_docs_chain_kwargs={"prompt": ...}
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. Custom Prompts — condenser + QA template")
print("=" * 60)

from langchain_core.prompts import PromptTemplate

CONDENSE_TEMPLATE = """Given the conversation history and a follow-up question,
rephrase the follow-up into a self-contained standalone question.
If there is no history, return the question unchanged.

Chat History:
{chat_history}

Follow-up Question: {question}
Standalone Question:"""

CONDENSE_PROMPT = PromptTemplate.from_template(CONDENSE_TEMPLATE)

QA_TEMPLATE = """You are a helpful HR assistant. Use only the context below.
If the answer is not in the context, say "I don't know based on the documents."
Be concise (3 sentences max). End every answer with "Hope that helps!".

Context:
{context}

Question: {question}
Answer:"""

QA_PROMPT = PromptTemplate.from_template(QA_TEMPLATE)

memory_custom = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)
qa_custom = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory_custom,
    return_source_documents=True,
    output_key="answer",
    condense_question_prompt=CONDENSE_PROMPT,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    verbose=False,
)

_ask(qa_custom, "What is the difference between fine-tuning and RAG?", show_sources=True)
_ask(qa_custom, "Which approach is better suited for enterprise applications?", show_sources=False)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Interactive Terminal Chat
#
#    Type questions interactively.  The chain maintains full conversation
#    history across turns.  Type 'exit', 'quit', or press Ctrl+C to stop.
#    Type 'reset' to clear memory and start a fresh conversation.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. INTERACTIVE CHAT  (type 'exit' to quit, 'reset' to clear memory)")
print("=" * 60)

memory_interactive = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)
qa_interactive = build_chain(memory_interactive, retriever, llm)

print("\n  Chat is ready. Ask anything about the documents.\n")

def _show_history(memory) -> None:
    msgs = memory.chat_memory.messages
    if not msgs:
        print("  (no history yet)")
        return
    for i, msg in enumerate(msgs):
        role = "You" if i % 2 == 0 else "Bot"
        print(f"  [{role}] {str(msg.content)[:100]}")

try:
    while True:
        user_input = input("  You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("  Goodbye!")
            break
        if user_input.lower() == "reset":
            memory_interactive.clear()
            print("  Memory cleared. Starting fresh.\n")
            continue
        if user_input.lower() == "history":
            _show_history(memory_interactive)
            continue

        result = qa_interactive.invoke({"question": user_input})
        answer = result.get("answer", "")
        print(f"\n  Bot: {answer}")
        src_docs = result.get("source_documents", [])
        if src_docs:
            srcs = {
                f"{d.metadata.get('source','?').split('/')[-1]} p{d.metadata.get('page','?')}"
                for d in src_docs
            }
            print(f"  Sources: {', '.join(srcs)}\n")
        else:
            print()

except KeyboardInterrupt:
    print("\n  Interrupted. Goodbye!")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY — Memory type selection guide")
print("=" * 60)
print("""
  Memory type                   Stores         Token usage   Best for
  ─────────────────────────────────────────────────────────────────────
  ConversationBufferMemory      All messages   Grows w/ conv  Short chats
  ConversationBufferWindowMemory Last k turns  Bounded        Long chats (recent context)
  ConversationSummaryMemory     LLM summary    Bounded        Long chats (full gist)
  ConversationTokenBufferMemory Token-capped   Bounded        Token-sensitive deployments

  All types work with ConversationalRetrievalChain via memory_key="chat_history".
  Set output_key="answer" when return_source_documents=True.

  Next: l6_chat_ui.py — Gradio web interface for the same chain.
""")
