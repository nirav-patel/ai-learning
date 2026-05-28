"""
Question Answering Demo - LangChain Chat with Data
====================================================
Based on DeepLearning.AI "LangChain: Chat with Your Data" L5 notebook.
Uses HuggingFaceEmbeddings (local) + AWS Bedrock Claude.

WHAT THIS COVERS
────────────────
  RetrievalQA is LangChain's main RAG chain.  It wires together:
    retriever  → fetches relevant chunks from the vector store
    LLM        → reads those chunks and answers the question

  CHAIN TYPES (how retrieved docs are fed to the LLM)
  ─────────────────────────────────────────────────────
  1. stuff      (default) — all docs stuffed into one prompt
                            Fast.  Fails if total text > context window.
  2. map_reduce           — each doc answered independently (map),
                            then answers are combined (reduce).
                            Good for many/large docs.  More LLM calls.
  3. refine               — answer built iteratively: start with first doc,
                            then refine with each subsequent doc.
                            Most thorough but slowest.
  4. map_rerank           — each doc gets an answer + confidence score,
                            highest-scored answer is returned.
                            Good when one chunk clearly contains the answer.

  LANGSMITH TRACING
  ─────────────────
  Set LANGCHAIN_API_KEY in .env to enable tracing.
  All chain invocations are then visible in the LangSmith UI.

  LIMITATIONS DEMO
  ─────────────────
  RetrievalQA has no memory — each call is stateless.
  Follow-up questions that refer to prior context fail.
  Fix: ConversationalRetrievalChain (covered in l6).

SETUP REUSE
───────────
  Loads the persistent ChromaDB built by l3_vectorstores_and_embeddings.py.
  Run l3 first if sample_chroma_db/ does not exist.

RUN
───
    cd langchain-chat-with-data
    uv run python l5_question_answering.py

LANGSMITH SETUP
───────────────
  Add these to your .env file (API key provided separately):
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_ENDPOINT=https://api.smith.langchain.com   # US (default)
    # LANGCHAIN_ENDPOINT=https://apac.api.smith.langchain.com  # APAC region
    # LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com    # EU region
    LANGCHAIN_API_KEY=<your-key-here>
    LANGCHAIN_PROJECT=langchain-chat-with-data   # optional project label

  NOTE: Use the region-specific endpoint that matches where your LangSmith
  account was provisioned — using the wrong region returns 403 Forbidden.
"""

from __future__ import annotations

import logging
import os
import sys
import textwrap
import warnings

import boto3
import certifi

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.append("..")
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# ── Configuration ─────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_ID     = "us.anthropic.claude-sonnet-4-6"
AWS_REGION       = os.getenv("AWS_REGION", "us-west-2")
PERSIST_DIR      = "sample_chroma_db"
DOCS_DIR         = "docs"
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 150


# ── LangSmith tracing setup ───────────────────────────────────────────────────
# Reads LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY / LANGCHAIN_ENDPOINT from .env
# If the key is not set, tracing is silently disabled.
_ls_key = os.getenv("LANGCHAIN_API_KEY", "")
if _ls_key:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://apac.api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_PROJECT", "langchain-chat-with-data")
    print(f"  LangSmith tracing ENABLED  (project: {os.environ['LANGCHAIN_PROJECT']})")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("  LangSmith tracing DISABLED (set LANGCHAIN_API_KEY in .env to enable)")


# ── Client factories ───────────────────────────────────────────────────────────

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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_result(result: dict, label: str) -> None:
    answer = result.get("result", result.get("answer", ""))
    print(f"\n  Answer : {answer}")
    src_docs = result.get("source_documents", [])
    if src_docs:
        print(f"  Sources ({len(src_docs)}):")
        for d in src_docs:
            src = d.metadata.get("source", "?").split("/")[-1]
            pg  = d.metadata.get("page", "?")
            print(f"    • [{src} | p{pg}]  {d.page_content.strip().replace(chr(10),' ')[:80]!r}")


def _run_qa(chain, question: str, label: str) -> dict:
    print(f"\n  Q: {question!r}")
    result = chain.invoke({"query": question})
    _print_result(result, label)
    return result


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

embeddings = _make_embeddings()

if os.path.exists(PERSIST_DIR):
    print(f"\n  Reloading ChromaDB from '{PERSIST_DIR}' …")
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
else:
    print(f"\n  '{PERSIST_DIR}' not found — building ChromaDB …")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_pages: list = []
    for pdf_path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf"))):
        all_pages.extend(PyMuPDFLoader(pdf_path).load())
    splits = splitter.split_documents(all_pages)
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )

print(f"  Vector store chunks : {vectordb._collection.count()}")
print(f"\n  Initialising LLM (Bedrock Claude) …")
llm = _make_llm()
print(f"  LLM model           : {LLM_MODEL_ID}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. RetrievalQA — default chain_type="stuff"
#
#    All retrieved docs are "stuffed" into a single prompt as {context}.
#    Simple, fast, one LLM call.  Works well when total context is small.
#
#    Prompt template used internally by "stuff":
#      Use the following pieces of context to answer the question...
#      {context}
#      Question: {question}
#      Helpful Answer:
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. RetrievalQA — chain_type='stuff'  (default)")
print("=" * 60)

from langchain_classic.chains import RetrievalQA

qa_stuff = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    # chain_type="stuff" is the default — shown explicitly for clarity
    chain_type="stuff",
)

q1 = "What is an embedding model and how does it work?"
_run_qa(qa_stuff, q1, "stuff")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Custom Prompt Template
#
#    Override the default prompt to control tone, format, and guardrails.
#    The template must contain {context} and {question} placeholders.
#    Pass it via chain_type_kwargs={"prompt": <PromptTemplate>}.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. RetrievalQA — stuff + Custom Prompt")
print("=" * 60)

from langchain_core.prompts import PromptTemplate

CUSTOM_TEMPLATE = """You are an HR assistant. Use only the context below to answer.
If the answer is not in the context, say "I don't know — the documents don't cover this."
Be concise (3 sentences max). Always end with "Hope that helps!".

Context:
{context}

Question: {question}
Answer:"""

QA_PROMPT = PromptTemplate.from_template(CUSTOM_TEMPLATE)

qa_custom = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_PROMPT},
)

q2 = "What is the difference between a vector database and embeddings?"
_run_qa(qa_custom, q2, "stuff + custom prompt")


# ══════════════════════════════════════════════════════════════════════════════
# 3. chain_type="map_reduce"
#
#    HOW IT WORKS:
#      MAP   — for each retrieved doc, run the LLM independently to extract
#              a partial answer relevant to the question.
#      REDUCE — combine all partial answers in a final LLM call.
#
#    PROS: handles more documents than fit in one context window;
#          parallelisable across docs.
#    CONS: multiple LLM calls (1 per doc + 1 reduce); slower and costlier.
#    USE WHEN: you have many docs or very large chunks.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. RetrievalQA — chain_type='map_reduce'")
print("=" * 60)

qa_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type="map_reduce",
)

_run_qa(qa_mr, q2, "map_reduce")

# Show that map_reduce handles a broader question well
q3 = "What are the main steps to build a RAG (Retrieval Augmented Generation) pipeline?"
_run_qa(qa_mr, q3, "map_reduce — broad question")


# ══════════════════════════════════════════════════════════════════════════════
# 4. chain_type="refine"
#
#    HOW IT WORKS:
#      1. Generate an initial answer from the first doc.
#      2. For each subsequent doc, refine the answer using:
#           "Here is the existing answer: ...
#            Here is more context: ...
#            Refine the answer if needed."
#
#    PROS: most thorough — every doc contributes.
#          answer quality improves with more docs.
#    CONS: sequential (cannot parallelise); most LLM calls.
#    USE WHEN: answer quality is critical and latency is acceptable.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. RetrievalQA — chain_type='refine'")
print("=" * 60)

qa_refine = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type="refine",
)

_run_qa(qa_refine, q2, "refine")


# ══════════════════════════════════════════════════════════════════════════════
# 5. chain_type="map_rerank"
#
#    HOW IT WORKS:
#      MAP    — for each doc, the LLM produces an answer AND a relevance score.
#      RERANK — the answer with the highest score is returned.
#
#    PROS: transparent scoring; good when one chunk clearly contains the answer.
#    CONS: only returns one doc's answer (may miss synthesis across docs).
#    USE WHEN: you expect the answer to be in a single chunk.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. RetrievalQA — chain_type='map_rerank'")
print("=" * 60)

qa_rerank = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type="map_rerank",
)

_run_qa(qa_rerank, q2, "map_rerank")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Chain Type Comparison
#    Same question, all four chain types — side-by-side results.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. CHAIN TYPE COMPARISON — same question, all four types")
print("=" * 60)

comparison_q = "What are the key differences between RAG, fine-tuning, and prompt engineering? When should you use each approach?"  # broad question to show chain_type differences
print(f"\n  Question: {comparison_q!r}\n")

chains = [
    ("stuff      ", qa_stuff),
    ("map_reduce ", qa_mr),
    ("refine     ", qa_refine),
    ("map_rerank ", qa_rerank),
]

for label, chain in chains:
    result = chain.invoke({"query": comparison_q})
    answer = result.get("result", "")
    n_src  = len(result.get("source_documents", []))
    print(f"  [{label}]  sources={n_src}  answer: {answer[:200]!r}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. LIMITATIONS — RetrievalQA has no conversational memory
#
#    Each call is completely stateless.
#    A follow-up like "why are those needed?" gets no context from the
#    previous answer — the LLM has to guess what "those" refers to.
#
#    Fix: ConversationalRetrievalChain (l6) — wraps RetrievalQA with memory.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. LIMITATIONS — no conversational memory")
print("=" * 60)

qa_stateless = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
)

print("\n  Turn 1:")
r1 = qa_stateless.invoke({"query": "What is an embedding model and what role does it play in LLMs?"})
print(f"  Q: 'What is an embedding model and what role does it play in LLMs?'")
print(f"  A: {r1['result']}")

print("\n  Turn 2 (follow-up — expects context from turn 1):")
r2 = qa_stateless.invoke({"query": "What are the best open source options for it?"})
print(f"  Q: 'What are the best open source options for it?'")
print(f"  A: {r2['result']}")
print(f"\n  NOTE: turn 2 answer may be vague or unrelated because RetrievalQA")
print(f"        does not carry conversation history between calls.")
print(f"        Use ConversationalRetrievalChain (l6) to fix this.")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY — chain_type selection guide")
print("=" * 60)
print(textwrap.dedent("""
  chain_type    LLM calls  Parallelisable  Best for
  ──────────────────────────────────────────────────────────────────────
  stuff         1          —               Small # of short chunks (default)
  map_reduce    N+1        yes (map step)  Many / large chunks; speed matters
  refine        N          no (sequential) Best answer quality; latency OK
  map_rerank    N          yes             Single chunk expected to hold answer

  LangSmith tracing: set LANGCHAIN_API_KEY in .env to trace all chain calls
  at https://smith.langchain.com

  Next: l6 — ConversationalRetrievalChain adds memory to fix the stateless
        limitation shown in section 7.
"""))
