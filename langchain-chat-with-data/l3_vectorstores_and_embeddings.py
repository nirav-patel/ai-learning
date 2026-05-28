"""
Vector Stores & Embeddings Demo - LangChain Chat with Data
===========================================================
Based on DeepLearning.AI "LangChain: Chat with Your Data" L3 notebook.
Uses HuggingFaceEmbeddings (all-MiniLM-L6-v2, runs locally) instead of OpenAI.

WHAT ARE EMBEDDINGS?
────────────────────
An embedding maps a piece of text to a fixed-length vector of floats.
Semantically similar texts produce vectors that are "close" in vector space
(measured by cosine similarity or dot product).

Example:
  "I love dogs"  → [0.12, -0.34, 0.87, …]   ← close to each other
  "I love pets"  → [0.11, -0.31, 0.89, …]
  "The sky is blue" → [-0.52, 0.91, -0.04, …]  ← far from both

WHY VECTOR STORES?
────────────────────
Brute-force similarity over millions of vectors is slow.  A vector store
(like ChromaDB) indexes embeddings so nearest-neighbour lookup is fast
(milliseconds instead of seconds).

PIPELINE OVERVIEW
─────────────────
  PDF files
      │
      ▼  PyMuPDFLoader
  raw Documents (one per page)
      │
      ▼  RecursiveCharacterTextSplitter
  chunked Documents
      │
      ▼  HuggingFaceEmbeddings (all-MiniLM-L6-v2, local — no API calls)
  float vectors
      │
      ▼  ChromaDB (in-memory + persistent)
  vector store
      │
      ▼  similarity_search / MMR / score queries
  ranked result Documents

DEMOS
─────
  1. Embeddings basics            — cosine similarity between sentences
  2. Load all PDFs from docs/     — India Handbook + AI Career Roadmap
  3. Duplicate chunk failure mode — India Handbook loaded twice → 3 identical results
  4. MMR search                   — diversity fix for duplicates
  5. Similarity search with score — see cosine distances
  6. Metadata filtering           — query only one source document
  7. Persistent ChromaDB          — save to disk, reload, and re-query

RUN
───
    cd langchain-chat-with-data
    uv run python l3_vectorstores_and_embeddings.py

DEPENDENCIES
────────────
    chromadb, langchain-text-splitters, langchain-huggingface,
    sentence-transformers, pypdf
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import sys
import textwrap
import warnings

# Suppress noisy HuggingFace/tokenizer output before importing them
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
EMBED_MODEL_NAME  = "all-MiniLM-L6-v2"     # local model, no API calls needed
DOCS_DIR          = "docs"
PERSIST_DIR       = "sample_chroma_db"          # written next to this script
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 150


def _make_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


def _show_docs(docs: list, label: str = "") -> None:
    if label:
        print(f"\n  [{label}]")
    print(f"  Results: {len(docs)}")
    for i, d in enumerate(docs):
        src = d.metadata.get("source", "?").split("/")[-1]
        pg  = d.metadata.get("page", "?")
        snippet = d.page_content[:120].replace("\n", " ")
        print(f"  {i+1}. [{src} | page {pg}]  {snippet!r}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. EMBEDDINGS BASICS
#    Embed three sentences and compare cosine similarities.
#    Demonstrates that semantically similar text → similar vectors.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. EMBEDDINGS BASICS")
print("=" * 60)

embeddings = _make_embeddings()

sentence1 = "i love dogs"
sentence2 = "i love canines"       # near-synonym → high similarity
sentence3 = "the stock market dropped significantly today"   # unrelated → low

print(f"  Embedding 3 sentences via {EMBED_MODEL_NAME} (local HuggingFace) …")
e1, e2, e3 = embeddings.embed_documents([sentence1, sentence2, sentence3])

print(f"  Vector dimension : {len(e1)}")
print(f"\n  Cosine similarity:")
print(f"    '{sentence1}' ↔ '{sentence2}' : {_cosine(e1, e2):.4f}  (expected HIGH)")
print(f"    '{sentence1}' ↔ '{sentence3}' : {_cosine(e1, e3):.4f}  (expected LOW)")
print(f"    '{sentence2}' ↔ '{sentence3}' : {_cosine(e2, e3):.4f}  (expected LOW)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD ALL PDFs + SPLIT
#    Load every PDF under docs/ using PyMuPDFLoader.
#    Split all pages into overlapping chunks ready for embedding.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. LOAD & SPLIT ALL PDFs")
print("=" * 60)

import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))
print(f"\n  PDFs found: {len(pdf_files)}")
for p in pdf_files:
    print(f"    • {os.path.basename(p)}")

all_pages: list = []
for pdf_path in pdf_files:
    loader = PyMuPDFLoader(pdf_path)
    pages  = loader.load()
    all_pages.extend(pages)
    print(f"  Loaded {len(pages):3d} pages  ← {os.path.basename(pdf_path)}")

print(f"\n  Total pages : {len(all_pages)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
splits = splitter.split_documents(all_pages)
print(f"  Total chunks: {len(splits)}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD IN-MEMORY CHROMADB
#    Embed all chunks and store in ChromaDB (no persistence yet).
#    This is the simplest setup for experimenting.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. BUILD IN-MEMORY CHROMADB (normal — no duplicates)")
print("=" * 60)

from langchain_chroma import Chroma

print(f"\n  Embedding {len(splits)} chunks and loading into Chroma …")
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
)
print(f"  Collection count: {vectordb._collection.count()}")


# ── Basic similarity search ───────────────────────────────────────────────────
print("\n  --- Basic similarity_search ---")
query = "What is an embedding model and how does it work?"
results = vectordb.similarity_search(query, k=3)
_show_docs(results, f"query='{query}'  k=3")

print()
query2 = "What is the difference between a vector database and embeddings?"
results2 = vectordb.similarity_search(query2, k=3)
_show_docs(results2, f"query='{query2}'  k=3")

print()
query3 = "What are the main components needed to build an LLM pipeline?"
results3 = vectordb.similarity_search(query3, k=5)
_show_docs(results3, f"query='{query3}'  k=5")


# ══════════════════════════════════════════════════════════════════════════════
# 4. FAILURE MODE: DUPLICATE CHUNKS
#    Reload India-Handbook TWICE and add it to a new vector store.
#    A similarity search returns 3 nearly-identical chunks — all from the same
#    document — crowding out results from the other PDF.
#
#    ROOT CAUSE: cosine similarity ranks redundant copies just as highly as
#    diverse, informative chunks.  The top-k list becomes homogeneous.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. FAILURE MODE — Duplicate Chunks")
print("=" * 60)

HANDBOOK_PDF = os.path.join(DOCS_DIR, "India-Handbook-2024.pdf")
handbook_pages = PyMuPDFLoader(HANDBOOK_PDF).load()
handbook_splits = splitter.split_documents(handbook_pages)

# Build a corpus that contains the handbook TWICE + the AI roadmap once
# Build a corpus that contains the handbook TWICE + a synthetic "second doc"
# (uses the last 10 pages of the handbook with a different source label to simulate
#  a second distinct PDF when only one PDF is present in docs/)
if len(pdf_files) > 1:
    roadmap_pdf    = [p for p in pdf_files if p != HANDBOOK_PDF][0]
    roadmap_splits = splitter.split_documents(PyMuPDFLoader(roadmap_pdf).load())
else:
    # Simulate a second document using later pages of the handbook
    from copy import deepcopy
    second_doc_pages = PyMuPDFLoader(HANDBOOK_PDF).load()[-10:]
    roadmap_splits = splitter.split_documents(second_doc_pages)
    for d in roadmap_splits:
        d.metadata = dict(d.metadata)
        d.metadata["source"] = os.path.join(DOCS_DIR, "handbook-part2-synthetic.pdf")
    print("  (Only 1 PDF found — using last 10 pages as a synthetic second source)")


dup_splits = handbook_splits + handbook_splits + roadmap_splits   # handbook duplicated
print(f"\n  Corpus: handbook×2 ({len(handbook_splits)*2}) + roadmap×1 ({len(roadmap_splits)})"
      f" = {len(dup_splits)} total chunks")

print("  Embedding duplicate corpus …")
vectordb_dup = Chroma.from_documents(
    documents=dup_splits,
    embedding=embeddings,
)

query_dup = "what are the leave policies in India?"
dup_results = vectordb_dup.similarity_search(query_dup, k=3)
_show_docs(dup_results, f"Duplicate corpus — '{query_dup}'  k=3")

# Count how many unique sources are represented
sources_dup = {d.metadata.get("source", "?").split("/")[-1] for d in dup_results}
print(f"\n  Unique sources in top-3: {sources_dup}")
print("  → All 3 results are from the same doc!  Diversity is lost.")


# ══════════════════════════════════════════════════════════════════════════════
# 5. FIX: MMR (Maximum Marginal Relevance) Search
#    MMR balances relevance with diversity.  It iteratively selects chunks
#    that are relevant to the query BUT different from already-selected chunks.
#
#    Parameters:
#      k          — number of results to return
#      fetch_k    — candidates to consider before applying diversity filter
#      lambda_mult— 0 = max diversity, 1 = max relevance (default 0.5)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. FIX — MMR (Maximum Marginal Relevance) Search")
print("=" * 60)

VERIFY_CHARS = 200   # characters to show per chunk for diversity verification

def _show_docs_full(docs: list, label: str = "") -> None:
    """Print the first 2 chunks (VERIFY_CHARS each) for diversity verification."""
    if label:
        print(f"\n  ┌─ {label}")
    print(f"  │  Results: {len(docs)}")
    for i, d in enumerate(docs[:2]):
        src     = d.metadata.get("source", "?").split("/")[-1]
        pg      = d.metadata.get("page", "?")
        snippet = d.page_content.strip().replace("\n", " ")[:VERIFY_CHARS]
        print(f"\n  ── Result {i+1} [{src} | page {pg}]")
        print(f"  │  {snippet!r}")


# ── Similarity search on duplicate corpus (baseline — shows the problem) ──────
print(f"\n  Similarity search on DUPLICATE corpus (baseline):")
sim_dup_results = vectordb_dup.similarity_search(query_dup, k=3)
_show_docs_full(sim_dup_results, f"similarity_search — '{query_dup}'  k=3")

sim_sources = [d.metadata.get("source", "?").split("/")[-1] + f" p{d.metadata.get('page','?')}"
               for d in sim_dup_results]
print(f"\n  Sources returned : {sim_sources}")
all_same = len(set(sim_sources)) == 1
print(f"  All identical?   : {all_same}  ← {'YES — diversity lost!' if all_same else 'NO'}")

# ── MMR on duplicate corpus (the fix) ─────────────────────────────────────────
print(f"\n  MMR search on DUPLICATE corpus (fix):")
mmr_results = vectordb_dup.max_marginal_relevance_search(
    query_dup,
    k=3,
    fetch_k=10,      # consider top-10 candidates before applying diversity filter
    lambda_mult=0.5, # 0=max diversity, 1=max relevance
)
_show_docs_full(mmr_results, f"MMR — '{query_dup}'  k=3  fetch_k=10  lambda=0.5")

mmr_sources = [d.metadata.get("source", "?").split("/")[-1] + f" p{d.metadata.get('page','?')}"
               for d in mmr_results]
unique_pages = len(set(mmr_sources))
print(f"\n  Sources returned : {mmr_sources}")
print(f"  Unique pages     : {unique_pages} / {len(mmr_results)}"
      f"  ← {'diverse ✓' if unique_pages > 1 else 'still duplicated'}")

# ── MMR on clean corpus — confirm same behaviour without duplicates ────────────
print(f"\n  MMR search on CLEAN corpus (for reference):")
mmr_clean = vectordb.max_marginal_relevance_search(query_dup, k=3, fetch_k=10)
_show_docs_full(mmr_clean, "MMR on clean corpus — for comparison")

clean_sources = [d.metadata.get("source", "?").split("/")[-1] + f" p{d.metadata.get('page','?')}"
                 for d in mmr_clean]
print(f"\n  Sources returned : {clean_sources}")
print(f"  Unique pages     : {len(set(clean_sources))} / {len(mmr_clean)}")

# ── MMR via Retriever interface (preferred pattern for use in chains) ──────────
print(f"\n  MMR via Retriever interface (as_retriever):")
mmr_retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
)
mmr_ret_results = mmr_retriever.invoke(query_dup)
_show_docs(mmr_ret_results, "as_retriever(search_type='mmr') — cleaner API for chains")
print(f"  NOTE: Use this form when wiring retrievers into RetrievalQA / LCEL chains.")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SIMILARITY SEARCH WITH SCORE
#    Returns (Document, score) tuples where score is the cosine *distance*
#    (lower = more similar).  Useful for setting relevance thresholds.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. SIMILARITY SEARCH WITH SCORE")
print("=" * 60)

scored = vectordb.similarity_search_with_score(
    "how do transformer models and attention mechanisms work?", k=4
)
print(f"\n  {'Score':>8}  Source / snippet")
print(f"  {'-'*8}  {'-'*55}")
for doc, score in scored:
    src     = doc.metadata.get("source", "?").split("/")[-1]
    snippet = doc.page_content[:80].replace("\n", " ")
    print(f"  {score:8.4f}  [{src}]  {snippet!r}")

print("\n  NOTE: ChromaDB returns L2 distance (lower = more similar).")


# ══════════════════════════════════════════════════════════════════════════════
# 7. METADATA FILTERING
#    Restrict search to chunks whose metadata matches a filter.
#    Useful when one vector store holds multiple documents and you want
#    to query only a specific source.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. METADATA FILTERING — query a single source")
print("=" * 60)

# Inspect available metadata keys / values
sample_meta = splits[0].metadata
print(f"\n  Sample chunk metadata: {sample_meta}")

handbook_source = HANDBOOK_PDF   # full path as stored by PyMuPDFLoader

# Filter: only return chunks from the India Handbook
filtered = vectordb.similarity_search(
    "employee benefits and leave entitlements",
    k=3,
    filter={"source": handbook_source},
)
_show_docs(filtered, "filter={'source': 'India-Handbook-2024.pdf'}")

# Filter: only return chunks from the AI Roadmap PDF
roadmap_source = roadmap_pdf if len(pdf_files) > 1 else os.path.join(DOCS_DIR, "handbook-part2-synthetic.pdf")
filtered_roadmap = vectordb.similarity_search(
    "skills needed to become an AI or LLM engineer",
)
_show_docs(filtered_roadmap, f"filter={{'source': '{os.path.basename(roadmap_source)}'}}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. PERSISTENT CHROMADB
#    Write the vector store to disk.  On subsequent runs the embeddings are
#    reused — no re-embedding needed (saves API calls and time).
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. PERSISTENT CHROMADB — save & reload")
print("=" * 60)

# Clean up any leftover persist directory from a previous run
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
    print(f"\n  Removed old persist dir: {PERSIST_DIR}")

# Save to disk
print(f"  Persisting {len(splits)} chunks to '{PERSIST_DIR}' …")
vectordb_persist = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
)
print(f"  Saved.  Collection count: {vectordb_persist._collection.count()}")

# Reload from disk (no re-embedding)
print(f"\n  Reloading from disk …")
vectordb_reload = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)
print(f"  Reloaded.  Collection count: {vectordb_reload._collection.count()}")

# Verify the reloaded store works
reload_results = vectordb_reload.similarity_search(
    "how do large language models work and what are their key components?", k=2
)
_show_docs(reload_results, "Query against reloaded store")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(textwrap.dedent("""
  Concept                     Key takeaway
  ──────────────────────────────────────────────────────────────
  Embeddings                  Semantic closeness = cosine similarity
  ChromaDB                    Fast ANN index over embedded chunks
  similarity_search()         Ranked by cosine distance — can surface duplicates
  Duplicate chunk problem     Top-k results all from same document (redundant)
  MMR search                  Balances relevance + diversity; fixes duplicates
  similarity_search_with_score  Expose the raw distance for thresholding
  Metadata filtering          Scope queries to specific source documents
  Persistent store            Save embeddings to disk; reload without re-embedding
"""))
