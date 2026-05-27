"""
Retrieval Demo - LangChain Chat with Data
==========================================
Based on DeepLearning.AI "LangChain: Chat with Your Data" L4 notebook.
Uses HuggingFaceEmbeddings (local) + AWS Bedrock Claude for LLM-based retrievers.

WHY RETRIEVAL MATTERS
─────────────────────
Vector similarity alone has three failure modes:

  1. DIVERSITY     — top-k results are near-duplicates from the same chunk
                     Fix: MMR (done in l3) or EnsembleRetriever

  2. SPECIFICITY   — query says "page 5 of the handbook" but similarity
                     doesn't understand metadata constraints
                     Fix: Self-Query Retriever (LLM extracts filter)

  3. VERBOSITY     — retrieved chunk contains the answer buried in noise
                     Fix: Contextual Compression (LLM extracts just the answer)

RETRIEVAL METHODS COVERED
──────────────────────────
  1. Retriever interface       — as_retriever() with similarity / MMR / threshold
  2. MultiQueryRetriever       — LLM generates N rephrasings; union of results
  3. Self-Query Retriever      — LLM parses query → vector search + metadata filter
  4. Contextual Compression    — LLMChainExtractor shrinks chunks to relevant parts
  5. TF-IDF Retriever          — sparse keyword-based (no embeddings)
  6. BM25 Retriever            — improved sparse retrieval (BM25 scoring)
  7. EnsembleRetriever         — combines BM25 + ChromaDB with RRF fusion

SETUP REUSE
───────────
  Loads the persistent ChromaDB built by l3_vectorstores_and_embeddings.py
  (sample_chroma_db/).  Run l3 first if that directory does not exist.

RUN
───
    cd langchain-chat-with-data
    uv run python l4_retrieval.py

DEPENDENCIES
────────────
    chromadb, langchain-chroma, langchain-huggingface, langchain-aws,
    langchain-community, rank-bm25, scikit-learn, lark, boto3, certifi, pypdf
"""

from __future__ import annotations

import logging
import os
import sys
import textwrap
import warnings

import boto3
import certifi

# Suppress noisy HuggingFace/tokenizer output
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
AWS_REGION       = os.getenv("AWS_REGION", "us-west-1")
DOCS_DIR         = "docs"
PERSIST_DIR      = "sample_chroma_db"          # created by l3_vectorstores_and_embeddings.py
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 150


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


# ── Helper ─────────────────────────────────────────────────────────────────────

def _show(docs: list, label: str = "") -> None:
    if label:
        print(f"\n  [{label}]")
    print(f"  Results: {len(docs)}")
    for i, d in enumerate(docs):
        src     = d.metadata.get("source", "?").split("/")[-1]
        pg      = d.metadata.get("page", "?")
        snippet = d.page_content.strip().replace("\n", " ")[:150]
        print(f"  {i+1}. [{src} | p{pg}]  {snippet!r}")


# ══════════════════════════════════════════════════════════════════════════════
# SETUP — load embeddings + build/reload vector store
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
    print(f"\n  Reloading persistent ChromaDB from '{PERSIST_DIR}' …")
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    splits = []   # already stored; rebuild for BM25/TFIDF retrievers below
    pdf_files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for pdf_path in pdf_files:
        splits.extend(splitter.split_documents(PyMuPDFLoader(pdf_path).load()))
else:
    print(f"\n  '{PERSIST_DIR}' not found — building fresh ChromaDB …")
    pdf_files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_pages: list = []
    for pdf_path in pdf_files:
        all_pages.extend(PyMuPDFLoader(pdf_path).load())
    splits = splitter.split_documents(all_pages)
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )

print(f"  Vector store chunks : {vectordb._collection.count()}")
print(f"  In-memory splits    : {len(splits)} (used by sparse retrievers)")


# ══════════════════════════════════════════════════════════════════════════════
# 1. RETRIEVER INTERFACE
#    as_retriever() wraps a vector store into the standard LangChain Retriever
#    interface.  All retrievers share .invoke(query) → list[Document].
#    This makes them drop-in replaceable in any chain.
#
#    search_type options:
#      "similarity"              — ranked by cosine distance (default)
#      "mmr"                     — Maximum Marginal Relevance (diversity)
#      "similarity_score_threshold" — only return results above a threshold
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. RETRIEVER INTERFACE — as_retriever()")
print("=" * 60)

query = "What is an embedding model and how does it work?"

# 1a. Default similarity
sim_retriever = vectordb.as_retriever(search_kwargs={"k": 3})
sim_results = sim_retriever.invoke(query)
_show(sim_results, "search_type='similarity'  k=3")

# 1b. MMR — diversity-aware
mmr_retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
)
mmr_results = mmr_retriever.invoke(query)
_show(mmr_results, "search_type='mmr'  k=3  fetch_k=10  lambda=0.5")

sim_pages = [d.metadata.get("page") for d in sim_results]
mmr_pages = [d.metadata.get("page") for d in mmr_results]
print(f"\n  similarity pages : {sim_pages}  unique={len(set(sim_pages))}")
print(f"  mmr        pages : {mmr_pages}  unique={len(set(mmr_pages))}")

# 1c. Score threshold — only return chunks with distance below threshold
thresh_retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 5},
)
thresh_results = thresh_retriever.invoke(query)
_show(thresh_results, "search_type='similarity_score_threshold'  threshold=0.7")
print(f"\n  NOTE: ChromaDB score threshold filters out low-relevance chunks."
      f"  Returned {len(thresh_results)} of max 5.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. MULTI-QUERY RETRIEVER
#    Problem: a single query phrasing may miss relevant chunks that use
#             different vocabulary.
#    Fix: ask the LLM to rephrase the query in N different ways, run all
#         of them, then deduplicate the union of results.
#
#    Example: "leave policy" → also generates:
#      "vacation entitlement", "PTO rules", "absence guidelines" …
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. MULTI-QUERY RETRIEVER")
print("=" * 60)

from langchain_classic.retrievers.multi_query import MultiQueryRetriever

print("\n  Initialising LLM (Bedrock Claude) for query generation …")
try:
    llm = _make_llm()

    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        llm=llm,
    )

    # Enable logging to see the generated queries
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    mq_query = "how do large language models generate text?"
    mq_results = mq_retriever.invoke(mq_query)

    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.ERROR)

    _show(mq_results, f"MultiQueryRetriever — '{mq_query}'")
    print(f"\n  MultiQuery returned {len(mq_results)} unique chunks "
          f"(union across all generated query variants).")
except Exception as exc:
    print(f"  [SKIP] MultiQueryRetriever requires Bedrock access: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. SELF-QUERY RETRIEVER
#    Problem: user says "find leave policies on page 18" — similarity search
#             ignores the page-number constraint entirely.
#    Fix: Self-Query uses an LLM to parse the natural-language query into:
#           • semantic part  → embedded for vector search
#           • filter part    → applied as a metadata filter
#
#    Requires: pip install lark  (for the filter expression grammar)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. SELF-QUERY RETRIEVER")
print("=" * 60)

from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_community.query_constructors.chroma import ChromaTranslator

# Describe the metadata fields so the LLM knows what filters are available
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The full file path of the PDF document. "
                    "One of: 'docs/India-Handbook-2024.pdf', "
                    "'docs/AI Engineering.pdf', or "
                    "'docs/LLM Engineers Handbook.pdf'",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page number within the PDF document (0-indexed).",
        type="integer",
    ),
]

document_content_description = (
    "HR policies, employee benefits, leave rules, and compliance guidelines "
    "from the India Employee Handbook; practical AI engineering guidance, "
    "LLM architecture, RAG pipelines, embeddings, vector databases, and "
    "fine-tuning techniques from the AI Engineering and LLM Engineers Handbook."
)

try:
    llm = _make_llm()

    sq_retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectordb,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        structured_query_translator=ChromaTranslator(),
        verbose=True,
    )

    # Query 1: semantic only — no metadata hint
    sq_q1 = "What is RAG and how does retrieval augmented generation work?"
    print(f"\n  Query 1 (semantic only): '{sq_q1}'")
    sq_r1 = sq_retriever.invoke(sq_q1)
    _show(sq_r1)

    # Query 2: with explicit page constraint
    sq_q2 = "what does page 5 of the LLM Engineers Handbook say about embeddings?"
    print(f"\n  Query 2 (page constraint): '{sq_q2}'")
    sq_r2 = sq_retriever.invoke(sq_q2)
    _show(sq_r2)
    if sq_r2:
        pages = [d.metadata.get("page") for d in sq_r2]
        print(f"  Pages returned: {pages}  ← self-query applied page filter")

    # Query 3: with source document constraint
    sq_q3 = "what fine-tuning techniques are described in the AI Engineering document?"
    print(f"\n  Query 3 (source constraint): '{sq_q3}'")
    sq_r3 = sq_retriever.invoke(sq_q3)
    _show(sq_r3)
    if sq_r3:
        sources = list({d.metadata.get("source", "?").split("/")[-1] for d in sq_r3})
        print(f"  Sources returned: {sources}  ← self-query filtered by doc")

except Exception as exc:
    print(f"  [SKIP] SelfQueryRetriever error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONTEXTUAL COMPRESSION
#    Problem: a retrieved chunk is 1000 characters but only 50 characters
#             answer the question — the LLM downstream gets 950 chars of noise.
#    Fix: wrap the retriever in a ContextualCompressionRetriever that uses an
#         LLM to extract only the sentence(s) that answer the query.
#
#    Two compressors:
#      LLMChainExtractor — extracts the relevant passage (may paraphrase)
#      LLMChainFilter    — keeps or drops whole chunks (no paraphrasing)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. CONTEXTUAL COMPRESSION")
print("=" * 60)

from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter

try:
    llm = _make_llm()
    base_retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    comp_query = "What is the difference between a vector database and traditional search?"

    # ── 4a. LLMChainExtractor ─────────────────────────────────────────────────
    extractor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=extractor,
        base_retriever=base_retriever,
    )
    print(f"\n  Query: '{comp_query}'")
    print(f"\n  Base retriever (uncompressed):")
    base_results = base_retriever.invoke(comp_query)
    for i, d in enumerate(base_results[:2]):
        src = d.metadata.get("source", "?").split("/")[-1]
        print(f"  {i+1}. [{src} | p{d.metadata.get('page')}]  "
              f"len={len(d.page_content)} chars  "
              f"{d.page_content.strip().replace(chr(10),' ')[:100]!r}")

    print(f"\n  ContextualCompressionRetriever (LLMChainExtractor):")
    compressed = compression_retriever.invoke(comp_query)
    for i, d in enumerate(compressed):
        src = d.metadata.get("source", "?").split("/")[-1]
        print(f"  {i+1}. [{src} | p{d.metadata.get('page')}]  "
              f"len={len(d.page_content)} chars  "
              f"{d.page_content.strip().replace(chr(10),' ')[:150]!r}")
    print(f"\n  Chunks reduced from {len(base_results)} → {len(compressed)} "
          f"(irrelevant chunks dropped or compressed).")

    # ── 4b. LLMChainFilter ────────────────────────────────────────────────────
    print(f"\n  LLMChainFilter (keep/drop whole chunks, no paraphrasing):")
    filt = LLMChainFilter.from_llm(llm)
    filter_retriever = ContextualCompressionRetriever(
        base_compressor=filt,
        base_retriever=base_retriever,
    )
    filtered = filter_retriever.invoke(comp_query)
    _show(filtered, f"LLMChainFilter — '{comp_query}'")

    # ── 4c. Combination: MMR base + LLMChainExtractor ─────────────────────────
    #    MMR already removes near-duplicate chunks before compression.
    #    Compression then strips irrelevant content from each diverse chunk.
    #    Result: diverse AND concise — the best of both worlds.
    print(f"\n  Combination: MMR base retriever + LLMChainExtractor:")
    mmr_base = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    mmr_compression_retriever = ContextualCompressionRetriever(
        base_compressor=LLMChainExtractor.from_llm(llm),
        base_retriever=mmr_base,
    )
    mmr_compressed = mmr_compression_retriever.invoke(comp_query)
    for i, d in enumerate(mmr_compressed):
        src = d.metadata.get("source", "?").split("/")[-1]
        print(f"  {i+1}. [{src} | p{d.metadata.get('page')}]  "
              f"len={len(d.page_content)} chars  "
              f"{d.page_content.strip().replace(chr(10), ' ')[:150]!r}")

    mmr_base_results = mmr_base.invoke(comp_query)
    mmr_pages     = [d.metadata.get("page") for d in mmr_base_results]
    compr_pages   = [d.metadata.get("page") for d in mmr_compressed]
    print(f"\n  MMR base pages      : {mmr_pages}  unique={len(set(mmr_pages))}")
    print(f"  After compression   : {compr_pages} chunks "
          f"(MMR diversity preserved, content trimmed to relevant parts)")

except Exception as exc:
    print(f"  [SKIP] Contextual compression requires Bedrock access: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TF-IDF RETRIEVER
#    Pure keyword-based retrieval (no embeddings, no LLM).
#    Scores each document by the TF-IDF weight of query terms.
#    Fast and interpretable; works well for exact keyword matches.
#    Weakness: misses synonyms and paraphrases ("PTO" ≠ "paid time off").
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. TF-IDF RETRIEVER  (sparse keyword-based)")
print("=" * 60)

from langchain_community.retrievers.tfidf import TFIDFRetriever

tfidf = TFIDFRetriever.from_documents(splits, k=3)

tfidf_q1 = "transformer architecture attention mechanism LLM"
tfidf_results = tfidf.invoke(tfidf_q1)
_show(tfidf_results, f"TF-IDF — '{tfidf_q1}'")

# Show limitation: paraphrase that embedding search handles but TF-IDF misses
tfidf_q2 = "how neural networks learn to understand language"   # semantically same, different words
tfidf_r2  = tfidf.invoke(tfidf_q2)
vec_r2    = vectordb.as_retriever(search_kwargs={"k": 3}).invoke(tfidf_q2)

print(f"\n  Paraphrase comparison: '{tfidf_q2}'")
tfidf_srcs = [d.metadata.get("source","?").split("/")[-1]+f" p{d.metadata.get('page','?')}"
              for d in tfidf_r2]
vec_srcs   = [d.metadata.get("source","?").split("/")[-1]+f" p{d.metadata.get('page','?')}"
              for d in vec_r2]
print(f"  TF-IDF results   : {tfidf_srcs}")
print(f"  Vector results   : {vec_srcs}")
print(f"  → TF-IDF may miss paraphrases; vector search handles semantic similarity.")


# ══════════════════════════════════════════════════════════════════════════════
# 6. BM25 RETRIEVER
#    Improvement over TF-IDF: adds document-length normalisation and term
#    saturation so frequent terms don't dominate.  Still keyword-based.
#    Widely used as the baseline in information retrieval benchmarks.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. BM25 RETRIEVER  (improved sparse retrieval)")
print("=" * 60)

from langchain_community.retrievers.bm25 import BM25Retriever

bm25 = BM25Retriever.from_documents(splits, k=3)

bm25_q = "vector embeddings semantic search similarity"
bm25_results = bm25.invoke(bm25_q)
_show(bm25_results, f"BM25 — '{bm25_q}'")

# Compare TF-IDF vs BM25 on the same query
tfidf_bm25_q = tfidf.invoke(bm25_q)
tfidf_bm25_srcs = [d.metadata.get("source","?").split("/")[-1]+f" p{d.metadata.get('page','?')}"
                   for d in tfidf_bm25_q]
bm25_srcs = [d.metadata.get("source","?").split("/")[-1]+f" p{d.metadata.get('page','?')}"
             for d in bm25_results]
print(f"\n  Query: '{bm25_q}'")
print(f"  TF-IDF pages : {tfidf_bm25_srcs}")
print(f"  BM25   pages : {bm25_srcs}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. ENSEMBLE RETRIEVER (BM25 + ChromaDB)
#    Combines a sparse retriever (BM25) and a dense retriever (ChromaDB)
#    using Reciprocal Rank Fusion (RRF).
#
#    WHY IT WORKS BETTER
#    ─────────────────────
#    Dense (vector) retrieval:  finds semantically similar text
#    Sparse (BM25) retrieval:   finds exact keyword matches
#    Ensemble:                  catches what either method alone misses
#
#    Weights: [0.5, 0.5] = equal; increase BM25 weight for keyword-heavy queries
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. ENSEMBLE RETRIEVER  (BM25 + ChromaDB)")
print("=" * 60)

from langchain_classic.retrievers.ensemble import EnsembleRetriever

bm25_retriever  = BM25Retriever.from_documents(splits, k=3)
chroma_retriever = vectordb.as_retriever(search_kwargs={"k": 3})

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.5, 0.5],
)

ens_query = "how do you fine-tune a large language model?"
ens_results   = ensemble.invoke(ens_query)
bm25_only     = bm25_retriever.invoke(ens_query)
chroma_only   = chroma_retriever.invoke(ens_query)

_show(bm25_only,   f"BM25 only   — '{ens_query}'")
_show(chroma_only, f"Chroma only — '{ens_query}'")
_show(ens_results, f"Ensemble    — '{ens_query}'  (BM25 0.5 + Chroma 0.5)")

bm25_pages   = {d.metadata.get("page") for d in bm25_only}
chroma_pages = {d.metadata.get("page") for d in chroma_only}
ens_pages    = {d.metadata.get("page") for d in ens_results}
print(f"\n  BM25   unique pages : {sorted(bm25_pages)}")
print(f"  Chroma unique pages : {sorted(chroma_pages)}")
print(f"  Ensemble pages      : {sorted(ens_pages)}  ← union via RRF fusion")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY — Choosing a retriever")
print("=" * 60)
print(textwrap.dedent("""
  Retriever                  Best for
  ─────────────────────────────────────────────────────────────────────
  similarity (as_retriever)  General semantic search; default choice
  MMR (as_retriever)         When diversity matters (avoid duplicate chunks)
  score_threshold            Filter out low-confidence results
  MultiQueryRetriever        Query phrasing varies; need broader recall
  SelfQueryRetriever         User queries contain metadata hints (page, source)
  ContextualCompression      Noisy chunks; pass only relevant passage to LLM
  TF-IDF                     Exact keyword match; fast; no embedding needed
  BM25                       Better than TF-IDF; still sparse; no embedding
  EnsembleRetriever          Best recall overall: combines dense + sparse
"""))
