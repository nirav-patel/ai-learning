"""
LangChain: Q&A over Documents — Amazon Bedrock Edition
=======================================================
Based on the DeepLearning.AI "LangChain for LLM Application Development" L4
course notebook, adapted to use AWS Bedrock instead of OpenAI.

WHAT IT DOES
------------
Allows you to ask natural-language questions against a product catalog stored
in a CSV file.  The program finds the most relevant catalog entries and
synthesises an accurate, cited answer using a Claude model on Amazon Bedrock.

HOW IT WORKS (pipeline)
-----------------------
1. Load   → CSVLoader reads OutdoorClothingCatalog_1000.csv into LangChain
            Document objects (one Document per CSV row).

2. Embed  → HuggingFaceEmbeddings (all-MiniLM-L6-v2, runs locally) converts each
            document's text into a numeric vector that captures its meaning.
            No API calls — the model runs entirely on your machine.

3. Store  → InMemoryVectorStore (langchain-core) stores all vectors in RAM so similarity
            search is instant (no external DB needed for this demo).
            Because embeddings are local, building the index takes <2 s for 1000 docs.

4. Query  → At query time the question is also embedded, and the k nearest
            catalog entries are retrieved from the vector store.

5. Answer → Two approaches are demonstrated:
            A. VectorstoreIndexCreator (one-liner convenience wrapper) – quick
               to write, suitable for simple interactive use.
            B. RetrievalQA chain (step-by-step) – explicit control over the
               retriever, chain type ("stuff"), LLM and verbosity.

Both approaches ultimately pass the retrieved catalog snippets plus the user
question into Claude to generate a final answer.

RUN
---
    cd lang-chain-demo
    python l4_qna.py

DEPENDENCIES (see ../requirements.txt)
---------------------------------------
    langchain, langchain-aws, langchain-huggingface, langchain-community, langchain-core,
    langchain-classic, sentence-transformers, python-dotenv, boto3, botocore, certifi
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import textwrap
import warnings

# ── Suppress noisy third-party warnings BEFORE importing them ─────────────────
# These env vars must be set before huggingface/transformers packages are loaded.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import certifi
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv, find_dotenv

from langchain_aws import ChatBedrock
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.indexes import VectorstoreIndexCreator
from langchain_classic.chains import RetrievalQA

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv(find_dotenv())

# ── Configuration ─────────────────────────────────────────────────────────────
CSV_FILE = os.path.join(os.path.dirname(__file__), "OutdoorClothingCatalog_1000.csv")

EMBEDDING_MODEL_ID  = "all-MiniLM-L6-v2"  # local model, no API calls
LLM_MODEL_ID        = "us.anthropic.claude-sonnet-4-6"
AWS_REGION          = os.getenv("AWS_REGION", "us-west-1")

# Sample queries that exercise the catalog nicely
QUERY_SUN_SHIRTS = (
    "Please list all your shirts with sun protection "
    "in a table in markdown and summarize each one."
)
QUERY_RAINY_DAY = "What rain jackets do you have under 200 g that also pack small?"
QUERY_BASE_LAYER = "Recommend a base layer for cold and wet winter hikes."


# ── Helper builders ───────────────────────────────────────────────────────────

def _make_bedrock_client() -> boto3.client:
    """Return an authenticated bedrock-runtime boto3 client for the LLM region."""
    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        verify=certifi.where(),
    )


def _make_embeddings() -> HuggingFaceEmbeddings:
    """Return a local HuggingFace embeddings model (no API calls, runs on CPU)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)


def _make_llm(client: boto3.client) -> ChatBedrock:
    """Return a ChatBedrock LLM (Claude) with temperature 0 for factual answers."""
    return ChatBedrock(
        client=client,
        model_id=LLM_MODEL_ID,
        model_kwargs={"temperature": 0.0},
    )


def _load_docs() -> list[Document]:
    """Load the CSV catalog into LangChain Documents using stdlib csv (no community dep)."""
    with open(CSV_FILE, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [
            Document(
                page_content="\n".join(f"{k}: {v}" for k, v in row.items()),
                metadata={"source": CSV_FILE, "row": i},
            )
            for i, row in enumerate(reader)
        ]


# ── Approach A: VectorstoreIndexCreator (quick one-liner) ────────────────────

def demo_index_query(query: str, embeddings: HuggingFaceEmbeddings, llm: ChatBedrock) -> str:
    """
    Build an in-memory vector index from the catalog CSV and run a query.

    VectorstoreIndexCreator is a convenience wrapper that combines:
    CSVLoader → local embeddings → InMemoryVectorStore → RetrievalQA in one call.
    """
    index = VectorstoreIndexCreator(
        vectorstore_cls=InMemoryVectorStore,
        embedding=embeddings,
    ).from_documents(_load_docs())

    return index.query(query, llm=llm)


# ── Approach B: Step-by-step with RetrievalQA ────────────────────────────────

def demo_retrieval_qa(
    query: str,
    embeddings: HuggingFaceEmbeddings,
    llm: ChatBedrock,
    k: int = 4,
) -> str:
    """
    Manually build the pipeline:
      1. Load documents from CSV.
      2. Embed and store in DocArrayInMemorySearch.
      3. Create a retriever that fetches the top-k most similar documents.
      4. Wire into a RetrievalQA chain with chain_type="stuff" (concatenate
         all retrieved docs into a single prompt).

    The "stuff" chain type is simple and effective for small retrieval sets.
    """
    docs = _load_docs()

    # Build the in-memory vector store
    db = InMemoryVectorStore.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": k})

    # Build the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=False,
    )

    return qa_chain.invoke({"query": query})["result"]


# ── Optional: demonstrate similarity_search directly ─────────────────────────

def demo_similarity_search(query: str, embeddings: HuggingFaceEmbeddings, top_k: int = 4) -> None:
    """Print the raw documents retrieved by similarity search (no LLM step)."""
    docs = _load_docs()
    db = InMemoryVectorStore.from_documents(docs, embeddings)
    results = db.similarity_search(query, k=top_k)

    print(f"\n── Similarity search: '{query}' (top {top_k}) ──")
    for i, doc in enumerate(results, 1):
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"  {i}. {preview}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(textwrap.dedent("""\
        ╔══════════════════════════════════════════════════════════╗
        ║   LangChain Q&A over Documents  –  AWS Bedrock Edition   ║
        ╚══════════════════════════════════════════════════════════╝
    """))

    try:
        client     = _make_bedrock_client()
        embeddings = _make_embeddings()
        llm        = _make_llm(client)
    except Exception as exc:
        print(f"[ERROR] Could not initialise: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Step 1: raw similarity search (no LLM) ───────────────────────────────
    demo_similarity_search("shirt with sun blocking", embeddings, top_k=3)

    # ── Step 2: Approach A – VectorstoreIndexCreator ─────────────────────────
    print(f"\n{'='*60}")
    print("APPROACH A  –  VectorstoreIndexCreator (one-liner index)")
    print(f"{'='*60}")
    print(f"Query: {QUERY_SUN_SHIRTS}\n")
    try:
        response_a = demo_index_query(QUERY_SUN_SHIRTS, embeddings, llm)
        print(response_a)
    except ClientError as exc:
        print(f"[AWS ERROR] {exc}", file=sys.stderr)

    # ── Step 3: Approach B – Explicit RetrievalQA chain ──────────────────────
    print(f"\n{'='*60}")
    print("APPROACH B  –  RetrievalQA chain (step-by-step)")
    print(f"{'='*60}")
    print(f"Query: {QUERY_SUN_SHIRTS}\n")
    try:
        response_b = demo_retrieval_qa(QUERY_SUN_SHIRTS, embeddings, llm)
        print(response_b)
    except ClientError as exc:
        print(f"[AWS ERROR] {exc}", file=sys.stderr)

    # ── Step 4: Additional query via RetrievalQA ──────────────────────────────
    print(f"\n{'='*60}")
    print("ADDITIONAL QUERY  –  RetrievalQA")
    print(f"{'='*60}")
    print(f"Query: {QUERY_RAINY_DAY}\n")
    try:
        response_c = demo_retrieval_qa(QUERY_RAINY_DAY, embeddings, llm)
        print(response_c)
    except ClientError as exc:
        print(f"[AWS ERROR] {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
