"""
LangChain: Evaluation — Amazon Bedrock Edition
===============================================
Based on the DeepLearning.AI "LangChain for LLM Application Development" L5
course notebook, adapted to use AWS Bedrock instead of OpenAI.

WHAT IT DOES
------------
Demonstrates three approaches to evaluating a RetrievalQA pipeline:
1. Hard-coded test examples   — manually crafted query/answer pairs.
2. LLM-generated examples     — Claude auto-generates Q&A pairs from catalog
                                 documents via QAGenerateChain.
3. LLM-assisted grading       — Claude grades predicted answers vs. ground
                                 truth via QAEvalChain (CORRECT / INCORRECT).

HOW IT WORKS (pipeline)
-----------------------
Build phase  → Same as l4_qna.py: CSV → local embeddings → InMemoryVectorStore
               → RetrievalQA chain backed by Claude on Bedrock.

Evaluation phases:
  A. Manual inspection   – run a single query with langchain.debug=True so
                           every chain/LLM event is logged to stdout, giving
                           visibility into exactly what the model was sent and
                           what it returned.
  B. Batch prediction    – run every example through the QA chain and collect
                           (query, expected_answer, predicted_answer) triples.
  C. LLM grading         – feed the triples to QAEvalChain; Claude responds
                           CORRECT or INCORRECT for each one.

RUN
---
    cd lang-chain-demo
    python l5_evaluation.py

DEPENDENCIES (see ../requirements.txt)
---------------------------------------
    langchain, langchain-aws, langchain-huggingface, langchain-core,
    langchain-classic, sentence-transformers, python-dotenv, boto3, certifi
"""

from __future__ import annotations

import csv
import logging
import os
import random
import sys
import textwrap
import warnings

# ── Suppress noisy third-party warnings BEFORE importing them ─────────────────
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
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.chains import RetrievalQA

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv(find_dotenv())

# ── Configuration ─────────────────────────────────────────────────────────────
CSV_FILE           = os.path.join(os.path.dirname(__file__), "OutdoorClothingCatalog_1000.csv")
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"   # local model — no API calls
LLM_MODEL_ID       = "us.anthropic.claude-sonnet-4-6"
AWS_REGION         = os.getenv("AWS_REGION", "us-west-1")

# Number of catalog documents to feed into QAGenerateChain for auto-generation
NUM_GEN_DOCS = 5


# ── Helper builders ───────────────────────────────────────────────────────────

def _make_bedrock_client() -> boto3.client:
    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        verify=certifi.where(),
    )


def _make_embeddings() -> HuggingFaceEmbeddings:
    """Local HuggingFace embeddings — no API calls, runs entirely on CPU."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)


def _make_llm(client: boto3.client) -> ChatBedrock:
    return ChatBedrock(
        client=client,
        model_id=LLM_MODEL_ID,
        model_kwargs={"temperature": 0.0},
    )


def _load_docs() -> list[Document]:
    """Load the CSV catalog using stdlib csv (no langchain_community dep)."""
    with open(CSV_FILE, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [
            Document(
                page_content="\n".join(f"{k}: {v}" for k, v in row.items()),
                metadata={"source": CSV_FILE, "row": i},
            )
            for i, row in enumerate(reader)
        ]


def _build_qa_chain(
    docs: list[Document],
    embeddings: HuggingFaceEmbeddings,
    llm: ChatBedrock,
    k: int = 4,
) -> RetrievalQA:
    """Build an in-memory vector store and wire it into a RetrievalQA chain."""
    db = InMemoryVectorStore.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": k})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=False,
        return_source_documents=True,
    )


# ── Step 1: Hard-coded test examples ─────────────────────────────────────────

def get_hardcoded_examples() -> list[dict]:
    """
    Manually crafted ground-truth pairs that are easy to verify by reading our
    OutdoorClothingCatalog_1000.csv directly.

    Row 0  → Women's UPF 50+ Sun Shield Shirt: "blocks 98% of UV rays"
    Row 5  → Women's Solar Defender Long-Sleeve Shirt: "UPF 50 sun protection"
    """
    return [
        {
            "query": "What percentage of UV rays does the Women's UPF 50+ Sun Shield Shirt block?",
            "answer": "98%",
        },
        {
            "query": "What type of sun protection does the Women's Solar Defender Long-Sleeve Shirt offer?",
            "answer": "UPF 50 sun protection",
        },
    ]


# ── QA generation prompt (replaces QAGenerateChain) ─────────────────────────

_QA_GEN_TEMPLATE = """You are a teacher creating a quiz. Given the product description
below, write ONE factual question that can be answered solely from the text, and
provide a concise answer.

Product description:
{doc}

Respond in this exact format (no extra text):
QUESTION: <question here>
ANSWER: <answer here>"""

_QA_GEN_PROMPT = PromptTemplate(
    input_variables=["doc"],
    template=_QA_GEN_TEMPLATE,
)


def _parse_qa_gen_output(text: str) -> dict | None:
    """Parse QUESTION:/ANSWER: lines from the LLM response."""
    question, answer = None, None
    for line in text.splitlines():
        if line.startswith("QUESTION:"):
            question = line.removeprefix("QUESTION:").strip()
        elif line.startswith("ANSWER:"):
            answer = line.removeprefix("ANSWER:").strip()
    if question and answer:
        return {"query": question, "answer": answer}
    return None


# ── QA evaluation prompt (replaces QAEvalChain) ───────────────────────────────

_QA_EVAL_TEMPLATE = """You are grading a student's answer to a question.

Question: {query}
True Answer: {answer}
Student's Answer: {result}

Is the student's answer correct? Reply with exactly one word: CORRECT or INCORRECT."""

_QA_EVAL_PROMPT = PromptTemplate(
    input_variables=["query", "answer", "result"],
    template=_QA_EVAL_TEMPLATE,
)


# ── Step 2: LLM-generated test examples ──────────────────────────────────────

def generate_examples(docs: list[Document], llm: ChatBedrock) -> list[dict]:
    """
    Automatically create Q&A pairs from catalog documents.

    Sends each document to Claude with a structured prompt asking for one
    factual question + answer grounded in that document.  Produces additional
    test coverage without manual effort.

    Mirrors what the original notebook's QAGenerateChain did, but implemented
    directly with PromptTemplate + LLMChain so it works regardless of which
    langchain evaluation sub-packages are installed.
    """
    gen_chain = _QA_GEN_PROMPT | llm

    sampled = random.sample(docs, min(NUM_GEN_DOCS, len(docs)))
    print(f"  Generating Q&A pairs from {len(sampled)} randomly sampled catalog documents …")
    new_examples: list[dict] = []
    for i, doc in enumerate(sampled):
        try:
            msg = gen_chain.invoke({"doc": doc.page_content})
            qa_pair = _parse_qa_gen_output(msg.content)
            if qa_pair:
                new_examples.append(qa_pair)
                print(f"    Doc {i}: Q — {qa_pair['query'][:80]}")
            else:
                print(f"    Doc {i}: could not parse LLM output")
        except Exception as exc:
            print(f"    Doc {i}: generation failed — {exc}")

    return new_examples


# ── Step 3: Manual evaluation (debug mode) ───────────────────────────────────

def demo_manual_evaluation(
    qa_chain: RetrievalQA,
    examples: list[dict],
) -> None:
    """
    Enable langchain.debug so every chain and LLM event is printed to stdout.
    This reveals exactly what context was retrieved, what prompt was built, and
    what the model returned — useful for debugging wrong answers.
    """
    import langchain  # type: ignore[import]

    print("\nEnabling langchain.debug = True …")
    langchain.debug = True
    try:
        result = qa_chain.invoke({"query": examples[0]["query"]})
        print("\nAnswer:", result["result"])
    finally:
        langchain.debug = False
        print("\nlangchain.debug = False (restored)")


# ── Step 4: Batch predictions ─────────────────────────────────────────────────

def run_predictions(
    qa_chain: RetrievalQA,
    examples: list[dict],
) -> list[dict]:
    """
    Run every example through the QA chain and collect structured predictions.

    Returns a list of dicts with keys:
      query   — the original question
      answer  — the ground-truth expected answer
      result  — the model's predicted answer
    """
    predictions: list[dict] = []
    for i, ex in enumerate(examples):
        print(f"  [{i + 1}/{len(examples)}] {ex['query'][:70].strip()} …")
        try:
            out = qa_chain.invoke({"query": ex["query"]})
            predictions.append({
                "query":  ex["query"],
                "answer": ex["answer"],
                "result": out["result"],
            })
        except ClientError as exc:
            print(f"    [AWS ERROR] {exc}")
            predictions.append({
                "query":  ex["query"],
                "answer": ex["answer"],
                "result": "ERROR",
            })
    return predictions


# ── Step 5: LLM-assisted grading ─────────────────────────────────────────────

def grade_predictions(
    llm: ChatBedrock,
    examples: list[dict],
    predictions: list[dict],
) -> list[dict]:
    """
    Ask Claude to grade each prediction as CORRECT or INCORRECT.

    Mirrors what QAEvalChain did in the original notebook, implemented directly
    with PromptTemplate + LLMChain so it works with current langchain versions.

    For each prediction the prompt provides:
      - the original question
      - the ground-truth expected answer
      - the model's predicted answer
    Claude replies with exactly one word: CORRECT or INCORRECT.
    """
    eval_chain = _QA_EVAL_PROMPT | llm
    graded: list[dict] = []
    for pred in predictions:
        try:
            msg = eval_chain.invoke({
                "query":  pred["query"],
                "answer": pred["answer"],
                "result": pred["result"],
            })
            verdict = msg.content.strip().upper()
            # normalise to CORRECT / INCORRECT
            if "INCORRECT" in verdict:
                verdict = "INCORRECT"
            elif "CORRECT" in verdict:
                verdict = "CORRECT"
            graded.append({"text": verdict})
        except Exception as exc:
            graded.append({"text": f"ERROR: {exc}"})
    return graded


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(textwrap.dedent("""\
        ╔══════════════════════════════════════════════════════════╗
        ║   LangChain: Evaluation  –  AWS Bedrock Edition          ║
        ╚══════════════════════════════════════════════════════════╝
    """))

    # Initialise shared resources
    try:
        client     = _make_bedrock_client()
        embeddings = _make_embeddings()
        llm        = _make_llm(client)
        docs       = _load_docs()
    except Exception as exc:
        print(f"[ERROR] Initialisation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(docs)} catalog documents.\n")

    # ── Build QA chain (reused across all evaluation steps) ──────────────────
    print("Building RetrievalQA chain …")
    qa_chain = _build_qa_chain(docs, embeddings, llm)

    # ── 1. Hard-coded examples ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1 — Hard-coded test examples")
    print("=" * 60)
    examples = get_hardcoded_examples()
    for ex in examples:
        print(f"  Q: {ex['query']}")
        print(f"  A: {ex['answer']}\n")

    # Peek at the raw documents backing those examples (rows 0 and 5)
    print("Raw catalog documents for reference:")
    for doc in docs[0:1] + docs[5:6]:
        print(f"  {doc.page_content[:200].replace(chr(10), ' ')}")
    print()

    # ── 2. LLM-generated examples ─────────────────────────────────────────────
    print("=" * 60)
    print("STEP 2 — LLM-generated test examples  (Claude → catalog docs)")
    print("=" * 60)
    new_examples = generate_examples(docs, llm)
    if new_examples:
        print(f"\n  Generated {len(new_examples)} new example(s).")
        for ex in new_examples:
            print(f"  Q: {ex.get('query', '').strip()}")
            print(f"  A: {ex.get('answer', '').strip()}\n")
    examples += new_examples
    print(f"Total examples: {len(examples)}")

    # ── 3. Manual evaluation (debug trace) ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Manual evaluation  (langchain.debug trace for example 0)")
    print("=" * 60)
    try:
        demo_manual_evaluation(qa_chain, examples)
    except Exception as exc:
        print(f"  [SKIP] debug trace failed: {exc}")

    # ── 4. Batch predictions ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Batch predictions  (all examples through QA chain)")
    print("=" * 60)
    predictions = run_predictions(qa_chain, examples)

    # ── 5. LLM-assisted grading ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — LLM-assisted grading  (QAEvalChain via Claude)")
    print("=" * 60)
    graded_outputs = grade_predictions(llm, examples, predictions)

    # ── Print final scorecard ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SCORECARD")
    print("=" * 60)
    correct = 0
    for i, (ex, pred, grade) in enumerate(zip(examples, predictions, graded_outputs)):
        verdict = grade.get("text", "N/A").strip()
        if verdict == "CORRECT":
            correct += 1
        print(f"\nExample {i}:")
        print(f"  Question        : {pred['query'].strip()}")
        print(f"  Expected Answer : {pred['answer'].strip()}")
        print(f"  Predicted Answer: {pred['result'].strip()}")
        print(f"  Grade           : {verdict}")

    total = len(predictions)
    print(f"\n{'=' * 60}")
    print(f"Score: {correct}/{total} correct")
    print("=" * 60)


if __name__ == "__main__":
    main()
