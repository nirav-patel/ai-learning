# AI Learning — AWS Bedrock + LangChain

A hands-on collection of progressively advanced demos covering LLM fundamentals, LangChain patterns, and production-grade RAG pipelines — all powered by **AWS Bedrock (Claude Sonnet 4.6)** and local HuggingFace embeddings.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Module Summaries](#module-summaries)
  - [api-call-demo — LLM API Fundamentals & Prompt Engineering](#api-call-demo)
  - [langchain-basics — LangChain Core Patterns](#langchain-basics)
  - [langchain-chat-with-data — RAG & Conversational AI](#langchain-chat-with-data)
- [Running the Files](#running-the-files)

---

## Project Structure

```
learning/
├── requirements.txt                  # Shared Python dependencies
├── api-call-demo/                    # Direct Bedrock API + prompt engineering
├── langchain-basics/                 # LangChain fundamentals (models, memory, chains, agents)
└── langchain-chat-with-data/         # RAG pipeline (loading → splitting → retrieval → chat UI)
```

---

## Prerequisites

- Python 3.11+
- AWS credentials configured (`~/.aws/credentials` or environment variables) with Bedrock access
- AWS region with Bedrock enabled (e.g. `us-east-1`)

---

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download required NLP models (first time only)

```bash
# SpaCy English model (used in document splitting)
python -m spacy download en_core_web_sm

# NLTK tokenizer data (used in document splitting)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 4. Configure environment variables (optional)

Create a `.env` file in the project root if you need to override defaults:

```dotenv
AWS_DEFAULT_REGION=us-east-1
# LANGCHAIN_API_KEY=<your-langsmith-key>   # optional, for tracing in l5_question_answering.py
```

---

## Module Summaries

### api-call-demo

> **Theme:** Learn the fundamentals of calling LLMs directly via the Bedrock API, building safe and evaluable systems from first principles.

| File | What it covers |
|---|---|
| `l1_language_models.py` | Basic completions, system roles, token usage tracking |
| `l2_classification.py` | Routing customer queries via structured JSON classification |
| `l3_moderation.py` | Input safety — harmful content detection & prompt injection prevention |
| `l4_chain_of_thought.py` | Step-by-step reasoning to reduce hallucinations |
| `l5_chaining_prompts.py` | Multi-step pipelines: extract → lookup → respond |
| `l6_check_outputs.py` | Output validation — moderation + factual accuracy checks |
| `l7_end_to_end_demo.py` | Complete 7-step production pipeline with a Panel web UI |
| `l8_evaluate_with_ideal.py` | Evaluating extraction quality against hand-crafted ideal answers |
| `l9_evaluate_with_rubric.py` | Rubric-based and A–E scale evaluation of response quality |

### langchain-basics

> **Theme:** Build idiomatic LangChain applications using LCEL, memory strategies, chain composition, and ReAct agents.

| File | What it covers |
|---|---|
| `l1_langchain_models_prompts_parsers.py` | ChatBedrock, PromptTemplate, StrOutputParser, StructuredOutputParser |
| `l2_memory_examples.py` | Buffer, Window, Token-limited, and Summary memory strategies |
| `l3_chains_demo.py` | LLMChain, SequentialChain, RouterChain (legacy + LCEL) |
| `l4_qna.py` | Vector-based Q&A over a 1000-item product CSV using local embeddings |
| `l5_evaluation.py` | Auto-generating test cases and LLM-assisted grading (CORRECT/INCORRECT) |
| `l6_agents_demo.py` | ReAct agents with calculator, Wikipedia, Python REPL, and date tools |

### langchain-chat-with-data

> **Theme:** Build a full RAG pipeline — from document ingestion to a conversational chat UI with persistent memory.

| File | What it covers |
|---|---|
| `l1_document_loading.py` | Load PDFs, web pages, Notion exports, and YouTube transcripts |
| `l2_document_splitting.py` | 8 chunking strategies (Character, Recursive, Token, NLTK, SpaCy, etc.) |
| `l3_vectorstores_and_embeddings.py` | HuggingFace embeddings + ChromaDB, MMR, metadata filtering |
| `l4_retrieval.py` | MultiQuery, Self-Query, Contextual Compression, BM25, EnsembleRetriever |
| `l5_question_answering.py` | RetrievalQA with stuff / map_reduce / refine / map_rerank chain types |
| `l6_chat.py` | ConversationalRetrievalChain with Buffer, BufferWindow, and Summary memory |
| `minilm_chatbot.py` | Gradio chat UI (port 7070) — MiniLM embeddings, 256-token chunks |
| `nomic_chatbot.py` | Gradio chat UI (port 6060) — Nomic embeddings, 512-token chunks, 8k context |

---

## Running the Files

Make sure your virtual environment is active (`source .venv/bin/activate`) before running any script.

Each module has numbered files (`l1_`, `l2_`, …) — run them in order, as each lesson builds on the previous.

```bash
# api-call-demo: direct Bedrock API + prompt engineering
cd api-call-demo && python l1_language_models.py

# langchain-basics: LangChain patterns
cd langchain-basics && python l1_langchain_models_prompts_parsers.py

# langchain-chat-with-data: RAG pipeline
cd langchain-chat-with-data && python l1_document_loading.py
```

### Interactive Chatbot UIs

After running the `langchain-chat-with-data` scripts, launch a Gradio chat UI:

```bash
cd langchain-chat-with-data

python minilm_chatbot.py   # http://localhost:7070  (MiniLM, 256-token chunks)
python nomic_chatbot.py    # http://localhost:6060  (Nomic, 512-token chunks, 8k context)
```

> **Note:** The first run downloads HuggingFace model weights (~90 MB for MiniLM, ~270 MB for Nomic). Subsequent runs use the local cache.

