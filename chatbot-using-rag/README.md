# RAG Chatbot — Chat with Your PDFs

A production-ready Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDF documents and ask questions about them. Built with LangChain, Weaviate, and Gradio.

---

## How it works

```
Your question
     │
     ▼
 1. Condenser ── (LLM) ──► standalone question  (uses chat history)
     │
     ▼
 2. Retriever ─────────► top-k relevant chunks  (hybrid vector + BM25)
     │
     ▼
 3. Answerer ── (LLM) ──► streamed answer        (uses context + history)
```

1. **Condense** — your follow-up question is rephrased into a self-contained query using conversation history
2. **Retrieve** — the query fetches the most relevant document chunks via hybrid search (semantic + keyword)
3. **Answer** — the LLM streams an answer grounded strictly in the retrieved context

---

## Project structure

```
chatbot-using-rag/
├── main.py                          ← entry point — run this
├── pyproject.toml                   ← dependencies
├── .env.example                     ← copy to .env and configure
├── data-sources/                    ← place your PDFs here
├── vector-store/                    ← Weaviate data (auto-created)
└── chatbot/
    ├── config.py                    ← AppConfig — all settings in one place
    ├── state.py                     ← AppState — runtime state per instance
    ├── app.py                       ← Gradio UI
    ├── pipeline.py                  ← RAGPipeline class (3-step chain)
    ├── ingestion.py                 ← DocumentLoader class (PDF → chunks)
    ├── storage.py                   ← WeaviateStore class (vector DB)
    ├── eval/
    │   ├── __init__.py              ← eval package
    │   ├── eval_questions.toml      ← question bank (edit this to add/change questions)
    │   ├── run_evaluation.py        ← offline batch eval (20 questions, RAG Triad)
    │   ├── prod_tracing.py          ← live PROD tracing wrapper (opt-in)
    │   └── trulens.db               ← SQLite results DB (auto-created, gitignored)
    ├── providers/
    │   ├── llm.py                   ← LLM factory (bedrock / openai / ollama)
    │   └── embeddings.py            ← Embedding factory (huggingface / openai / ollama)
    └── infrastructure/
        ├── logging.py               ← structured logging
        ├── observability.py         ← Phoenix tracing
        └── env_loader.py            ← .env loader
```

---

## Quickstart

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd chatbot-using-rag
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` for your setup. Minimum required for AWS Bedrock (default):

```env
AWS_REGION=us-west-2
LLM_PROVIDER=bedrock
EMBED_PROVIDER=huggingface
```

### 3. Run

```bash
python main.py
```

- **Chatbot UI** → http://localhost:7860
- **Phoenix traces** → http://localhost:6006 *(auto-starts)*

### 4. Upload a PDF and start chatting

Use the **🗄 Database** tab to upload PDFs, then switch to **💬 Conversation** to ask questions.

---

## Configuration

All settings live in `AppConfig` (`chatbot/config.py`) and can be overridden via `.env`:

| Env var | Default | Description |
|---------|---------|-------------|
| `LLM_PROVIDER` | `bedrock` | LLM backend: `bedrock` \| `openai` \| `ollama` |
| `EMBED_PROVIDER` | `huggingface` | Embedding backend: `huggingface` \| `openai` \| `ollama` |
| `AWS_REGION` | `us-west-2` | AWS region (Bedrock only) |
| `OPENAI_API_KEY` | — | Required when using `openai` provider |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `DATA_SOURCES_DIR` | `./data-sources` | Directory scanned for PDFs on first run |
| `WEAVIATE_PERSIST_DIR` | `./vector-store` | On-disk path where Weaviate persists data |
| `PHOENIX_ENABLED` | `true` | Enable/disable Phoenix tracing |
| `PHOENIX_PORT` | `6006` | Phoenix dashboard port |
| `LOG_LEVEL` | `INFO` | Logging level: `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |

### Switching LLM or embedding model

No code changes needed — update `.env` only:

```env
# Use OpenAI instead of AWS Bedrock
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Use a different embedding model
EMBED_PROVIDER=openai
```

Or run locally with Ollama:

```env
LLM_PROVIDER=ollama
EMBED_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Providers

### LLM providers

| Provider | Package | Notes |
|----------|---------|-------|
| `bedrock` | `langchain-aws` | Default. Requires AWS credentials (`aws configure`) |
| `openai` | `langchain-openai` | Requires `OPENAI_API_KEY` |
| `ollama` | `langchain-ollama` | Requires running Ollama server |

### Embedding providers

| Provider | Package | Notes |
|----------|---------|-------|
| `huggingface` | `langchain-huggingface` | Default. Runs locally, no API key |
| `openai` | `langchain-openai` | Requires `OPENAI_API_KEY` |
| `ollama` | `langchain-ollama` | Requires running Ollama server |

Install Ollama support:
```bash
pip install langchain-ollama
```

---

## Observability

[Phoenix by Arize](https://phoenix.arize.com/) is embedded — no Docker or external account needed. It auto-starts when the app launches and traces every LLM call, retriever call, and chain step.

**Dashboard:** http://localhost:6006

To disable:
```env
PHOENIX_ENABLED=false
```

---

## Tech stack

| Component | Technology |
|-----------|-----------|
| LLM | AWS Bedrock / OpenAI / Ollama |
| Embeddings | HuggingFace (nomic-embed-text-v1.5) / OpenAI / Ollama |
| Vector store | [Weaviate](https://weaviate.io/) (embedded, no Docker needed) |
| Retrieval | Hybrid search — dense vector + BM25 (Reciprocal Rank Fusion) |
| PDF loading | [pymupdf4llm](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) — Markdown-aware (tables, headings preserved) |
| Chunking | tiktoken BPE token-based splitting with Markdown-aware separators |
| UI | [Gradio](https://gradio.app/) |
| Tracing / Observability | [Phoenix (Arize)](https://phoenix.arize.com/) — real-time LLM/retriever traces, latency, token counts |
| RAG Evaluation | [TruLens](https://www.trulens.org/) — LLM-as-judge quality scores (Groundedness, Relevance) |
| Orchestration | [LangChain LCEL](https://python.langchain.com/docs/concepts/lcel/) |

---

## Evaluation (TruLens RAG Triad)

The `chatbot/eval/` folder contains a TruLens-powered quality harness that measures the RAG pipeline on three metrics using Bedrock Claude as the LLM judge. There are **two modes**: offline batch testing and live PROD monitoring.

### RAG Triad metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **Answer Relevance** | Is the answer relevant to the question? | > 0.7 |
| **Context Relevance** | Are the retrieved chunks relevant to the question? | > 0.7 |
| **Groundedness** | Is the answer grounded in the retrieved context (no hallucination)? | > 0.8 |

### Setup (both modes)

```bash
pip install -e ".[eval]"
```

AWS credentials must be configured — the same Bedrock credentials used by the chatbot are reused as the LLM judge.

---

### Mode 1 — Offline batch evaluation

Runs 20 curated AI/ML/Transformer questions (easy → complex) through the full pipeline and scores every answer. No live traffic needed. Results are stored in `chatbot/eval/trulens.db`.

```bash
# Basic run — prints leaderboard to console
python -m chatbot.eval.run_evaluation

# Also print per-question score breakdown to console (no browser needed)
python -m chatbot.eval.run_evaluation --show-records

# Fresh run — clears previous results first
python -m chatbot.eval.run_evaluation --reset

# Label a run for version comparison
python -m chatbot.eval.run_evaluation --reset --app-version v2

# Run and immediately open browser dashboard
python -m chatbot.eval.run_evaluation --reset --dashboard
```

**View previous results without re-running:**

```bash
# Console only — leaderboard + per-question scores (no browser)
python -m chatbot.eval.run_evaluation --results-only

# Browser dashboard
python -m chatbot.eval.run_evaluation --dashboard-only
# → http://localhost:8501
```

**Comparing two versions** (e.g. after tuning chunking, retriever k, or prompt):

```bash
python -m chatbot.eval.run_evaluation --app-version v1        # baseline
# ... make changes ...
python -m chatbot.eval.run_evaluation --app-version v2 --dashboard   # compare
```

Both versions appear side-by-side in the leaderboard. Omit `--reset` to keep history across runs.

---

### Mode 2 — Live PROD tracing

Every real user chat turn is automatically recorded to the TruLens DB while the chatbot runs normally. Useful for monitoring quality drift on real traffic over time.

**Enable in `.env`:**

```env
TRULENS_PROD_ENABLED=true

# Also score every live turn with RAG Triad metrics (async, ~5-10s background latency):
# TRULENS_PROD_FEEDBACK=true

# Label this deployment in the dashboard:
# APP_VERSION=prod
```

**Run the chatbot as normal:**

```bash
python main.py
# Chatbot UI      → http://localhost:7860
# TruLens records every turn to chatbot/eval/trulens.db
```

**View live results in browser:**

```bash
python -m chatbot.eval.run_evaluation --dashboard-only
# → http://localhost:8501
```

---

### Dashboard walkthrough

| Tab | What you see |
|-----|-------------|
| **Leaderboard** | Avg Answer Relevance / Context Relevance / Groundedness per app version |
| **Evaluations** | Per-question scores with chain-of-thought reasoning from the judge LLM |
| **Records** | Full trace per turn: input → retrieved chunks → answer |

---

### Current baseline scores (offline, v1)

```
                     Answer Relevance  Context Relevance  Groundedness  avg latency
RAGChatbot v1               0.50              0.57            1.00         ~7s
```

> **Tip:** Context Relevance and Answer Relevance improve when the PDFs in `data-sources/` match the domain of questions. Add AI/ML papers or documentation to see scores rise.

---
- Implement multi-modal RAG for this chatbot like PDF RAG for PROD?
- Convert to agentic RAG and add steps like query parser, result evaluator etc?
- Implement moderation for user queries?

- Sentence Window Retrieval vs. Auto merging retrieval
