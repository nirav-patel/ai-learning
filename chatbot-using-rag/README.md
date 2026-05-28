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
| Observability | [Phoenix (Arize)](https://phoenix.arize.com/) |
| Orchestration | [LangChain LCEL](https://python.langchain.com/docs/concepts/lcel/) |


## TODOs
- Implement multi-modal RAG for this chatbot like PDF RAG for PROD?
- Convert to agentic RAG and add steps like query parser, result evaluator etc?
- Implement moderation for user queries?