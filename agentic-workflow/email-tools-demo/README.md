# Email Tools Demo

An agentic workflow demo where an LLM manages a simulated email inbox by calling REST API tools.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  main.py                                                    │
│  ├── starts email_server (FastAPI + SQLite) as subprocess   │
│  └── runs 4 demo scenarios via run_with_tools()             │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────────────────────┐
│  agent.py           │────▶│  email_tools.py                  │
│  run_with_tools()   │     │  list_all_emails()               │
│  Bedrock agentic    │     │  list_unread_emails()            │
│  loop               │     │  search_emails(query)            │
└─────────────────────┘     │  get_email(email_id)             │
                            │  mark_email_as_read(email_id)    │
                            │  mark_email_as_unread(email_id)  │
                            │  send_email(recipient, subj, body│
                            │  delete_email(email_id)          │
                            │  search_unread_from_sender(sender│
                            └──────────────────────────────────┘
                                         │ HTTP REST
                                         ▼
                            ┌──────────────────────────────────┐
                            │  email_server/                   │
                            │  FastAPI + SQLAlchemy + SQLite   │
                            │  Pre-loaded with 6 sample emails │
                            └──────────────────────────────────┘
```

## Demo Scenarios

| # | Prompt | Tools available | What it shows |
|---|--------|-----------------|---------------|
| 1 | "Check unread from boss, mark as read, send reply" | search_unread, mark_read, send | Multi-step reasoning |
| 2 | "Delete alice@work.com email" | All **except** delete | Tool gap — LLM refuses gracefully |
| 3 | "Delete alice@work.com email" | All tools including delete | Succeeds after inbox reset |
| 4 | "Delete the happy hour email" | search + delete | Subject-based search & delete |

## Running

```bash
# From repo root
source .venv/bin/activate

# Install extra deps if needed
pip install fastapi uvicorn sqlalchemy pydantic[email]

cd agentic-workflow/email-tools-demo
python main.py
```

The email server starts automatically on port 5000 and shuts down when the demo ends.

## Key Design Choices

- **`email_server/`** — standalone FastAPI app backed by SQLite; pre-loads 6 sample emails on startup; exposes a `/reset_database` endpoint used between demos
- **`email_tools.py`** — plain Python functions that call the server via HTTP; these are the tools the LLM can invoke
- **`agent.py`** — reusable Bedrock agentic loop; auto-generates Bedrock tool specs from function docstrings and type hints; handles `tool_use` / `end_turn` stop reasons
- **Demo 2 vs 3** — the same prompt is run twice to show how the agent behaves when a required tool is missing vs. available

## Models

| Role | Model |
|------|-------|
| Agent | `SMART_MODEL` (`us.anthropic.claude-sonnet-4-6`) |
