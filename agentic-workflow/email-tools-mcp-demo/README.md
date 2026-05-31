# Email Tools MCP Demo

An agentic workflow demo that exposes email operations as an **MCP (Model Context Protocol) server** and uses a Bedrock-powered agent to interact with them through the MCP protocol.

## How it differs from `email-tools-demo`

| | `email-tools-demo` | `email-tools-mcp-demo` |
|---|---|---|
| Tool invocation | Direct Python function call | MCP protocol (stdio transport) |
| Tool discovery | Hardcoded `ALL_TOOLS` list | Dynamic via `client.list_tools()` |
| Agent loop | Synchronous | Async (`asyncio`) |
| Reusability | In-process only | MCP server usable by Claude Desktop, Cursor, etc. |

## Architecture

```
main.py  (asyncio.run)
  │
  ├── FastAPI email server  (subprocess, port 5001, SQLite)
  │
  └── EmailMCPClient  (async context manager)
        │  stdio transport
        └── mcp_server/email_mcp_server.py  (subprocess)
              │  HTTP calls
              └── FastAPI email server
```

### Files

| File | Purpose |
|------|---------|
| `email_server/` | FastAPI + SQLAlchemy REST API (pre-loaded with 6 sample emails) |
| `mcp_server/email_mcp_server.py` | FastMCP server — 10 email tools via `@mcp.tool()` |
| `mcp_client.py` | `EmailMCPClient` — async context manager; bridges MCP ↔ Bedrock tool specs |
| `agent.py` | `async run_with_mcp_tools()` — Bedrock Converse loop with MCP tool dispatch |
| `display.py` | Terminal trace printer |
| `main.py` | Entry point — starts servers, runs 4 async demo scenarios |
| `bedrock_client.py` | Shared Bedrock wrapper |

## Demo Scenarios

| # | Prompt | What it shows |
|---|--------|---------------|
| 1 | "Check unread from boss, mark as read, send reply" | Multi-step MCP tool chain |
| 2 | "Delete alice@work.com email" | Search by sender → delete |
| 3 | "Delete the happy hour email" | Search by subject keyword → delete |
| 4 | "Summarise all unread emails" | Read-only MCP tool use |

## Running

```bash
source .venv/bin/activate
cd agentic-workflow/email-tools-mcp-demo
python main.py
```

The email server (port 5001) and MCP server (stdio) both start automatically and are shut down when the demo ends.

## MCP Server standalone

You can also run the MCP server on its own and connect with any MCP-compatible client (e.g. Claude Desktop):

```bash
EMAIL_SERVER_URL=http://127.0.0.1:5001 python -m mcp_server.email_mcp_server
```

## Models

| Role | Model |
|------|-------|
| Agent | `us.anthropic.claude-sonnet-4-6` (SMART_MODEL) |
