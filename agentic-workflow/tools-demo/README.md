# Tools Demo — Turning Functions into LLM Tools

Demonstrates how to give an LLM controlled access to Python functions via the AWS Bedrock Converse API — mirroring the reference lab (M3_UGL_1) but adapted for a script-based Bedrock workflow.

## Overview

```
User prompt
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Agentic Loop (agent.py)                             │
│                                                      │
│  ┌─────────┐   tool_use   ┌──────────┐              │
│  │  Bedrock │ ──────────▶ │  Execute │              │
│  │  Model  │ ◀────────── │  Locally │              │
│  └─────────┘  tool_result └──────────┘              │
│       │ end_turn                                     │
│       ▼                                              │
│  Final answer + tool trace                           │
└──────────────────────────────────────────────────────┘
```

### Key concepts

| Concept | Description |
|---------|-------------|
| **Tool spec auto-generation** | `agent.py` converts any Python function into a Bedrock tool spec using its docstring and type annotations — no manual schema needed |
| **Agentic loop** | Bedrock returns `stopReason: "tool_use"` when it wants to call a tool; the loop executes it locally and feeds results back |
| **Tool selection** | The model picks the right tool(s) from the full list based on the prompt — no hard-coding |
| **Multi-step execution** | A single prompt can trigger multiple sequential tool calls (see Demo 4) |

## Prerequisites

Valid AWS credentials via the default boto3 chain. `AWS_REGION` defaults to `us-east-1`.

```bash
pip install -r requirements.txt
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point — runs four demo scenarios |
| `agent.py` | `run_with_tools()` — agentic loop + auto tool-spec generation |
| `tools.py` | Four tool functions: `get_current_time`, `get_weather_from_ip`, `write_txt_file`, `generate_qr_code` |
| `display.py` | Terminal-friendly trace printer (mirrors reference `display_functions.py`) |
| `bedrock_client.py` | Shared Bedrock Converse wrapper |

## Running

```bash
cd agentic-workflow/tools-demo
python main.py
```

## Demo Scenarios

| Demo | Prompt | Tools used |
|------|--------|-----------|
| 1 | "What time is it?" | `get_current_time` |
| 2 | "Get the weather for my location" | `get_weather_from_ip` |
| 3 | "Create a reminder file" | `write_txt_file` |
| 4 | "Make a QR code AND write a weather note" | `generate_qr_code` → `get_weather_from_ip` → `write_txt_file` |

## Output Artifacts

Generated files are saved in the `output/` directory:
- `output/reminders.txt` — text reminder (Demo 3)
- `output/github_qr.png` — QR code image (Demo 4)
- `output/weather_note.txt` — weather note (Demo 4)

## Model Configuration

Default model: `us.anthropic.claude-sonnet-4-6`

Edit `MODEL` at the top of `main.py` to switch models.
