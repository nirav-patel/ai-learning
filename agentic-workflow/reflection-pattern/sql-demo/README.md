# SQL Agent with Reflection — Query Improvement

Demonstrates the **reflection design pattern** applied to SQL generation: one LLM writes an initial query, a second evaluates the *actual execution output* and proposes a corrected version.

## Overview

```
User question
     │
     ▼
┌─────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Generate   │─────▶│  Execute V1  │─────▶│   Reflect    │─────▶│  Execute V2  │
│  SQL (V1)   │      │  → DataFrame │      │  on output   │      │  (final ans) │
│ Nova Lite   │      │              │      │ Claude Sonnet│      │              │
└─────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
```

### Why reflection matters

The initial query may be syntactically valid yet semantically wrong (e.g., `SUM(qty_delta)` returns a negative revenue because sales are stored as negative quantities). The reflection step receives the **real output** as external feedback, allowing it to catch sign errors, missing filters, and grouping issues that are invisible from query text alone.

## Prerequisites

The `.env` file at the repository root must contain valid AWS credentials.  
`AWS_REGION` is read from environment; Bedrock credentials come from the default boto3 credential chain (instance profile, `~/.aws/credentials`, etc.).

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point — configures models and runs the workflow |
| `workflow.py` | Core reflection pipeline (`generate_sql`, `refine_sql_with_feedback`, `run_sql_workflow`) |
| `utils.py` | DB helpers: `create_transactions_db`, `get_schema`, `execute_sql` |
| `bedrock_client.py` | Thin wrapper around AWS Bedrock Converse API |
| `products.db` | SQLite database (auto-generated on first run) |

## Running

```bash
cd agentic-workflow/reflection-pattern/sql-demo
python main.py
```

## Database Schema

The auto-generated `products.db` uses an **event-sourced** design — one `transactions` table, all analytics derived from event history:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-increment event ID |
| `product_id` | INTEGER | Product identifier |
| `product_name` | TEXT | e.g. "Nike shoes" |
| `brand` | TEXT | Nike / Adidas / Puma … |
| `category` | TEXT | shoes / hoodie / t-shirt … |
| `color` | TEXT | black / white / red … |
| `action` | TEXT | `insert` / `restock` / `sale` / `price_update` |
| `qty_delta` | INTEGER | + for stock-in, − for sales, 0 for price updates |
| `unit_price` | REAL | Price at event time (NULL for restocks) |
| `notes` | TEXT | Human-readable description |
| `ts` | DATETIME | Event timestamp |

## Model Configuration

| Role | Default model |
|------|--------------|
| Generation (V1) | `us.amazon.nova-2-lite-v1:0` |
| Reflection (V2) | `us.anthropic.claude-sonnet-4-6` |

Edit the constants at the top of `main.py` to swap models.

## Key Learning Points

- **Reflection pattern** — how one LLM evaluates and corrects another's SQL.
- **External feedback** — using real query output (not just query text) to detect semantic errors.
- **Error recovery** — fixing sign conventions, missing filters, wrong aggregations.
- **Multi-model cooperation** — pairing a fast generator with a careful evaluator.
