# Research Agent — Reflective Essay Writing

Demonstrates the **reflection design pattern** applied to essay writing: one LLM drafts, a second critiques, and the first revises — mirroring how a thoughtful writer would self-edit.

## Overview

```
Topic
  │
  ▼
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│  Generate   │─────▶│   Reflect    │─────▶│    Revise    │
│    Draft    │      │  on Draft    │      │  with        │
│  Nova Lite  │      │ Claude Sonnet│      │  Feedback    │
└─────────────┘      └──────────────┘      │  Nova Lite   │
                                           └──────────────┘
```

### Why reflection matters

A single LLM pass often produces a structurally sound but argumentatively weak first draft. The reflection step uses a more analytically capable model to surface concrete issues (weak transitions, unsupported claims, tone mismatches). The revision step incorporates all of that feedback, producing a materially better essay than either model could achieve alone in one shot.

## Prerequisites

Valid AWS credentials must be available via the default boto3 credential chain (instance profile, `~/.aws/credentials`, or environment variables). `AWS_REGION` can be overridden via the environment (defaults to `us-east-1`).

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point — sets topic, configures models, prints results |
| `workflow.py` | Core pipeline: `generate_draft`, `reflect_on_draft`, `revise_draft`, `run_essay_workflow` |
| `bedrock_client.py` | Thin wrapper around AWS Bedrock Converse API |

## Running

```bash
cd agentic-workflow/research-agent
python main.py
```

## Model Configuration

| Role | Default model |
|------|--------------|
| Generation / Revision | `us.amazon.nova-2-lite-v1:0` |
| Reflection (critique) | `us.anthropic.claude-sonnet-4-6` |

Edit the constants at the top of `main.py` to swap models.

## Workflow Functions

| Function | Input | Output |
|----------|-------|--------|
| `generate_draft(topic, model)` | Essay topic string | Initial draft string |
| `reflect_on_draft(draft, model)` | Draft string | Feedback string covering structure, clarity, argument, style |
| `revise_draft(original_draft, reflection, model)` | Draft + feedback | Revised essay string |
| `run_essay_workflow(topic, model_generation, model_reflection)` | Topic + model names | `WorkflowResult` dataclass |

## Key Learning Points

- **Reflection pattern** — chaining generation → critique → revision for iterative improvement.
- **Separation of roles** — a fast model drafts; a stronger model critiques; avoids over-spending on every step.
- **Prompt engineering** — each step has a focused, role-specific prompt to keep outputs on task.
