# Research Agent

Implements two complementary agentic workflows on top of **AWS Bedrock**:

1. **Essay Reflection Pipeline** — one LLM drafts, a second critiques, and the first revises.
2. **Research Pipeline with Tools** — LLM calls arXiv + Tavily search tools, reflects on the report, and publishes styled HTML.

---

## Pipelines

### Pipeline 1 — Essay Reflection

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

A single LLM pass often produces a structurally sound but argumentatively weak first draft. The reflection step uses a more analytically capable model to surface concrete issues; the revision step incorporates all feedback.

### Pipeline 2 — Research with Tools (search → reflection → HTML)

```
Prompt
  │
  ▼
┌──────────────────┐     ┌─────────────────────┐     ┌──────────────┐
│ Search + Report  │────▶│ Reflect + Rewrite    │────▶│   HTML       │
│ (arXiv + Tavily) │     │ (JSON: reflection +  │     │  Output      │
│  Claude Sonnet   │     │  revised_report)     │     │              │
└──────────────────┘     └─────────────────────┘     └──────────────┘
       ▲ tool loop
  arXiv API / Tavily API
```

The LLM autonomously decides when to call tools, gathers sources, writes a cited report, critiques it, rewrites it, and outputs a self-contained HTML file.

---

## Prerequisites

**AWS credentials** must be available via the default boto3 credential chain (instance profile, `~/.aws/credentials`, or environment variables). `AWS_REGION` defaults to `us-east-1`.

**Tavily API key** is required for Pipeline 2. Add it to a `.env` file in this directory:

```
TAVILY_API_KEY=your_key_here
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running

```bash
cd agentic-workflow/research-agent

python main.py                   # run both pipelines
python main.py --essay-only      # essay reflection pipeline only
python main.py --research-only   # research + tools pipeline only
```

Pipeline 2 saves the final HTML report to `research_report.html` in the same directory.

---

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point — runs one or both pipelines, prints results, runs unit tests |
| `workflow.py` | All pipeline functions for both workflows (see tables below) |
| `bedrock_client.py` | AWS Bedrock Converse API wrapper — text generation, system prompts, tool-calling loop |
| `research_tools.py` | arXiv + Tavily search functions, Bedrock `toolSpec` definitions, `TOOL_MAPPING`, `parse_input` |
| `eval.py` | Component-level evaluation — source quality checker comparing URLs against `TOP_DOMAINS` |
| `tests.py` | Unit test helpers for all pipeline functions (including eval) |
| `requirements.txt` | Python dependencies |

---

## Model Configuration

| Role | Default model |
|------|--------------|
| Generation / Revision (essay) | `us.amazon.nova-2-lite-v1:0` |
| Reflection / Research (tools) | `us.anthropic.claude-sonnet-4-6` |

Edit the constants at the top of `main.py` to swap models.

---

## Workflow Functions

### Pipeline 1 — Essay Reflection (`workflow.py`)

| Function | Input | Output |
|----------|-------|--------|
| `generate_draft(topic, model)` | Essay topic | Initial draft string |
| `reflect_on_draft(draft, model)` | Draft string | Feedback string (structure, clarity, argument, style) |
| `revise_draft(draft, reflection, model)` | Draft + feedback | Revised essay string |
| `run_essay_workflow(topic, model_generation, model_reflection)` | Topic + model names | `WorkflowResult` dataclass |

### Pipeline 2 — Research with Tools (`workflow.py`)

| Function | Input | Output |
|----------|-------|--------|
| `generate_research_report_with_tools(prompt, model)` | Research topic | `(report_str, messages_list)` — cited report + full conversation history |
| `reflection_and_rewrite(report, model, temperature)` | Report text or messages list | `dict` with `"reflection"` and `"revised_report"` keys |
| `convert_report_to_html(report, model, temperature)` | Report text or messages list | Complete HTML document string |
| `run_research_pipeline(topic, generation_model, reflection_model)` | Topic + model names | `ResearchResult` dataclass + saves `research_report.html` |

### Evaluation (`eval.py`)

| Function | Input | Output |
|----------|-------|--------|
| `evaluate_report_sources(report_text, top_domains, min_ratio)` | Plain-text report | `(bool flag, markdown_report)` — PASS/FAIL based on preferred URL ratio |
| `evaluate_tool_call_sources(messages, tool_name, top_domains, min_ratio)` | Bedrock message history | `(bool flag, markdown_report)` — evaluates Tavily raw results directly |

`TOP_DOMAINS` is a predefined set of ~30 trusted academic/scientific domains used as ground truth (e.g., `arxiv.org`, `nature.com`, `nasa.gov`, `harvard.edu`).

The evaluation runs automatically in `run_research_pipeline` after Step 1 and is stored in `ResearchResult.eval_report`. It can also be run standalone:

```bash
python eval.py research_report.html
```

### Bedrock Client (`bedrock_client.py`)

| Function | Purpose |
|----------|---------|
| `generate_text(model_id, prompt, ...)` | Single-turn text generation |
| `generate_text_with_system(model_id, system_prompt, user_prompt, ...)` | Single-turn with system prompt |
| `run_tool_loop(model_id, system_prompt, initial_prompt, tools, tool_mapping, ...)` | Agentic tool-calling loop — handles `toolUse` / `toolResult` rounds until final answer |
| `generate_with_image(model_id, prompt, image_path, ...)` | Multimodal (text + image) generation |

---

## Key Learning Points

- **Component-level evaluation** — checking source quality independently of the full pipeline avoids noisy end-to-end reruns; a single step evaluated with an objective metric (preferred domain ratio) gives a clear improvement signal.
- **Reflection pattern** — chaining generation → critique → revision for iterative quality improvement.
- **Tool-calling (agentic loop)** — Bedrock `toolConfig` with `toolChoice: auto`; detects `toolUse` blocks, executes Python functions, feeds `toolResult` messages back, loops until a plain-text answer is returned.
- **Structured JSON output** — prompting the model to return only valid JSON (no response-format API needed on Bedrock).
- **HTML report generation** — converting plain-text research output to shareable, styled HTML via prompt engineering.
- **Separation of roles** — fast model drafts; stronger model critiques and researches; avoids over-spending on every step.
