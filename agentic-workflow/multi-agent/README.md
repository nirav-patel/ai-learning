# 🕶️ Multi-Agent Sunglasses Campaign Workflow

An autonomous **multi-agent pipeline** that researches fashion trends, generates a campaign image, writes a tagline, and produces an executive Markdown report — all without human intervention.

Built to demonstrate **creative + reliable** autonomous AI workflows: each agent is independently testable, loosely coupled, and hardened for production use.

---

## 📐 Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                       CampaignPipeline                             │
│                                                                    │
│  ┌──────────────────┐     ┌──────────────────┐                    │
│  │ MarketResearch   │────▶│ GraphicDesigner  │                    │
│  │    Agent  🕵️    │     │    Agent  🎨      │                    │
│  │                  │     │                  │                    │
│  │ Tools:           │     │ LLM → prompt     │                    │
│  │ • Tavily search  │     │ Amazon Titan     │                    │
│  │ • Product catalog│     │ → PNG image      │                    │
│  └──────────────────┘     └────────┬─────────┘                    │
│         │ trend_summary            │ image_path                   │
│         │                          ▼                              │
│         │             ┌──────────────────┐                        │
│         └────────────▶│  Copywriter      │                        │
│                        │    Agent  ✍️    │                        │
│                        │                  │                        │
│                        │ Multimodal LLM   │                        │
│                        │ image + text     │                        │
│                        │ → quote + why    │                        │
│                        └────────┬─────────┘                        │
│                                 │ quote, justification             │
│                                 ▼                                  │
│                        ┌──────────────────┐                        │
│                        │   Packaging      │                        │
│                        │    Agent  📦     │                        │
│                        │                  │                        │
│                        │ LLM beautifies   │                        │
│                        │ → Markdown report│                        │
│                        └──────────────────┘                        │
└────────────────────────────────────────────────────────────────────┘
```

### Agent responsibilities

| Agent | Input | Output | Tools |
|---|---|---|---|
| **MarketResearchAgent** | (none) | Trend summary text | Tavily web search, Product catalog |
| **GraphicDesignerAgent** | Trend summary | PNG image + prompt + caption | Amazon Titan Image Generator (Bedrock) |
| **CopywriterAgent** | Image + trend summary | Campaign quote + justification | Claude vision via Anthropic Bedrock SDK |
| **PackagingAgent** | All of the above | Markdown executive report | *(LLM only)* |

---

## 📂 Project Structure

```
multi-agent/
├── main.py                          # CLI entry point
├── config.py                        # Env-based configuration
├── requirements.txt
├── .env.example                     # Copy → .env and fill in your keys
│
├── data/
│   └── inventory.py                 # Sunglasses catalog DataFrames
│
├── tools/
│   ├── base_tool.py                 # Abstract BaseTool interface
│   ├── search_tool.py               # TavilySearchTool (with retry)
│   ├── catalog_tool.py              # ProductCatalogTool
│   └── registry.py                  # ToolRegistry — register + dispatch
│
├── agents/
│   ├── base_agent.py                # BaseAgent — generic agentic loop
│   ├── market_research_agent.py
│   ├── graphic_designer_agent.py
│   ├── copywriter_agent.py
│   └── packaging_agent.py
│
├── pipeline/
│   └── campaign_pipeline.py         # CampaignPipeline orchestrator
│
└── tests/
    ├── conftest.py                  # Shared fixtures (mock clients, tools)
    ├── unit/                        # Fast tests — no API keys required
    │   ├── test_inventory.py
    │   ├── test_tools.py
    │   ├── test_agents.py
    │   └── test_pipeline.py
    └── integration/                 # Live tests — auto-skipped without keys
        ├── test_tools_integration.py
        └── test_pipeline_integration.py
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

**Required credentials:**

| Variable | Required for | Notes |
|---|---|---|
| `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | All agents + image gen | Or use an IAM role / `AWS_PROFILE` |
| `AWS_REGION` | All Bedrock calls | Region with Bedrock access enabled (default: `us-east-1`) |
| `TAVILY_API_KEY` | MarketResearchAgent | [app.tavily.com](https://app.tavily.com) |

> **Bedrock model access:** Enable `us.anthropic.claude-sonnet-4-6` and `amazon.titan-image-generator-v2:0` in your AWS console under **Bedrock → Model access**.

### 3. Run the pipeline

```bash
python main.py
```

Output files are saved to `./output/` by default.

**Options:**

```bash
python main.py --help

python main.py --output-dir ./my-campaign --log-level DEBUG

python main.py --output-file my_report.md --caption-style "elegant and sophisticated"
```

---

## 🧪 Testing

### Unit tests (no API keys needed)

```bash
pytest tests/unit/ -v
```

All external services (Bedrock LLM, Amazon Titan image gen, Tavily) are replaced with lightweight mocks. These tests run instantly and are safe to run in CI.

```
tests/unit/test_inventory.py   — 25 tests  (data layer)
tests/unit/test_tools.py       — 19 tests  (tool definitions, registry, dispatch)
tests/unit/test_agents.py      — 19 tests  (agentic loop, all 4 agents)
tests/unit/test_pipeline.py    — 9 tests   (orchestration, data handoff)
```

### Integration tests (requires API keys in `.env`)

```bash
pytest tests/integration/ -v -s
```

Integration tests are automatically **skipped** if `TAVILY_API_KEY` or AWS credentials are absent — no special flags needed.

> ⚠️ These tests make real API calls and will incur costs.

### Run everything with coverage

```bash
pytest tests/ -v --cov=. --cov-report=term-missing --ignore=tests/integration
```

---

## 🔧 How It Works

### The Agentic Loop (BaseAgent)

All text agents share a loop driven by the Anthropic SDK (`anthropic.AnthropicBedrock`):

```
messages = [system, user]
loop:
  response = client.messages.create(messages, tools=registry.anthropic_definitions)
  if stop_reason == "end_turn" → return AgentResult(content)
  if stop_reason == "tool_use":
      tool_results = []
      for each ToolUseBlock in response.content:
          result = registry.dispatch_anthropic(block)
          tool_results.append({"type": "tool_result", "tool_use_id": block.id, ...})
      messages.append({"role": "user", "content": tool_results})   # ← ONE message, all results
      continue loop
```

The key difference from the OpenAI/aisuite protocol is that **all tool results for one assistant turn must be batched into a single user message** — Bedrock rejects consecutive `user` messages. `BaseAgent` handles this automatically.

Every agent inherits this loop from `BaseAgent`. Concrete agents only override:
- `build_system_prompt()` — role definition
- `build_user_prompt(**kwargs)` — task-specific instructions

### Tool Registry

The `ToolRegistry` is a central hub that:
1. Holds `BaseTool` instances keyed by name
2. Exposes their JSON schemas (`registry.anthropic_definitions`) in Anthropic tool format
3. Dispatches `ToolUseBlock` objects returned by the LLM to the right tool

```python
registry = ToolRegistry()
registry.register(TavilySearchTool(api_key="..."))
registry.register(ProductCatalogTool())

# Pass to an agent
agent = MarketResearchAgent(client=client, model="us.anthropic.claude-sonnet-4-6", registry=registry)
```

### Data Flow

```
MarketResearchAgent.run()
  └─▶ AgentResult.content  (trend_summary: str)
        │
        ▼
GraphicDesignerAgent.run(trend_summary=...)
  └─▶ {"image_path": "...", "prompt": "...", "caption": "..."}
        │
        ▼
CopywriterAgent.run(image_path=..., trend_summary=...)
  └─▶ {"quote": "...", "justification": "...", "image_path": "..."}
        │
        ▼
PackagingAgent.run(trend_summary=..., image_path=..., quote=..., justification=...)
  └─▶ "/path/to/campaign_summary_YYYY-MM-DD.md"
```

### LLM Clients

| Agent | LLM client | Why |
|---|---|---|
| **MarketResearchAgent** | `anthropic.AnthropicBedrock` | Native tool-use batching |
| **GraphicDesignerAgent** (text step) | `anthropic.AnthropicBedrock` | Native tool-use batching |
| **CopywriterAgent** | `anthropic.AnthropicBedrock` | Multimodal image + text input |
| **PackagingAgent** | `anthropic.AnthropicBedrock` | Summarisation call |
| **GraphicDesignerAgent** (image step) | `boto3.invoke_model` | Amazon Titan Image Generator v2 |

> All text agents share a **single** `AnthropicBedrock` client instance created in `CampaignPipeline.from_env()`. The Anthropic SDK is used (not `aisuite`) because Bedrock requires all tool results for one assistant turn to be batched into a **single** user message — a constraint that `aisuite`'s request converter does not satisfy when the LLM makes 2+ tool calls at once.

---



All settings can be overridden in `.env`:

| Variable | Default | Description |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | *(env / IAM role)* | AWS access key — optional if using IAM role or profile |
| `AWS_SECRET_ACCESS_KEY` | *(env / IAM role)* | AWS secret key |
| `AWS_REGION` | `us-east-1` | AWS region where Bedrock models are enabled |
| `AWS_PROFILE` | *(not set)* | Named AWS profile from `~/.aws/credentials` |
| `TAVILY_API_KEY` | *(required)* | Tavily search API key |
| `AGENT_MODEL` | `us.anthropic.claude-sonnet-4-6` | Bedrock model ID for text agents (bare cross-region inference ID) |
| `COPYWRITER_MODEL` | `us.anthropic.claude-sonnet-4-6` | Bedrock model ID for multimodal vision calls |
| `IMAGE_MODEL` | `amazon.titan-image-generator-v2:0` | Bedrock model ID for image generation (Amazon Titan) |
| `IMAGE_SIZE` | `1024x1024` | Image dimensions in `WxH` format |
| `OUTPUT_DIR` | `./output` | Where images and reports are saved |
| `MAX_AGENT_ITERATIONS` | `10` | Safety cap on the agentic loop |
| `TAVILY_MAX_RETRIES` | `3` | Retry attempts for Tavily searches |
| `TAVILY_RETRY_DELAY` | `1.5` | Base delay (seconds) between retries |

---

## 🏗️ Extending the Pipeline

### Add a new tool

```python
# tools/my_tool.py
from tools.base_tool import BaseTool, ToolDefinition

class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def definition(self) -> ToolDefinition:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "...",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

    def run(self, **kwargs):
        return {"result": "..."}
```

Then register it:

```python
registry.register(MyTool())
```

### Add a new agent

```python
from agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "MyAgent"

    def build_system_prompt(self) -> str:
        return "You are a specialist in ..."

    def build_user_prompt(self, trend_summary: str = "", **kwargs) -> str:
        return f"Analyse: {trend_summary}"
```

---

## 📄 Sample Output

After a successful run you'll find in `./output/`:

- `generated_image.png` — AI-generated campaign visual
- `campaign_summary_YYYY-MM-DD_HH-MM-SS.md` — Executive Markdown report

The report includes:
- 📊 Refined trend insights (executive prose)
- 🎯 Campaign visual (embedded image)
- ✍️ Campaign quote
- ✅ Quote justification

---

## 🔬 Design Principles

| Principle | How it's applied |
|---|---|
| **Autonomy** | Agents run unsupervised; the agentic loop handles tool use dynamically |
| **Creativity** | GraphicDesigner + Copywriter agents produce original visual + text assets |
| **Reliability** | Retry logic in TavilySearchTool; max_iterations cap; injectable dependencies for testing |
| **Testability** | All external clients injected via constructors; 100% unit-testable without API keys |
| **Extensibility** | New tools / agents require only 1 file + `registry.register()` |
