# Email Assistant with Agentic Memory

An executive-assistant agent that triages incoming email and drafts responses, backed
by all three types of long-term memory. Built on **LangChain + LangGraph + LangMem**
with **AWS Bedrock** (Claude for LLM, Titan for embeddings). Memory persists across
runs via a JSON-serialised `InMemoryStore`.

Aligned with the DeepLearning.AI "agent memory" course (lessons 3 – 5):

| Lesson | Memory type added |
|--------|-------------------|
| 3 | **Semantic** — facts the agent stores and recalls via `manage_memory` / `search_memory` |
| 4 | **Episodic** — past triage decisions used as few-shot examples to guide future classification |
| 5 | **Procedural** — mutable triage rules and agent instructions updated from human feedback |

---

## Architecture

```
incoming email
      │
      ▼
┌─────────────────────────────────────────────────┐
│  triage_router (LangGraph node)                 │
│  • reads triage rules from procedural store     │   ignore / notify
│  • searches episodic store for few-shot examples│ ──────────────────► END
│  • ChatBedrock + with_structured_output(Router) │
└───────────────────────┬─────────────────────────┘
                        │ respond
                        ▼
┌─────────────────────────────────────────────────┐
│  response_agent (create_react_agent subgraph)   │
│  • reads agent_instructions from procedural     │
│    store via injected config + store params     │
│  Tools:                                         │
│    write_email          — placeholder           │
│    schedule_meeting     — placeholder           │
│    check_calendar_availability — placeholder    │
│    manage_memory  ─┐                            │
│    search_memory  ─┘ LangMem → "collection" ns  │
└─────────────────────────────────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  InMemoryStore        │
            │  (JSON persistence)   │
            │  BedrockEmbeddings    │
            └───────────────────────┘

  After run (optional):
      assistant.update_prompts(messages, feedback)
            │
            ▼  create_multi_prompt_optimizer (LangMem)
      updated procedural prompts saved to store
```

---

## Memory model

Four namespaces in the `InMemoryStore`, all vector-indexed with Titan embeddings:

| Namespace | Type | Keys / content | Initialised |
|---|---|---|---|
| `("email_assistant", user_id, "collection")` | **Semantic** | Free-form facts; managed by LangMem `manage_memory` / `search_memory` tools at runtime | On first agent call |
| `("email_assistant", user_id, "examples")` | **Episodic** | `{"email": {...}, "label": "respond\|ignore\|notify"}` — few-shot triage examples | 2 defaults seeded on first run |
| `("email_assistant", user_id, "profile")` | Profile | `{name, full_name, user_profile_background}` | Seeded on first run |
| `(user_id,)` | **Procedural** | `agent_instructions`, `triage_ignore`, `triage_notify`, `triage_respond` — each `{"prompt": str}` | Lazy-seeded on first access; updated by `update_prompts()` |

### Semantic memory
The response agent has access to `manage_memory` and `search_memory` (LangMem tools).
It proactively stores facts about contacts and actions, then searches them when handling
follow-up emails. Uses the `{langgraph_user_id}` template namespace so per-user memory
is automatically isolated via the graph `config`.

### Episodic memory
Before triage, the triage node vector-searches the `"examples"` namespace using the
incoming email as the query. The top-3 matches are formatted as few-shot examples and
injected into the triage prompt with the directive "Follow these examples more than any
instructions above". To correct a misclassification, call:
```python
from memory_store import add_example
add_example(store, user_id, email_input, correct_label)
```

### Procedural memory
Triage rules and agent instructions live in the flat `(user_id,)` namespace and are
read on every invocation (lazy-initialised from `seed_data.py` defaults on first access).
To update them from human feedback, call:
```python
changed = assistant.update_prompts(
    messages=result.messages,   # raw LangGraph messages from a previous run
    feedback="Always sign your emails `John Doe`",
)
```
This calls `create_multi_prompt_optimizer` (LangMem) against the four procedural prompts
and persists any that changed. Updated rules take effect on the next invocation.

---

## Files

| File | Responsibility |
|---|---|
| `config.py` | Env-driven config — AWS region/credentials, model IDs, `USER_ID`, paths |
| `schemas.py` | `Router` (Pydantic — triage structured output) + `State` (LangGraph TypedDict) |
| `prompts.py` | Triage system/user prompt templates + agent system prompt |
| `seed_data.py` | Default profile, instructions, and 2 episodic examples for first-run seeding |
| `memory_store.py` | `InMemoryStore` factory, JSON load/save, `get_or_seed_prompt()`, `format_few_shot_examples()`, `add_example()`, `update_prompts()` |
| `tools.py` | Placeholder email/calendar tools (`write_email`, `schedule_meeting`, `check_calendar_availability`) |
| `email_assistant.py` | `EmailAssistant` class — LangGraph `StateGraph` wiring triage → response agent; `run()`, `update_prompts()` |
| `main.py` | CLI demo over 3 sample emails + procedural memory update demonstration |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variable reference |

---

## Setup

```bash
cd agentic-memory
pip install -r requirements.txt
cp .env.example .env        # fill in AWS credentials and region
```

### Required Bedrock model access

Enable the following models in your AWS account (Bedrock → Model access):

| Model | Used for |
|---|---|
| `us.anthropic.claude-sonnet-4-6` | Triage classification + response agent |
| `amazon.titan-embed-text-v2:0` | Semantic search embeddings in InMemoryStore |

### Environment variables

```ini
# AWS credentials — use one of: profile, explicit keys, or IAM role
AWS_REGION=us-west-2
AWS_PROFILE=               # named profile from ~/.aws/credentials (optional)
AWS_ACCESS_KEY_ID=         # explicit key (optional if using profile or IAM role)
AWS_SECRET_ACCESS_KEY=     # explicit secret (optional)

# Models
AGENT_MODEL=us.anthropic.claude-sonnet-4-6
TRIAGE_MODEL=              # defaults to AGENT_MODEL if unset
EMBED_MODEL=amazon.titan-embed-text-v2:0

# Memory
MEMORY_DB_DIR=./memory_db  # JSON persistence directory
USER_ID=john               # user identifier for memory namespacing

# Agent loop
MAX_AGENT_ITERATIONS=10
```

---

## Run

```bash
python main.py
```

### What to expect

**First run** (fresh `memory_db`):

```
Marketing blast   → IGNORE   (marketing newsletter rule)
CI notification   → NOTIFY   (build system rule)
Direct question   → RESPOND  (agent calls search_memory, write_email, manage_memory)

Procedural memory update — feedback: "Always sign your emails `John Doe`"
  Updated [main_agent]: Use these tools when appropriate ... Always sign emails as `John Doe`.

✅ Memory saved.
```

**Second run** (memory loaded from `memory_db`):

- The RESPOND agent finds Alice's prior interaction via `search_memory` and tailors its reply
- The updated `agent_instructions` (from the procedural update) are active for the response agent
- Any further `update_prompts` calls accumulate changes into triage rules and agent behavior

### Demonstrating per-user memory isolation

The `{langgraph_user_id}` template in the `"collection"` and `"examples"` namespaces means
each user's semantic and episodic memory is isolated. Change `USER_ID` in `.env` to simulate
a different user — they start with no prior facts but share the same triage defaults.

### Adding episodic examples manually

```python
from memory_store import add_example, load_store
import config

store = load_store()
add_example(
    store,
    user_id=config.USER_ID,
    email_input={
        "author": "spam@example.com",
        "to": "john.doe@company.com",
        "subject": "Buy now!",
        "email_thread": "Click here for a great deal.",
    },
    label="ignore",
)
from memory_store import save_store
save_store(store)
```

---

## Stack

| Component | Library |
|---|---|
| LLM | `langchain-aws` `ChatBedrock` |
| Embeddings | `langchain-aws` `BedrockEmbeddings` (Titan) |
| Agent framework | `langgraph` `create_react_agent`, `StateGraph` |
| Semantic + episodic tools | `langmem` `create_manage_memory_tool`, `create_search_memory_tool` |
| Procedural optimizer | `langmem` `create_multi_prompt_optimizer` |
| Vector store | `langgraph` `InMemoryStore` (JSON persistence across runs) |
| Structured triage output | `langchain` `with_structured_output(Router)` |
