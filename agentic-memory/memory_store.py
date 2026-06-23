"""
InMemoryStore wrapper with Bedrock embeddings and JSON persistence.

Namespaces:
    ("email_assistant", user_id, "collection")  — semantic memory (LangMem tools)
    ("email_assistant", user_id, "examples")    — episodic memory (triage few-shots)
    ("email_assistant", user_id, "profile")     — user profile singleton
    ("email_assistant", user_id, "instructions")— agent profile/background (legacy)
    (user_id,)                                  — procedural memory (lesson_5 pattern)
        keys: "agent_instructions", "triage_ignore", "triage_notify", "triage_respond"
        values: {"prompt": "..."}  — lazy-initialized on first access, updated by
                                     update_prompts() via create_multi_prompt_optimizer
"""
from __future__ import annotations

import json
import logging
import os
import uuid

import boto3
from langchain_aws import BedrockEmbeddings
from langchain_core.language_models import BaseChatModel
from langgraph.store.memory import InMemoryStore
from langmem import create_multi_prompt_optimizer

import config
from seed_data import DEFAULT_EXAMPLES, DEFAULT_INSTRUCTIONS, DEFAULT_PROFILE

logger = logging.getLogger(__name__)

# ── Namespace constants ───────────────────────────────────────────────────────

def _ns(kind: str) -> tuple[str, ...]:
    return ("email_assistant", config.USER_ID, kind)


PROFILE_NS = _ns("profile")
INSTRUCTIONS_NS = _ns("instructions")

_STORE_FILE = config.MEMORY_DB_DIR / "store.json"


# ── Store factory ─────────────────────────────────────────────────────────────

def _bedrock_client():
    """Build a boto3 bedrock-runtime client, clearing empty AWS_PROFILE env var."""
    if not config.AWS_PROFILE:
        os.environ.pop("AWS_PROFILE", None)
    session = boto3.Session(region_name=config.AWS_REGION)
    return session.client("bedrock-runtime")


def _make_store() -> InMemoryStore:
    embeddings = BedrockEmbeddings(
        model_id=config.EMBED_MODEL,
        client=_bedrock_client(),
    )
    return InMemoryStore(index={"dims": 1024, "embed": embeddings})


# ── Load / save ───────────────────────────────────────────────────────────────

def load_store() -> InMemoryStore:
    """Load persisted items into a fresh InMemoryStore, or seed defaults."""
    store = _make_store()

    if _STORE_FILE.exists():
        logger.info("Loading memory from %s", _STORE_FILE)
        with open(_STORE_FILE) as f:
            items = json.load(f)
        for item in items:
            store.put(
                namespace=tuple(item["namespace"]),
                key=item["key"],
                value=item["value"],
            )
    else:
        logger.info("No existing store — seeding defaults")
        _seed_singletons(store)

    return store


def save_store(store: InMemoryStore) -> None:
    """Serialise the store's items to JSON for cross-run persistence."""
    config.MEMORY_DB_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    for namespace in store.list_namespaces():
        for item in store.search(namespace, limit=10_000):
            items.append({
                "namespace": list(item.namespace),
                "key": item.key,
                "value": item.value,
            })
    with open(_STORE_FILE, "w") as f:
        json.dump(items, f, indent=2, default=str)
    logger.debug("Saved %d memory items to %s", len(items), _STORE_FILE)


# ── Singletons ────────────────────────────────────────────────────────────────

def _seed_singletons(store: InMemoryStore) -> None:
    store.put(namespace=PROFILE_NS, key="profile", value=DEFAULT_PROFILE)
    store.put(namespace=INSTRUCTIONS_NS, key="instructions", value=DEFAULT_INSTRUCTIONS)
    examples_ns = _ns("examples")
    for example in DEFAULT_EXAMPLES:
        store.put(namespace=examples_ns, key=str(uuid.uuid4()), value=example)


def get_profile(store: InMemoryStore) -> dict:
    item = store.get(namespace=PROFILE_NS, key="profile")
    return item.value if item else dict(DEFAULT_PROFILE)


def get_instructions(store: InMemoryStore) -> dict:
    item = store.get(namespace=INSTRUCTIONS_NS, key="instructions")
    return item.value if item else dict(DEFAULT_INSTRUCTIONS)


# ── Episodic memory helpers ───────────────────────────────────────────────────

# Template mirrors lesson_4 exactly
_EXAMPLE_TEMPLATE = """Email Subject: {subject}
Email From: {from_email}
Email To: {to_email}
Email Content:
```
{content}
```
> Triage Result: {result}"""


def format_few_shot_examples(examples: list) -> str:
    """Format episodic store items into a few-shot block for the triage prompt."""
    if not examples:
        return "No examples available."
    strs = ["Here are some previous examples:"]
    for eg in examples:
        val = eg.value
        strs.append(
            _EXAMPLE_TEMPLATE.format(
                subject=val["email"]["subject"],
                from_email=val["email"]["author"],
                to_email=val["email"]["to"],
                content=val["email"]["email_thread"][:400],
                result=val["label"],
            )
        )
    return "\n\n------------\n\n".join(strs)


def add_example(store: InMemoryStore, user_id: str, email_input: dict, label: str) -> None:
    """Persist a triage decision as an episodic few-shot example."""
    ns = ("email_assistant", user_id, "examples")
    store.put(namespace=ns, key=str(uuid.uuid4()), value={"email": email_input, "label": label})


# ── Procedural memory helpers (lesson_5 pattern) ──────────────────────────────
#
# Procedural memory lives in the flat (user_id,) namespace with four keys.
# Values are {"prompt": str}. Reads lazy-initialize from DEFAULT_INSTRUCTIONS
# on first access so the store is always authoritative after the first run.

_PROC_KEYS = {
    "agent_instructions": lambda: DEFAULT_INSTRUCTIONS["agent_instructions"],
    "triage_ignore":      lambda: DEFAULT_INSTRUCTIONS["triage_rules"]["ignore"],
    "triage_notify":      lambda: DEFAULT_INSTRUCTIONS["triage_rules"]["notify"],
    "triage_respond":     lambda: DEFAULT_INSTRUCTIONS["triage_rules"]["respond"],
}


def get_or_seed_prompt(store: InMemoryStore, user_id: str, key: str) -> str:
    """Return the current procedural prompt for `key`, seeding the default if absent."""
    ns = (user_id,)
    item = store.get(namespace=ns, key=key)
    if item is None:
        default = _PROC_KEYS[key]()
        store.put(namespace=ns, key=key, value={"prompt": default})
        return default
    return item.value["prompt"]


def update_prompts(
    store: InMemoryStore,
    user_id: str,
    llm: BaseChatModel,
    conversations: list[tuple],
) -> dict[str, str]:
    """Run the multi-prompt optimizer and persist any changed procedural prompts.

    Args:
        store:         The InMemoryStore instance.
        user_id:       The user whose procedural namespace to update.
        llm:           A ChatBedrock (or any BaseChatModel) used by the optimizer.
        conversations: List of (messages, feedback_str) tuples — mirrors lesson_5.

    Returns:
        Dict mapping prompt name → new prompt text for every prompt that changed.
    """
    ns = (user_id,)
    _keep_short = "Keep the instructions short and to the point."
    prompts = [
        {
            "name": "main_agent",
            "prompt": get_or_seed_prompt(store, user_id, "agent_instructions"),
            "update_instructions": _keep_short,
            "when_to_update": (
                "Update this prompt whenever there is feedback on how the agent "
                "should write emails or schedule events."
            ),
        },
        {
            "name": "triage-ignore",
            "prompt": get_or_seed_prompt(store, user_id, "triage_ignore"),
            "update_instructions": _keep_short,
            "when_to_update": (
                "Update this prompt whenever there is feedback on which emails should be ignored."
            ),
        },
        {
            "name": "triage-notify",
            "prompt": get_or_seed_prompt(store, user_id, "triage_notify"),
            "update_instructions": _keep_short,
            "when_to_update": (
                "Update this prompt whenever there is feedback on which emails the "
                "user should be notified of."
            ),
        },
        {
            "name": "triage-respond",
            "prompt": get_or_seed_prompt(store, user_id, "triage_respond"),
            "update_instructions": _keep_short,
            "when_to_update": (
                "Update this prompt whenever there is feedback on which emails should be responded to."
            ),
        },
    ]

    _key_map = {
        "main_agent":     "agent_instructions",
        "triage-ignore":  "triage_ignore",
        "triage-notify":  "triage_notify",
        "triage-respond": "triage_respond",
    }

    optimizer = create_multi_prompt_optimizer(llm, kind="prompt_memory")
    updated_list = optimizer.invoke({"trajectories": conversations, "prompts": prompts})

    changed: dict[str, str] = {}
    for updated_prompt, old_prompt in zip(updated_list, prompts):
        if updated_prompt["prompt"] != old_prompt["prompt"]:
            name = old_prompt["name"]
            store.put(namespace=ns, key=_key_map[name], value={"prompt": updated_prompt["prompt"]})
            logger.info("Procedural memory updated: %s", name)
            changed[name] = updated_prompt["prompt"]

    return changed
