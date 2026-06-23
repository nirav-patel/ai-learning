"""
Email assistant built on LangChain + LangGraph + LangMem with AWS Bedrock.

Aligned with lesson_3 of the DeepLearning.AI "agent memory" course:
    - LangMem tools use the "{langgraph_user_id}" template namespace
    - Graph is invoked with config={"configurable": {"langgraph_user_id": ...}}
    - Agent prompt matches lesson_3's simpler semantic-memory version
    - Triage respond message mirrors lesson_3: f"Respond to the email {email_input}"

Two-stage LangGraph flow:
    1. triage_router  — ChatBedrock + with_structured_output(Router)
    2. response_agent — create_react_agent with email/calendar + LangMem memory tools

Memory persists across runs via JSON file (InMemoryStore + BedrockEmbeddings).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Literal

import boto3
from langchain_aws import ChatBedrock
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from langmem import create_manage_memory_tool, create_search_memory_tool

import config
from memory_store import (
    INSTRUCTIONS_NS,
    PROFILE_NS,
    format_few_shot_examples,
    get_instructions,
    get_or_seed_prompt,
    get_profile,
    update_prompts as _update_prompts,
)
from prompts import agent_system_prompt_memory, triage_system_prompt, triage_user_prompt
from schemas import Router, State
from seed_data import DEFAULT_INSTRUCTIONS, DEFAULT_PROFILE
from tools import check_calendar_availability, schedule_meeting, write_email

logger = logging.getLogger(__name__)

# LangGraph config key used to resolve the {langgraph_user_id} namespace template
_GRAPH_CONFIG = {"configurable": {"langgraph_user_id": config.USER_ID}}

# Semantic memory namespace — mirrors lesson_3 exactly
_COLLECTION_NS = ("email_assistant", "{langgraph_user_id}", "collection")


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class TriageResult:
    classification: str
    reasoning: str
    answer: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    messages: list = field(default_factory=list)  # raw LangGraph messages for update_prompts


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_email(email_input: dict) -> tuple[str, str, str, str]:
    return (
        email_input["author"],
        email_input["to"],
        email_input["subject"],
        email_input["email_thread"],
    )


# ── Main assistant class ──────────────────────────────────────────────────────

class EmailAssistant:
    """Triage + respond email assistant powered by LangGraph and LangMem."""

    def __init__(self, store: InMemoryStore) -> None:
        self.store = store

        # boto3 client — clears empty AWS_PROFILE env var set by dotenv
        if not config.AWS_PROFILE:
            os.environ.pop("AWS_PROFILE", None)
        _bedrock = boto3.Session(region_name=config.AWS_REGION).client("bedrock-runtime")

        self._llm = ChatBedrock(model_id=config.AGENT_MODEL, client=_bedrock)
        triage_llm = ChatBedrock(model_id=config.TRIAGE_MODEL, client=_bedrock)
        self._llm_router = triage_llm.with_structured_output(Router)

        # LangMem tools using {langgraph_user_id} template namespace (lesson_3 pattern)
        memory_tools = [
            create_manage_memory_tool(namespace=_COLLECTION_NS),
            create_search_memory_tool(namespace=_COLLECTION_NS),
        ]

        # Response agent — store passed here so LangMem tools can resolve the namespace
        response_agent = create_react_agent(
            model=self._llm,
            tools=[write_email, schedule_meeting, check_calendar_availability] + memory_tools,
            prompt=self._build_agent_prompt,
            store=store,
        )

        builder = StateGraph(State)
        builder.add_node("triage_router", self._triage_node)
        builder.add_node("response_agent", response_agent)
        builder.add_edge(START, "triage_router")
        self._graph = builder.compile(store=store)

    # ── Agent system prompt (lesson_5: reads procedural memory from store) ───────

    def _build_agent_prompt(self, state: State, config: RunnableConfig, store: BaseStore) -> list:
        """Dynamic system prompt — reads agent_instructions from the procedural
        (user_id,) namespace, lazy-initialising from defaults on first access."""
        user_id = config["configurable"]["langgraph_user_id"]
        profile = get_profile(self.store)
        instructions = get_or_seed_prompt(store, user_id, "agent_instructions")
        system = agent_system_prompt_memory.format(
            full_name=profile["full_name"],
            name=profile["name"],
            instructions=instructions,
        )
        return [{"role": "system", "content": system}] + list(state["messages"])

    # ── Triage node ───────────────────────────────────────────────────────────

    def _triage_node(
        self, state: State, config: RunnableConfig, store: BaseStore
    ) -> Command[Literal["response_agent", "__end__"]]:
        author, to, subject, email_thread = _parse_email(state["email_input"])
        profile = get_profile(self.store)

        # Procedural memory (lesson_5): read triage rules from (user_id,) namespace,
        # lazy-seeding defaults on first access so the store is always authoritative.
        user_id = config["configurable"]["langgraph_user_id"]
        ignore_prompt  = get_or_seed_prompt(store, user_id, "triage_ignore")
        notify_prompt  = get_or_seed_prompt(store, user_id, "triage_notify")
        respond_prompt = get_or_seed_prompt(store, user_id, "triage_respond")

        # Episodic memory (lesson_4): vector-search the "examples" namespace.
        examples_ns = ("email_assistant", user_id, "examples")
        raw_examples = store.search(
            examples_ns, query=str({"email": state["email_input"]}), limit=3
        )
        examples_str = format_few_shot_examples(raw_examples)

        sys_prompt = triage_system_prompt.format(
            full_name=profile["full_name"],
            name=profile["name"],
            user_profile_background=profile["user_profile_background"],
            triage_no=ignore_prompt,
            triage_notify=notify_prompt,
            triage_email=respond_prompt,
            examples=examples_str,
        )
        user_prompt = triage_user_prompt.format(
            author=author, to=to, subject=subject, email_thread=email_thread
        )

        result: Router = self._llm_router.invoke([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ])

        if result.classification == "respond":
            logger.info("📧 RESPOND — drafting a reply")
            return Command(
                goto="response_agent",
                update={
                    "classification": result.classification,
                    "reasoning": result.reasoning,
                    # Mirrors lesson_3 cell 44 exactly
                    "messages": [{
                        "role": "user",
                        "content": f"Respond to the email {state['email_input']}",
                    }],
                },
            )
        elif result.classification == "ignore":
            logger.info("🚫 IGNORE — this email can be safely ignored")
        else:
            logger.info("🔔 NOTIFY — %s: %s", author, subject)

        return Command(
            goto=END,
            update={"classification": result.classification, "reasoning": result.reasoning},
        )

    # ── Procedural memory update (lesson_5) ──────────────────────────────────

    def update_prompts(self, messages: list, feedback: str) -> dict[str, str]:
        """Update procedural prompts based on a conversation + human feedback.

        Mirrors lesson_5's create_multi_prompt_optimizer pattern. Pass the message
        history from a previous run and a natural-language feedback string. Returns
        a dict of prompt names → new text for every prompt that changed.

        Example:
            result = assistant.run(email)
            changed = assistant.update_prompts(
                messages=graph_messages,
                feedback="Always sign your emails `John Doe`",
            )
        """
        return _update_prompts(
            store=self.store,
            user_id=config.USER_ID,
            llm=self._llm,
            conversations=[(messages, feedback)],
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, email_input: dict) -> TriageResult:
        # Pass langgraph_user_id via config so LangMem resolves the namespace template
        final_state = self._graph.invoke(
            {"email_input": email_input, "messages": []},
            config=_GRAPH_CONFIG,
        )

        classification = final_state.get("classification", "unknown")
        reasoning = final_state.get("reasoning", "")
        messages: list[AnyMessage] = final_state.get("messages", [])

        answer = ""
        for msg in reversed(messages):
            role = getattr(msg, "type", None) or (
                msg.get("role") if isinstance(msg, dict) else None
            )
            if role in ("ai", "assistant"):
                answer = getattr(msg, "content", "") or (
                    msg.get("content", "") if isinstance(msg, dict) else ""
                )
                if isinstance(answer, list):
                    answer = " ".join(
                        block.get("text", "") for block in answer if isinstance(block, dict)
                    )
                break

        tool_calls = []
        for msg in messages:
            calls = getattr(msg, "tool_calls", None)
            if calls:
                tool_calls.extend({"name": c["name"], "args": c["args"]} for c in calls)

        return TriageResult(
            classification=classification,
            reasoning=reasoning,
            answer=answer,
            tool_calls=tool_calls,
            messages=messages,
        )
