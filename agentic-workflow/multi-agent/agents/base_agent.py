"""
Abstract BaseAgent with a generic agentic loop using the Anthropic Bedrock SDK.

The loop follows the standard ReAct / tool-use pattern:
  1. Call the LLM with the current conversation history.
  2. If stop_reason == "end_turn"  → extract text and return.
  3. If stop_reason == "tool_use"  → dispatch ALL tools, batch all results into
     a SINGLE user message (required by Bedrock), and loop back to step 1.

Concrete agents only need to implement ``build_system_prompt`` and
``build_user_prompt`` (and optionally override ``_post_process``).
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """
    Container for the output of a single agent run.

    Attributes:
        content: Final text output from the LLM.
        messages: Full conversation history (useful for multi-agent handoffs).
        tool_calls_made: Number of tool invocations during this run.
    """

    content: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_made: int = 0


class BaseAgent(ABC):
    """
    Generic text-based agentic loop backed by ``anthropic.AnthropicBedrock``.

    Args:
        client: An ``anthropic.AnthropicBedrock`` client instance.
        model: Bare Bedrock model ID (e.g. ``"us.anthropic.claude-sonnet-4-6"``).
        registry: ``ToolRegistry`` with all tools this agent may call.
                  If ``None``, the agent runs without tools.
        max_iterations: Safety cap on LLM + tool-dispatch rounds.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        registry: ToolRegistry | None = None,
        max_iterations: int = 10,
    ) -> None:
        self._client = client
        self._model = model
        self._registry = registry
        self._max_iterations = max_iterations

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name used in log messages."""

    @abstractmethod
    def build_system_prompt(self) -> str:
        """Return the system-role prompt for this agent."""

    @abstractmethod
    def build_user_prompt(self, **kwargs: Any) -> str:
        """
        Build the initial user-role message.

        Concrete agents receive their upstream inputs via **kwargs.
        """

    # ── Optional override ─────────────────────────────────────────────────────

    def _post_process(self, result: AgentResult, **kwargs: Any) -> AgentResult:
        """
        Optional hook called after the agentic loop completes.

        Override to parse / enrich ``result.content`` before it is returned.
        Default implementation is a no-op.
        """
        return result

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, **kwargs: Any) -> AgentResult:
        """
        Execute the agentic loop and return an ``AgentResult``.

        Args:
            **kwargs: Inputs forwarded to ``build_user_prompt``.

        Returns:
            ``AgentResult`` with the final answer and conversation history.

        Raises:
            RuntimeError: If the loop exceeds ``max_iterations`` without a
                          final answer.
        """
        logger.info("[%s] Starting run", self.name)

        system = self.build_system_prompt()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": self.build_user_prompt(**kwargs)},
        ]

        tools = self._registry.anthropic_definitions if self._registry else None
        tool_calls_made = 0

        for iteration in range(1, self._max_iterations + 1):
            logger.debug("[%s] Iteration %d", self.name, iteration)

            call_kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": 4096,
                "system": system,
                "messages": messages,
            }
            if tools:
                call_kwargs["tools"] = tools

            response = self._client.messages.create(**call_kwargs)

            # ── Tool use ──────────────────────────────────────────────────────
            if response.stop_reason == "tool_use":
                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

                # Preserve the full assistant message (contains ToolUseBlock objects)
                messages.append({"role": "assistant", "content": response.content})

                # All tool results MUST go into a single user message — Bedrock
                # rejects consecutive user messages with separate toolResult blocks.
                tool_results = []
                for block in tool_use_blocks:
                    logger.debug("[%s] Calling tool: %s", self.name, block.name)
                    result_data = self._registry.dispatch_anthropic(block)
                    tool_calls_made += 1

                    if isinstance(result_data, (dict, list)):
                        result_str = json.dumps(result_data)
                    else:
                        result_str = str(result_data)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

                messages.append({"role": "user", "content": tool_results})
                continue

            # ── Final text answer ─────────────────────────────────────────────
            if response.stop_reason == "end_turn":
                text_blocks = [b for b in response.content if b.type == "text"]
                content = "\n".join(b.text for b in text_blocks)
                logger.info("[%s] Finished in %d iteration(s)", self.name, iteration)
                result = AgentResult(
                    content=content,
                    messages=messages + [{"role": "assistant", "content": response.content}],
                    tool_calls_made=tool_calls_made,
                )
                return self._post_process(result, **kwargs)

            # ── Unexpected stop reason ────────────────────────────────────────
            logger.warning(
                "[%s] Unexpected stop_reason '%s' at iteration %d",
                self.name,
                response.stop_reason,
                iteration,
            )
            result = AgentResult(
                content="",
                messages=messages,
                tool_calls_made=tool_calls_made,
            )
            return self._post_process(result, **kwargs)

        raise RuntimeError(
            f"[{self.name}] Exceeded max_iterations ({self._max_iterations}) "
            "without reaching a final answer."
        )
