"""
Abstract BaseAgent with a generic agentic loop.

The loop follows the standard ReAct / tool-use pattern:
  1. Call the LLM with the current conversation history.
  2. If the LLM returns a final text answer → return it.
  3. If the LLM returns tool-calls → dispatch each via the ToolRegistry,
     append the results to the conversation, and loop back to step 1.
  4. If neither → raise or return an error sentinel.

Concrete agents only need to implement ``build_system_prompt`` and
``build_user_prompt`` (and optionally override ``_post_process``).
"""
from __future__ import annotations

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
    Generic text-based agentic loop.

    Args:
        client: An aisuite (or OpenAI-compatible) client with a
                ``chat.completions.create`` method.
        model: Model identifier string (e.g. ``"openai:gpt-4o-mini"``).
        registry: ``ToolRegistry`` with all tools this agent is allowed to call.
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

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(**kwargs)},
        ]

        tools_schema = self._registry.definitions if self._registry else None
        tool_calls_made = 0

        for iteration in range(1, self._max_iterations + 1):
            logger.debug("[%s] Iteration %d", self.name, iteration)

            call_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
            }
            if tools_schema:
                call_kwargs["tools"] = tools_schema
                call_kwargs["tool_choice"] = "auto"

            response = self._client.chat.completions.create(**call_kwargs)
            msg = response.choices[0].message

            # ── Final text answer ─────────────────────────────────────────────
            if msg.content and not getattr(msg, "tool_calls", None):
                logger.info("[%s] Finished in %d iteration(s)", self.name, iteration)
                result = AgentResult(
                    content=msg.content,
                    messages=messages + [{"role": "assistant", "content": msg.content}],
                    tool_calls_made=tool_calls_made,
                )
                return self._post_process(result, **kwargs)

            # ── Tool calls ────────────────────────────────────────────────────
            if getattr(msg, "tool_calls", None):
                messages.append(msg)
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    logger.debug("[%s] Calling tool: %s", self.name, tool_name)

                    result_data = self._registry.dispatch(tool_call)
                    tool_calls_made += 1

                    messages.append(
                        self._registry.build_tool_response_message(tool_call, result_data)
                    )
                continue

            # ── Unexpected response ───────────────────────────────────────────
            logger.warning(
                "[%s] Unexpected response (no content, no tool_calls) at iteration %d",
                self.name,
                iteration,
            )
            # Treat as a final answer with an empty string so the pipeline can continue
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
