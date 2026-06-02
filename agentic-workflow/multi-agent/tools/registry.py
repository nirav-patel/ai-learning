"""
Tool registry — registers BaseTool instances and dispatches LLM tool-calls.

Usage::

    registry = ToolRegistry()
    registry.register(TavilySearchTool(...))
    registry.register(ProductCatalogTool())

    # Get the schema list to pass to the LLM (OpenAI-compatible format)
    tools_schema = registry.definitions

    # Dispatch a tool_call object returned by the LLM
    result = registry.dispatch(tool_call)
"""
from __future__ import annotations

import json
import logging
from typing import Any

from .base_tool import BaseTool, ToolDefinition

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Holds a collection of ``BaseTool`` instances and routes LLM tool-calls to them.

    Attributes:
        definitions: List of OpenAI-compatible tool schemas for all registered tools.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, tool: BaseTool) -> "ToolRegistry":
        """
        Add a tool to the registry.

        Args:
            tool: A ``BaseTool`` subclass instance.

        Returns:
            ``self`` to allow chaining.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"A tool named '{tool.name}' is already registered. "
                "Use a different name or deregister the existing tool first."
            )
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)
        return self

    def deregister(self, name: str) -> None:
        """Remove a tool by name (no-op if not found)."""
        self._tools.pop(name, None)

    # ── Schema ────────────────────────────────────────────────────────────────

    @property
    def definitions(self) -> list[ToolDefinition]:
        """Return the JSON-schema list to pass as ``tools=`` to the LLM."""
        return [tool.definition for tool in self._tools.values()]

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def call(self, name: str, **kwargs: Any) -> Any:
        """
        Call a registered tool by name.

        Args:
            name: Tool name as registered.
            **kwargs: Arguments forwarded to ``tool.run()``.

        Returns:
            Whatever the tool's ``run()`` method returns.

        Raises:
            KeyError: If no tool with *name* is registered.
        """
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' is not registered. "
                f"Available tools: {list(self._tools.keys())}"
            )
        return self._tools[name].run(**kwargs)

    def dispatch(self, tool_call: Any) -> Any:
        """
        Dispatch an LLM tool_call object (OpenAI-compatible format).

        Parses ``tool_call.function.name`` and ``tool_call.function.arguments``
        and routes to the matching registered tool.

        Args:
            tool_call: Tool-call object from an LLM completion response.

        Returns:
            Tool result (any JSON-serialisable value).
        """
        name = tool_call.function.name
        kwargs = json.loads(tool_call.function.arguments)
        logger.debug("Dispatching tool '%s' with args: %s", name, kwargs)
        return self.call(name, **kwargs)

    @property
    def anthropic_definitions(self) -> list[dict[str, Any]]:
        """Return Anthropic-compatible tool schemas for ``client.messages.create(tools=...)``."""
        result = []
        for tool in self._tools.values():
            fn = tool.definition["function"]
            result.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn["parameters"],
            })
        return result

    def dispatch_anthropic(self, tool_use_block: Any) -> Any:
        """
        Dispatch an Anthropic ``ToolUseBlock`` (``block.type == "tool_use"``).

        Unlike ``dispatch``, the block's ``input`` field is already a dict —
        no JSON parsing needed.

        Args:
            tool_use_block: Object with ``.name`` and ``.input`` (dict) attributes.

        Returns:
            Tool result (any JSON-serialisable value).
        """
        name = tool_use_block.name
        kwargs = tool_use_block.input
        logger.debug("Dispatching tool '%s' with args: %s", name, kwargs)
        return self.call(name, **kwargs)

    def build_tool_response_message(self, tool_call: Any, result: Any) -> dict[str, Any]:
        """
        Build an OpenAI-compatible ``role=tool`` message dict.

        Args:
            tool_call: Original tool-call object from the LLM.
            result: Value returned by the tool.

        Returns:
            Message dict ready to append to the conversation history.

        Note:
            AWS Bedrock's converse API requires ``toolResult.content[].json`` to be a
            JSON *object* (dict), not an array. Lists are wrapped in ``{"results": [...]}``
            to satisfy this constraint while preserving full fidelity.
        """
        if isinstance(result, list):
            serializable: Any = {"results": result}
        elif not isinstance(result, dict):
            serializable = {"value": result}
        else:
            serializable = result

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": json.dumps(serializable),
        }
