"""
Abstract base class and type definitions for agent tools.

Every concrete tool must:
  1. Subclass ``BaseTool``.
  2. Implement ``definition`` (returns JSON-schema metadata consumed by the LLM).
  3. Implement ``run(**kwargs)`` (executes the tool logic).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict


class ToolDefinition(TypedDict):
    """OpenAI-compatible function-tool schema."""

    type: str  # always "function"
    function: dict[str, Any]


class BaseTool(ABC):
    """Abstract interface for a callable agent tool."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Machine-readable tool name (used as the function name in the schema)."""

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """
        Return the JSON-schema tool descriptor understood by the Anthropic SDK.

        Example::

            {
                "type": "function",
                "function": {
                    "name": "my_tool",
                    "description": "Does something useful.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        """

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the tool with the supplied keyword arguments."""
