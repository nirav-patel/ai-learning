"""
Tavily web-search tool with exponential-backoff retry.

Usage::

    tool = TavilySearchTool(api_key="tvly-...", base_url=None)
    results = tool.run(query="sunglasses trends 2025", max_results=5)
"""
from __future__ import annotations

import logging
import time
from typing import Any

from tavily import TavilyClient

from .base_tool import BaseTool, ToolDefinition

logger = logging.getLogger(__name__)


class TavilySearchTool(BaseTool):
    """
    Web-search tool powered by Tavily.

    Args:
        api_key: Tavily API key.
        base_url: Optional custom base URL (e.g. DLAI sandbox).
        max_retries: Number of attempts before giving up.
        retry_delay: Base delay (seconds) between attempts; multiplied by attempt index.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.5,
    ) -> None:
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["api_base_url"] = base_url
        self._client = TavilyClient(**kwargs)
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    # ── BaseTool interface ────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "tavily_search_tool"

    @property
    def definition(self) -> ToolDefinition:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Search the web for current fashion trends and news using Tavily.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default 5)",
                            "default": 5,
                        },
                        "include_images": {
                            "type": "boolean",
                            "description": "Whether to include image URLs in results",
                            "default": False,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def run(
        self,
        query: str,
        max_results: int = 5,
        include_images: bool = False,
    ) -> list[dict[str, str]]:
        """
        Search the web and return structured results.

        Args:
            query: Search query.
            max_results: Maximum number of text results.
            include_images: When True, image URLs are appended to the results.

        Returns:
            List of result dicts, each with ``title``, ``content``, ``url``
            keys (and optionally ``image_url``).  On repeated failure a single
            ``{"error": "..."}`` dict is returned.
        """
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = self._client.search(
                    query=query,
                    max_results=max_results,
                    include_images=include_images,
                )
                results: list[dict[str, str]] = [
                    {
                        "title": r.get("title", ""),
                        "content": r.get("content", ""),
                        "url": r.get("url", ""),
                    }
                    for r in response.get("results", [])
                ]
                if include_images:
                    results += [
                        {"image_url": img_url}
                        for img_url in response.get("images", [])
                    ]
                return results

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay * (attempt + 1)
                    logger.warning(
                        "Tavily search attempt %d/%d failed: %s. Retrying in %.1fs…",
                        attempt + 1,
                        self._max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        logger.error("Tavily search failed after %d attempts: %s", self._max_retries, last_error)
        return [{"error": str(last_error)}]
