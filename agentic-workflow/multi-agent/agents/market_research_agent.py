"""
Market Research Agent.

Discovers the latest sunglasses fashion trends via web search (Tavily) and
cross-references them with the internal product catalog.  Returns a structured
trend summary that downstream agents (Graphic Designer, Copywriter) can use.

Tools used:
  - ``tavily_search_tool`` — external web search
  - ``product_catalog_tool`` — internal inventory catalog
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from tools.registry import ToolRegistry

from .base_agent import AgentResult, BaseAgent


class MarketResearchAgent(BaseAgent):
    """
    Uses the web + catalog tools to produce a trend summary.

    Args:
        client: aisuite-compatible LLM client.
        model: Model identifier.
        registry: ``ToolRegistry`` containing at least ``tavily_search_tool``
                  and ``product_catalog_tool``.
        max_iterations: Safety cap on LLM iterations.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        registry: ToolRegistry,
        max_iterations: int = 10,
    ) -> None:
        super().__init__(client, model, registry, max_iterations)

    # ── BaseAgent interface ───────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "MarketResearchAgent"

    def build_system_prompt(self) -> str:
        return (
            "You are a fashion market research agent. "
            "Your job is to identify current sunglasses trends and recommend "
            "products from the internal catalog that best match those trends. "
            "Be concise, data-driven, and professional."
        )

    def build_user_prompt(self, **kwargs: Any) -> str:  # noqa: ARG002
        today = datetime.now().strftime("%Y-%m-%d")
        return f"""
You are preparing a trend analysis for a summer sunglasses marketing campaign.

Your tasks:
1. Search the web for the top current sunglasses fashion trends (today is {today}).
2. Retrieve the internal product catalog.
3. Identify which catalog products best align with the discovered trends.

Once done, write a concise summary that includes:
- The **top 2–3 trends** you found (with supporting evidence).
- The **recommended products** from the catalog (name + item_id) that match each trend.
- A brief **justification** for each recommendation.

Keep your summary under 400 words and suitable for a marketing executive audience.
""".strip()
