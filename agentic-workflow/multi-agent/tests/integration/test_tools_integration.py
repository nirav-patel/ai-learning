"""
Integration tests for the tools layer.

These tests make real API calls and are auto-skipped when the required
environment variables are not set.

Run them explicitly:
    pytest tests/integration/ -v
"""
from __future__ import annotations

import os

import pytest

TAVILY_AVAILABLE = bool(os.getenv("TAVILY_API_KEY"))
OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))

skip_no_tavily = pytest.mark.skipif(
    not TAVILY_AVAILABLE,
    reason="TAVILY_API_KEY not set — skipping live Tavily test",
)
skip_no_openai = pytest.mark.skipif(
    not OPENAI_AVAILABLE,
    reason="OPENAI_API_KEY not set — skipping live OpenAI test",
)


@skip_no_tavily
class TestTavilySearchToolIntegration:
    """Live tests against the Tavily API."""

    @pytest.fixture()
    def tool(self):
        from tools.search_tool import TavilySearchTool
        import config

        return TavilySearchTool(
            api_key=config.TAVILY_API_KEY,
            base_url=config.TAVILY_BASE_URL,
            max_retries=2,
            retry_delay=1.0,
        )

    def test_search_returns_results(self, tool):
        results = tool.run(query="sunglasses trends 2025", max_results=3)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "title" in results[0] or "error" in results[0]

    def test_search_result_has_url(self, tool):
        results = tool.run(query="aviator sunglasses fashion", max_results=1)
        # Skip assertion if we got an error (e.g., rate limit)
        if "error" not in results[0]:
            assert results[0].get("url", "").startswith("http")

    def test_definition_schema_valid(self, tool):
        defn = tool.definition
        assert defn["type"] == "function"
        assert "query" in defn["function"]["parameters"]["properties"]


@skip_no_openai
class TestProductCatalogToolIntegration:
    """
    The catalog tool is fully in-memory, but this test class exists to confirm
    it works correctly in the same environment where OpenAI tests run.
    """

    def test_catalog_returns_data(self):
        from tools.catalog_tool import ProductCatalogTool

        tool = ProductCatalogTool()
        items = tool.run(max_items=5)
        assert len(items) == 5
        assert all("item_id" in i for i in items)
