"""
Unit tests for the tools layer:
  - TavilySearchTool
  - ProductCatalogTool
  - ToolRegistry
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.base_tool import BaseTool, ToolDefinition
from tools.catalog_tool import ProductCatalogTool
from tools.registry import ToolRegistry
from tools.search_tool import TavilySearchTool


# ─── TavilySearchTool ─────────────────────────────────────────────────────────

class TestTavilySearchTool:
    @pytest.fixture()
    def tool(self):
        with patch("tools.search_tool.TavilyClient") as MockClient:
            MockClient.return_value.search.return_value = {
                "results": [
                    {"title": "Trend A", "content": "Oversized is back.", "url": "https://ex.com/a"}
                ],
                "images": [],
            }
            yield TavilySearchTool(api_key="fake-key", max_retries=2, retry_delay=0.0)

    def test_name(self, tool):
        assert tool.name == "tavily_search_tool"

    def test_definition_structure(self, tool):
        defn = tool.definition
        assert defn["type"] == "function"
        assert defn["function"]["name"] == "tavily_search_tool"
        assert "query" in defn["function"]["parameters"]["properties"]

    def test_run_returns_results(self, tool):
        results = tool.run(query="sunglasses trends")
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["title"] == "Trend A"

    def test_run_with_images(self):
        with patch("tools.search_tool.TavilyClient") as MockClient:
            MockClient.return_value.search.return_value = {
                "results": [],
                "images": ["https://img.example.com/a.jpg"],
            }
            tool = TavilySearchTool(api_key="fake", max_retries=1, retry_delay=0.0)
            results = tool.run(query="q", include_images=True)
            assert any("image_url" in r for r in results)

    def test_retry_on_failure_returns_error(self):
        with patch("tools.search_tool.TavilyClient") as MockClient:
            MockClient.return_value.search.side_effect = Exception("network error")
            tool = TavilySearchTool(api_key="fake", max_retries=2, retry_delay=0.0)
            results = tool.run(query="q")
            assert len(results) == 1
            assert "error" in results[0]

    def test_succeeds_after_one_retry(self):
        with patch("tools.search_tool.TavilyClient") as MockClient:
            search_mock = MockClient.return_value.search
            search_mock.side_effect = [
                Exception("transient error"),
                {"results": [{"title": "OK", "content": "c", "url": "u"}], "images": []},
            ]
            tool = TavilySearchTool(api_key="fake", max_retries=3, retry_delay=0.0)
            results = tool.run(query="q")
            assert results[0]["title"] == "OK"


# ─── ProductCatalogTool ───────────────────────────────────────────────────────

class TestProductCatalogTool:
    def test_name(self, catalog_tool):
        assert catalog_tool.name == "product_catalog_tool"

    def test_definition_structure(self, catalog_tool):
        defn = catalog_tool.definition
        assert defn["type"] == "function"
        assert defn["function"]["name"] == "product_catalog_tool"

    def test_run_returns_list_of_dicts(self, catalog_tool):
        items = catalog_tool.run()
        assert isinstance(items, list)
        assert all(isinstance(item, dict) for item in items)

    def test_run_max_items(self, catalog_tool):
        items = catalog_tool.run(max_items=2)
        assert len(items) == 2

    def test_run_fields(self, catalog_tool):
        item = catalog_tool.run(max_items=1)[0]
        for field in ("name", "item_id", "description", "quantity_in_stock", "price"):
            assert field in item

    def test_seed_reproducibility(self):
        tool1 = ProductCatalogTool(seed=1)
        tool2 = ProductCatalogTool(seed=1)
        assert tool1.run() == tool2.run()


# ─── ToolRegistry ─────────────────────────────────────────────────────────────

class TestToolRegistry:
    def test_register_and_definitions(self, catalog_tool):
        registry = ToolRegistry()
        registry.register(catalog_tool)
        assert len(registry.definitions) == 1
        assert registry.definitions[0]["function"]["name"] == "product_catalog_tool"

    def test_register_chaining(self, catalog_tool, mock_search_tool):
        registry = ToolRegistry().register(mock_search_tool).register(catalog_tool)
        assert len(registry.definitions) == 2

    def test_duplicate_registration_raises(self, catalog_tool):
        registry = ToolRegistry()
        registry.register(catalog_tool)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(ProductCatalogTool())

    def test_deregister(self, catalog_tool):
        registry = ToolRegistry()
        registry.register(catalog_tool)
        registry.deregister(catalog_tool.name)
        assert len(registry.definitions) == 0

    def test_call_dispatches_to_tool(self, catalog_tool):
        registry = ToolRegistry()
        registry.register(catalog_tool)
        result = registry.call("product_catalog_tool", max_items=1)
        assert len(result) == 1

    def test_call_unknown_tool_raises(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.call("unknown_tool")

    def test_dispatch_tool_call_object(self, catalog_tool):
        registry = ToolRegistry()
        registry.register(catalog_tool)
        tool_call = SimpleNamespace(
            id="call_001",
            function=SimpleNamespace(
                name="product_catalog_tool",
                arguments=json.dumps({"max_items": 2}),
            ),
        )
        result = registry.dispatch(tool_call)
        assert len(result) == 2

    def test_build_tool_response_message(self, catalog_tool):
        registry = ToolRegistry()
        registry.register(catalog_tool)
        tool_call = SimpleNamespace(
            id="call_abc",
            function=SimpleNamespace(name="product_catalog_tool", arguments="{}"),
        )
        result_data = [{"name": "Aviator"}]
        msg = registry.build_tool_response_message(tool_call, result_data)
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_abc"
        assert json.loads(msg["content"]) == result_data
