"""
Integration tests for the full CampaignPipeline.

These tests make real API calls to OpenAI and Tavily.
They are auto-skipped when the required environment variables are not set.

Run them explicitly:
    pytest tests/integration/ -v -s

Note: These tests will consume API credits.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
TAVILY_AVAILABLE = bool(os.getenv("TAVILY_API_KEY"))
ALL_KEYS_AVAILABLE = OPENAI_AVAILABLE and TAVILY_AVAILABLE

skip_no_keys = pytest.mark.skipif(
    not ALL_KEYS_AVAILABLE,
    reason="OPENAI_API_KEY and/or TAVILY_API_KEY not set — skipping live pipeline test",
)
skip_no_openai = pytest.mark.skipif(
    not OPENAI_AVAILABLE,
    reason="OPENAI_API_KEY not set — skipping live OpenAI test",
)


@skip_no_openai
class TestMarketResearchAgentIntegration:
    """Live test of the MarketResearchAgent using real LLM + tools."""

    def test_produces_non_empty_summary(self, tmp_path):
        import aisuite
        from tools.catalog_tool import ProductCatalogTool
        from tools.registry import ToolRegistry
        from tools.search_tool import TavilySearchTool
        from agents.market_research_agent import MarketResearchAgent
        import config

        registry = ToolRegistry().register(ProductCatalogTool())

        if TAVILY_AVAILABLE:
            registry.register(
                TavilySearchTool(
                    api_key=config.TAVILY_API_KEY,
                    base_url=config.TAVILY_BASE_URL,
                )
            )

        agent = MarketResearchAgent(
            client=aisuite.Client(),
            model=config.AGENT_MODEL,
            registry=registry,
            max_iterations=5,
        )
        result = agent.run()
        assert len(result.content) > 50, "Expected a substantive trend summary"


@skip_no_openai
class TestPackagingAgentIntegration:
    """Live test of the PackagingAgent to confirm it saves a readable Markdown file."""

    def test_report_saved_with_required_sections(self, tmp_path):
        import aisuite
        from agents.packaging_agent import PackagingAgent
        import config

        agent = PackagingAgent(
            client=aisuite.Client(),
            model=config.AGENT_MODEL,
            output_dir=tmp_path,
        )
        path = agent.run(
            trend_summary="Oversized frames and aviators dominate summer fashion.",
            image_path="placeholder.png",
            quote="Own your summer.",
            justification="Matches the oversized trend perfectly.",
            output_filename="integration_test_report.md",
        )
        content = Path(path).read_text()
        assert "Trend Insights" in content
        assert "Own your summer." in content


@skip_no_keys
class TestFullPipelineIntegration:
    """
    End-to-end test of the complete four-agent pipeline.

    This is the most expensive test — it will call Tavily + OpenAI chat
    + OpenAI image generation.
    """

    def test_pipeline_produces_all_outputs(self, tmp_path):
        from pipeline.campaign_pipeline import CampaignPipeline

        pipeline = CampaignPipeline.from_env(output_dir=tmp_path)
        result = pipeline.run(output_filename="integration_campaign.md")

        assert len(result.trend_summary) > 50, "Trend summary should be substantive"
        assert Path(result.image_path).exists(), "Image file should be created"
        assert len(result.quote) > 0, "Quote should not be empty"
        assert Path(result.report_path).exists(), "Report file should be created"

        report_text = Path(result.report_path).read_text()
        assert "Campaign" in report_text
