"""
Integration tests for the full CampaignPipeline.

These tests make real API calls to AWS Bedrock and Tavily.
They are auto-skipped when the required environment variables are not set.

Run them explicitly:
    pytest tests/integration/ -v -s

Note: These tests will consume AWS Bedrock and Tavily API credits.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

AWS_AVAILABLE = bool(os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE"))
TAVILY_AVAILABLE = bool(os.getenv("TAVILY_API_KEY"))
ALL_KEYS_AVAILABLE = AWS_AVAILABLE and TAVILY_AVAILABLE

skip_no_keys = pytest.mark.skipif(
    not ALL_KEYS_AVAILABLE,
    reason="AWS credentials and/or TAVILY_API_KEY not set — skipping live pipeline test",
)
skip_no_aws = pytest.mark.skipif(
    not AWS_AVAILABLE,
    reason="AWS credentials not set — skipping live Bedrock test",
)


@skip_no_aws
class TestMarketResearchAgentIntegration:
    """Live test of the MarketResearchAgent using real Bedrock LLM + tools."""

    def test_produces_non_empty_summary(self, tmp_path):
        import anthropic
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

        anthropic_kwargs: dict = {"aws_region": config.AWS_REGION}
        if config.AWS_ACCESS_KEY_ID and config.AWS_SECRET_ACCESS_KEY:
            anthropic_kwargs["aws_access_key"] = config.AWS_ACCESS_KEY_ID
            anthropic_kwargs["aws_secret_key"] = config.AWS_SECRET_ACCESS_KEY

        agent = MarketResearchAgent(
            client=anthropic.AnthropicBedrock(**anthropic_kwargs),
            model=config.AGENT_MODEL,
            registry=registry,
            max_iterations=5,
        )
        result = agent.run()
        assert len(result.content) > 50, "Expected a substantive trend summary"


@skip_no_aws
class TestPackagingAgentIntegration:
    """Live test of the PackagingAgent to confirm it saves a readable Markdown file."""

    def test_report_saved_with_required_sections(self, tmp_path):
        import anthropic
        from agents.packaging_agent import PackagingAgent
        import config

        anthropic_kwargs: dict = {"aws_region": config.AWS_REGION}
        if config.AWS_ACCESS_KEY_ID and config.AWS_SECRET_ACCESS_KEY:
            anthropic_kwargs["aws_access_key"] = config.AWS_ACCESS_KEY_ID
            anthropic_kwargs["aws_secret_key"] = config.AWS_SECRET_ACCESS_KEY

        agent = PackagingAgent(
            client=anthropic.AnthropicBedrock(**anthropic_kwargs),
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

    This is the most expensive test — it calls Tavily + AWS Bedrock Claude (text)
    + AWS Bedrock Titan (image generation) + Anthropic Bedrock (multimodal).
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
