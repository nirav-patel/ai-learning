"""
Unit tests for the CampaignPipeline orchestrator.

All four agents are replaced with mocks to test the pipeline's
orchestration logic in isolation.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.base_agent import AgentResult
from pipeline.campaign_pipeline import CampaignPipeline, PipelineResult


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_market_research_agent():
    agent = MagicMock()
    agent.run.return_value = AgentResult(
        content="Aviator and Wayfarer are trending for summer.",
        tool_calls_made=2,
    )
    return agent


@pytest.fixture()
def mock_graphic_designer_agent(tmp_output, sample_image):
    agent = MagicMock()
    agent.run.return_value = {
        "image_path": str(sample_image),
        "prompt": "Sunlit aviators on a beach.",
        "caption": "Own your summer.",
    }
    return agent


@pytest.fixture()
def mock_copywriter_agent():
    agent = MagicMock()
    agent.run.return_value = {
        "quote": "See the world in style.",
        "justification": "Matches the oversized trend perfectly.",
        "image_path": "output/img.png",
    }
    return agent


@pytest.fixture()
def mock_packaging_agent(tmp_output):
    agent = MagicMock()
    report_path = str(tmp_output / "campaign_summary.md")
    (tmp_output / "campaign_summary.md").write_text("# Report")
    agent.run.return_value = report_path
    return agent


@pytest.fixture()
def pipeline(
    mock_market_research_agent,
    mock_graphic_designer_agent,
    mock_copywriter_agent,
    mock_packaging_agent,
):
    return CampaignPipeline(
        market_research_agent=mock_market_research_agent,
        graphic_designer_agent=mock_graphic_designer_agent,
        copywriter_agent=mock_copywriter_agent,
        packaging_agent=mock_packaging_agent,
    )


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestCampaignPipeline:
    def test_run_returns_pipeline_result(self, pipeline):
        result = pipeline.run()
        assert isinstance(result, PipelineResult)

    def test_trend_summary_passed_from_market_research(self, pipeline, mock_market_research_agent):
        result = pipeline.run()
        assert result.trend_summary == mock_market_research_agent.run.return_value.content

    def test_graphic_designer_receives_trend_summary(
        self, pipeline, mock_market_research_agent, mock_graphic_designer_agent
    ):
        pipeline.run()
        mock_graphic_designer_agent.run.assert_called_once()
        call_kwargs = mock_graphic_designer_agent.run.call_args.kwargs
        assert call_kwargs["trend_summary"] == mock_market_research_agent.run.return_value.content

    def test_copywriter_receives_image_path_and_trend_summary(
        self, pipeline, mock_market_research_agent, mock_graphic_designer_agent, mock_copywriter_agent
    ):
        pipeline.run()
        mock_copywriter_agent.run.assert_called_once()
        call_kwargs = mock_copywriter_agent.run.call_args.kwargs
        assert call_kwargs["trend_summary"] == mock_market_research_agent.run.return_value.content
        assert call_kwargs["image_path"] == mock_graphic_designer_agent.run.return_value["image_path"]

    def test_packaging_receives_all_inputs(
        self,
        pipeline,
        mock_market_research_agent,
        mock_graphic_designer_agent,
        mock_copywriter_agent,
        mock_packaging_agent,
    ):
        pipeline.run()
        mock_packaging_agent.run.assert_called_once()
        call_kwargs = mock_packaging_agent.run.call_args.kwargs
        assert call_kwargs["trend_summary"] == mock_market_research_agent.run.return_value.content
        assert call_kwargs["image_path"] == mock_graphic_designer_agent.run.return_value["image_path"]
        assert call_kwargs["quote"] == mock_copywriter_agent.run.return_value["quote"]
        assert call_kwargs["justification"] == mock_copywriter_agent.run.return_value["justification"]

    def test_result_fields_populated(self, pipeline, tmp_output):
        result = pipeline.run()
        assert result.image_path != ""
        assert result.quote == "See the world in style."
        assert result.report_path != ""

    def test_metadata_contains_duration(self, pipeline):
        result = pipeline.run()
        assert "duration_seconds" in result.metadata
        assert result.metadata["duration_seconds"] >= 0

    def test_custom_caption_style_forwarded(self, pipeline, mock_graphic_designer_agent):
        pipeline.run(caption_style="elegant and timeless")
        call_kwargs = mock_graphic_designer_agent.run.call_args.kwargs
        assert call_kwargs["caption_style"] == "elegant and timeless"

    def test_all_four_agents_called_exactly_once(
        self,
        pipeline,
        mock_market_research_agent,
        mock_graphic_designer_agent,
        mock_copywriter_agent,
        mock_packaging_agent,
    ):
        pipeline.run()
        mock_market_research_agent.run.assert_called_once()
        mock_graphic_designer_agent.run.assert_called_once()
        mock_copywriter_agent.run.assert_called_once()
        mock_packaging_agent.run.assert_called_once()
