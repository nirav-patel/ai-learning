"""
Unit tests for all agent classes.

All LLM clients and external tools are mocked.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from agents.base_agent import AgentResult, BaseAgent
from agents.copywriter_agent import CopywriterAgent
from agents.graphic_designer_agent import GraphicDesignerAgent
from agents.market_research_agent import MarketResearchAgent
from agents.packaging_agent import PackagingAgent
from tools.registry import ToolRegistry


# ─── Helpers (also available via conftest) ────────────────────────────────────

def _make_msg(content: str | None = None, tool_calls: list | None = None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)

def _make_resp(content: str | None = None, tool_calls: list | None = None):
    return SimpleNamespace(choices=[SimpleNamespace(message=_make_msg(content, tool_calls))])

def _make_tool_call(name: str, args: dict, call_id: str = "c1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


# ─── BaseAgent (via concrete stub) ────────────────────────────────────────────

class _ConcreteAgent(BaseAgent):
    """Minimal concrete implementation for testing BaseAgent behaviour."""
    @property
    def name(self) -> str:
        return "TestAgent"
    def build_system_prompt(self) -> str:
        return "You are a test agent."
    def build_user_prompt(self, **kwargs) -> str:
        return "Do the thing."


class TestBaseAgent:
    def test_returns_final_answer_immediately(self, mock_llm_client):
        mock_llm_client.chat.completions.create.return_value = _make_resp("Final answer")
        agent = _ConcreteAgent(client=mock_llm_client, model="openai:gpt-4o-mini")
        result = agent.run()
        assert isinstance(result, AgentResult)
        assert result.content == "Final answer"
        assert result.tool_calls_made == 0

    def test_uses_tools_then_returns_final_answer(self, mock_llm_client, tool_registry):
        # First call → tool call; second call → final answer
        tool_call = _make_tool_call("product_catalog_tool", {"max_items": 1})
        mock_llm_client.chat.completions.create.side_effect = [
            _make_resp(None, [tool_call]),
            _make_resp("Here is the answer."),
        ]
        agent = _ConcreteAgent(
            client=mock_llm_client,
            model="openai:gpt-4o-mini",
            registry=tool_registry,
        )
        result = agent.run()
        assert result.content == "Here is the answer."
        assert result.tool_calls_made == 1

    def test_exceeds_max_iterations_raises(self, mock_llm_client, tool_registry):
        tool_call = _make_tool_call("product_catalog_tool", {})
        mock_llm_client.chat.completions.create.return_value = _make_resp(None, [tool_call])
        agent = _ConcreteAgent(
            client=mock_llm_client,
            model="openai:gpt-4o-mini",
            registry=tool_registry,
            max_iterations=2,
        )
        with pytest.raises(RuntimeError, match="max_iterations"):
            agent.run()

    def test_unexpected_response_returns_empty_content(self, mock_llm_client):
        mock_llm_client.chat.completions.create.return_value = _make_resp(None, None)
        agent = _ConcreteAgent(client=mock_llm_client, model="openai:gpt-4o-mini")
        result = agent.run()
        assert result.content == ""


# ─── MarketResearchAgent ──────────────────────────────────────────────────────

class TestMarketResearchAgent:
    def test_name(self, mock_llm_client, tool_registry):
        agent = MarketResearchAgent(mock_llm_client, "openai:gpt-4o-mini", tool_registry)
        assert agent.name == "MarketResearchAgent"

    def test_system_prompt_mentions_market_research(self, mock_llm_client, tool_registry):
        agent = MarketResearchAgent(mock_llm_client, "openai:gpt-4o-mini", tool_registry)
        assert "market research" in agent.build_system_prompt().lower()

    def test_user_prompt_contains_today(self, mock_llm_client, tool_registry):
        agent = MarketResearchAgent(mock_llm_client, "openai:gpt-4o-mini", tool_registry)
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        assert today in agent.build_user_prompt()

    def test_run_returns_agent_result(self, mock_llm_client, tool_registry):
        mock_llm_client.chat.completions.create.return_value = _make_resp("Trend summary text.")
        agent = MarketResearchAgent(mock_llm_client, "openai:gpt-4o-mini", tool_registry)
        result = agent.run()
        assert isinstance(result, AgentResult)
        assert result.content == "Trend summary text."


# ─── GraphicDesignerAgent ─────────────────────────────────────────────────────

class TestGraphicDesignerAgent:
    def _make_agent(self, mock_llm_client, mock_image_client, tmp_output):
        mock_llm_client.chat.completions.create.return_value = _make_resp(
            '{"prompt": "Sunny beach vibes.", "caption": "Own the sun."}'
        )
        return GraphicDesignerAgent(
            llm_client=mock_llm_client,
            image_client=mock_image_client,
            model="openai:gpt-4o-mini",
            image_model="dall-e-3",
            image_size="1024x1024",
            image_quality="standard",
            output_dir=tmp_output,
        )

    def test_run_returns_expected_keys(self, mock_llm_client, mock_image_client, tmp_output):
        agent = self._make_agent(mock_llm_client, mock_image_client, tmp_output)
        result = agent.run(trend_summary="Bold frames are trending.")
        assert "image_path" in result
        assert "prompt" in result
        assert "caption" in result

    def test_run_saves_image_file(self, mock_llm_client, mock_image_client, tmp_output):
        agent = self._make_agent(mock_llm_client, mock_image_client, tmp_output)
        result = agent.run(trend_summary="Trend text", output_filename="test_out.png")
        assert Path(result["image_path"]).exists()

    def test_prompt_and_caption_parsed(self, mock_llm_client, mock_image_client, tmp_output):
        agent = self._make_agent(mock_llm_client, mock_image_client, tmp_output)
        result = agent.run(trend_summary="Text")
        assert result["prompt"] == "Sunny beach vibes."
        assert result["caption"] == "Own the sun."

    def test_malformed_json_falls_back_gracefully(self, mock_llm_client, mock_image_client, tmp_output):
        mock_llm_client.chat.completions.create.return_value = _make_resp("not json at all")
        agent = GraphicDesignerAgent(
            llm_client=mock_llm_client,
            image_client=mock_image_client,
            model="m",
            output_dir=tmp_output,
        )
        result = agent.run(trend_summary="T")
        # Should not raise; falls back to raw content as prompt
        assert result["prompt"] == "not json at all"


# ─── CopywriterAgent ─────────────────────────────────────────────────────────

class TestCopywriterAgent:
    def test_run_returns_expected_keys(self, mock_llm_client, sample_image):
        mock_llm_client.chat.completions.create.return_value = _make_resp(
            '{"quote": "Style is eternal.", "justification": "Matches bold frames trend."}'
        )
        agent = CopywriterAgent(client=mock_llm_client, model="openai:gpt-4o-mini")
        result = agent.run(image_path=sample_image, trend_summary="Bold frames dominate.")
        assert result["quote"] == "Style is eternal."
        assert result["justification"] == "Matches bold frames trend."
        assert "image_path" in result

    def test_image_path_preserved(self, mock_llm_client, sample_image):
        mock_llm_client.chat.completions.create.return_value = _make_resp('{"quote":"Q","justification":"J"}')
        agent = CopywriterAgent(client=mock_llm_client, model="m")
        result = agent.run(image_path=sample_image, trend_summary="T")
        assert result["image_path"] == str(sample_image)

    def test_malformed_json_fallback(self, mock_llm_client, sample_image):
        mock_llm_client.chat.completions.create.return_value = _make_resp("plain text response")
        agent = CopywriterAgent(client=mock_llm_client, model="m")
        result = agent.run(image_path=sample_image, trend_summary="T")
        assert "quote" in result  # falls back gracefully


# ─── PackagingAgent ───────────────────────────────────────────────────────────

class TestPackagingAgent:
    def test_run_creates_markdown_file(self, mock_llm_client, tmp_output):
        mock_llm_client.chat.completions.create.return_value = _make_resp("Executive summary text.")
        agent = PackagingAgent(
            client=mock_llm_client,
            model="openai:gpt-4o-mini",
            output_dir=tmp_output,
        )
        path = agent.run(
            trend_summary="Raw trends",
            image_path="output/image.png",
            quote="Style matters.",
            justification="Bold frames = trend.",
            output_filename="test_report.md",
        )
        assert Path(path).exists()

    def test_report_contains_expected_sections(self, mock_llm_client, tmp_output):
        mock_llm_client.chat.completions.create.return_value = _make_resp("Beautified summary.")
        agent = PackagingAgent(client=mock_llm_client, model="m", output_dir=tmp_output)
        path = agent.run(
            trend_summary="raw",
            image_path="img.png",
            quote="The quote.",
            justification="The why.",
            output_filename="rep.md",
        )
        content = Path(path).read_text()
        assert "Trend Insights" in content
        assert "Campaign Visual" in content
        assert "Campaign Quote" in content
        assert "Why This Works" in content
        assert "The quote." in content

    def test_timestamped_filename_when_not_specified(self, mock_llm_client, tmp_output):
        mock_llm_client.chat.completions.create.return_value = _make_resp("Summary.")
        agent = PackagingAgent(client=mock_llm_client, model="m", output_dir=tmp_output)
        path = agent.run(
            trend_summary="t",
            image_path="i.png",
            quote="q",
            justification="j",
        )
        assert "campaign_summary_" in Path(path).name
