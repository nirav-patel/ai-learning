"""
Campaign Pipeline — orchestrates all four agents in sequence.

Flow:
  MarketResearchAgent → GraphicDesignerAgent → CopywriterAgent → PackagingAgent

Each agent's output is fed as input to the next. The pipeline can be
constructed manually (for tests) or via the ``from_env`` factory which
reads all credentials and settings from environment variables / config.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aisuite

import config
from agents.copywriter_agent import CopywriterAgent
from agents.graphic_designer_agent import GraphicDesignerAgent
from agents.market_research_agent import MarketResearchAgent
from agents.packaging_agent import PackagingAgent
from tools.catalog_tool import ProductCatalogTool
from tools.registry import ToolRegistry
from tools.search_tool import TavilySearchTool

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Container for the complete pipeline output.

    Attributes:
        trend_summary: Final text from MarketResearchAgent.
        image_path: Path to the generated campaign image.
        image_prompt: LLM-generated prompt that was sent to the image API.
        image_caption: Short caption for the image.
        quote: Campaign tagline from CopywriterAgent.
        quote_justification: Why the quote fits the image + trend.
        report_path: Path to the saved Markdown executive report.
    """

    trend_summary: str = ""
    image_path: str = ""
    image_prompt: str = ""
    image_caption: str = ""
    quote: str = ""
    quote_justification: str = ""
    report_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class CampaignPipeline:
    """
    Orchestrates the four-agent sunglasses campaign workflow.

    Args:
        market_research_agent: Configured ``MarketResearchAgent`` instance.
        graphic_designer_agent: Configured ``GraphicDesignerAgent`` instance.
        copywriter_agent: Configured ``CopywriterAgent`` instance.
        packaging_agent: Configured ``PackagingAgent`` instance.
    """

    def __init__(
        self,
        market_research_agent: MarketResearchAgent,
        graphic_designer_agent: GraphicDesignerAgent,
        copywriter_agent: CopywriterAgent,
        packaging_agent: PackagingAgent,
    ) -> None:
        self._market_research = market_research_agent
        self._graphic_designer = graphic_designer_agent
        self._copywriter = copywriter_agent
        self._packaging = packaging_agent

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls, output_dir: Path | None = None) -> "CampaignPipeline":
        """
        Build a fully-configured pipeline from environment variables.

        Uses settings from ``config.py`` (which reads from ``.env``).

        Args:
            output_dir: Override the output directory. Defaults to
                        ``config.OUTPUT_DIR``.

        Returns:
            A ready-to-run ``CampaignPipeline``.
        """
        import openai  # lazy import so the module loads without openai installed in tests

        out = Path(output_dir) if output_dir else config.OUTPUT_DIR

        llm_client = aisuite.Client()

        image_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

        registry = (
            ToolRegistry()
            .register(
                TavilySearchTool(
                    api_key=config.TAVILY_API_KEY,
                    base_url=config.TAVILY_BASE_URL,
                    max_retries=config.TAVILY_MAX_RETRIES,
                    retry_delay=config.TAVILY_RETRY_DELAY,
                )
            )
            .register(ProductCatalogTool())
        )

        return cls(
            market_research_agent=MarketResearchAgent(
                client=llm_client,
                model=config.AGENT_MODEL,
                registry=registry,
                max_iterations=config.MAX_AGENT_ITERATIONS,
            ),
            graphic_designer_agent=GraphicDesignerAgent(
                llm_client=llm_client,
                image_client=image_client,
                model=config.AGENT_MODEL,
                image_model=config.IMAGE_MODEL,
                image_size=config.IMAGE_SIZE,
                image_quality=config.IMAGE_QUALITY,
                output_dir=out,
            ),
            copywriter_agent=CopywriterAgent(
                client=llm_client,
                model=config.AGENT_MODEL,
            ),
            packaging_agent=PackagingAgent(
                client=llm_client,
                model=config.AGENT_MODEL,
                output_dir=out,
            ),
        )

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(
        self,
        output_filename: str | None = None,
        caption_style: str = "short and punchy",
    ) -> PipelineResult:
        """
        Execute the full four-agent pipeline.

        Args:
            output_filename: Override the Markdown report filename.
                             Defaults to ``campaign_summary_<timestamp>.md``.
            caption_style: Style hint passed to the Graphic Designer Agent.

        Returns:
            ``PipelineResult`` with all intermediate and final outputs.
        """
        started_at = datetime.now()
        result = PipelineResult()

        # ── Step 1: Market Research ───────────────────────────────────────────
        logger.info("=== Step 1/4: Market Research ===")
        research_result = self._market_research.run()
        result.trend_summary = research_result.content
        logger.info("Market research complete (%d tool calls)", research_result.tool_calls_made)

        # ── Step 2: Graphic Design ────────────────────────────────────────────
        logger.info("=== Step 2/4: Graphic Design ===")
        design_result = self._graphic_designer.run(
            trend_summary=result.trend_summary,
            caption_style=caption_style,
        )
        result.image_path = design_result["image_path"]
        result.image_prompt = design_result["prompt"]
        result.image_caption = design_result["caption"]
        logger.info("Image generated: %s", result.image_path)

        # ── Step 3: Copywriting ───────────────────────────────────────────────
        logger.info("=== Step 3/4: Copywriting ===")
        copy_result = self._copywriter.run(
            image_path=result.image_path,
            trend_summary=result.trend_summary,
        )
        result.quote = copy_result.get("quote", "")
        result.quote_justification = copy_result.get("justification", "")
        logger.info("Quote: %s", result.quote)

        # ── Step 4: Packaging ─────────────────────────────────────────────────
        logger.info("=== Step 4/4: Packaging ===")
        report_path = self._packaging.run(
            trend_summary=result.trend_summary,
            image_path=result.image_path,
            quote=result.quote,
            justification=result.quote_justification,
            output_filename=output_filename,
        )
        result.report_path = report_path
        logger.info("Report saved: %s", result.report_path)

        duration = (datetime.now() - started_at).total_seconds()
        result.metadata = {"duration_seconds": duration, "started_at": started_at.isoformat()}
        logger.info("Pipeline complete in %.1fs", duration)

        return result
