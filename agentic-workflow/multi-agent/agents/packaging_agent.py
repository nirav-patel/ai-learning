"""
Packaging Agent.

Assembles all campaign assets (trend summary, image, quote) into a polished
Markdown executive report and saves it to disk.

A short LLM call is used to rewrite the trend summary in executive prose
before embedding it in the report.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PackagingAgent:
    """
    Produces a Markdown campaign report from the upstream agents' outputs.

    Args:
        client: aisuite-compatible LLM client.
        model: Model identifier.
        output_dir: Directory where the report file is saved.
    """

    name = "PackagingAgent"

    def __init__(
        self,
        client: Any,
        model: str,
        output_dir: Path = Path("./output"),
    ) -> None:
        self._client = client
        self._model = model
        self._output_dir = Path(output_dir)

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        trend_summary: str,
        image_path: str | Path,
        quote: str,
        justification: str,
        output_filename: str | None = None,
    ) -> str:
        """
        Build and save the executive Markdown report.

        Args:
            trend_summary: Raw text from the Market Research Agent.
            image_path: Path to the generated campaign image.
            quote: Campaign tagline from the Copywriter Agent.
            justification: Quote justification from the Copywriter Agent.
            output_filename: Optional filename override; defaults to a
                             timestamped name like ``campaign_summary_YYYY-MM-DD.md``.

        Returns:
            Absolute path of the saved Markdown file as a string.
        """
        logger.info("[%s] Assembling executive report", self.name)

        beautified = self._beautify_summary(trend_summary)

        filename = output_filename or f"campaign_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir / filename

        markdown = self._build_markdown(
            beautified_summary=beautified,
            image_path=str(image_path),
            quote=quote,
            justification=justification,
        )

        output_path.write_text(markdown, encoding="utf-8")
        logger.info("[%s] Report saved to %s", self.name, output_path)
        return str(output_path)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _beautify_summary(self, raw_summary: str) -> str:
        """Rewrite the raw trend summary in executive-friendly prose."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a marketing communication expert. "
                        "Rewrite trend summaries to be clear, professional, and engaging "
                        "for a CEO-level audience. Preserve all key information."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Please rewrite the following trend summary for executives:\n\n"
                        f'"""\n{raw_summary.strip()}\n"""'
                    ),
                },
            ],
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _build_markdown(
        beautified_summary: str,
        image_path: str,
        quote: str,
        justification: str,
    ) -> str:
        """Assemble the final Markdown document."""
        today = datetime.now().strftime("%Y-%m-%d")
        return f"""# 🕶️ Summer Sunglasses Campaign — Executive Summary

## 📊 Trend Insights

{beautified_summary}

## 🎯 Campaign Visual

![Campaign Image]({image_path})

## ✍️ Campaign Quote

> {quote.strip()}

## ✅ Why This Works

{justification.strip()}

---

*Report generated on {today}*
"""
