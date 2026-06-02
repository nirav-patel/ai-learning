"""
Copywriter Agent.

Receives the campaign image (as a local file path) and the trend summary,
then writes a short campaign quote + justification using a multimodal LLM call.

The agent uses the Anthropic SDK with AWS Bedrock (``anthropic.AnthropicBedrock``)
because Claude's vision API requires the Anthropic-native image content format
(``type: image``, ``source.type: base64``) rather than the OpenAI-style
``image_url`` format.  All other agents use the Anthropic Bedrock SDK for text-only calls.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CopywriterAgent:
    """
    Produces a campaign quote from an image and a trend summary.

    Args:
        client: ``anthropic.AnthropicBedrock`` (or any Anthropic-compatible client)
                that exposes a ``messages.create`` method.
        model: Bedrock model ID (bare, without ``aws:`` prefix), e.g.
               ``"us.anthropic.claude-sonnet-4-6"``.
    """

    name = "CopywriterAgent"

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, image_path: str | Path, trend_summary: str) -> dict[str, str]:
        """
        Generate a campaign quote for the given image + trend context.

        Args:
            image_path: Local path to the generated campaign image (PNG).
            trend_summary: Text output from the Market Research Agent.

        Returns:
            Dict with keys ``quote``, ``justification``, ``image_path``.
        """
        logger.info("[%s] Building campaign quote", self.name)

        b64_image = self._encode_image(image_path)
        messages, system = self._build_messages(b64_image, trend_summary)

        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=system,
            messages=messages,
        )
        content = response.content[0].text.strip()

        parsed = self._parse_response(content)
        parsed["image_path"] = str(image_path)

        logger.info("[%s] Quote generated: %s", self.name, parsed.get("quote", ""))
        return parsed

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _encode_image(image_path: str | Path) -> str:
        """Read the image file and return it as a base64 string."""
        with open(image_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")

    def _build_messages(
        self, b64_image: str, trend_summary: str
    ) -> tuple[list[dict[str, Any]], str]:
        """
        Build the message list and system prompt for the Anthropic messages API.

        Returns a ``(messages, system)`` tuple so callers can pass them separately
        to ``client.messages.create(system=..., messages=...)``.
        """
        system = (
            "You are a creative copywriter that crafts elegant, memorable campaign "
            "quotes based on a visual and marketing trend context."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"""
Here is a campaign image and a trend analysis:

Trend summary:
\"\"\"{trend_summary}\"\"\"

Please return a JSON object with exactly two keys:
- "quote": A short, elegant campaign phrase (max 12 words).
- "justification": One sentence explaining why this quote matches the image and trend.

Respond with raw JSON only — no markdown fences or extra text.
""".strip(),
                    },
                ],
            }
        ]
        return messages, system

    def _parse_response(self, content: str) -> dict[str, str]:
        """Extract the JSON dict from the LLM response."""
        try:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            return json.loads(match.group(0)) if match else {"quote": content, "justification": ""}
        except (json.JSONDecodeError, AttributeError):
            logger.warning("[%s] Could not parse JSON response; storing raw content", self.name)
            return {"quote": content, "justification": ""}
