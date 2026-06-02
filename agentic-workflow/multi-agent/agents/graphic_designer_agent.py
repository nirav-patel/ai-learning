"""
Graphic Designer Agent.

Takes a trend summary from the Market Research Agent and:
  1. Uses the LLM to write a vivid image-generation prompt + marketing caption.
  2. Calls the OpenAI Images API (DALL-E) to generate the campaign visual.
  3. Saves the image to disk and returns metadata.

This agent does **not** use the tool-based agentic loop because image generation
is a deterministic two-step process (text → prompt → image), not a dynamic
reasoning loop.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GraphicDesignerAgent:
    """
    Generates a campaign image from trend insights.

    Args:
        llm_client: aisuite-compatible client (used for prompt + caption generation).
        image_client: ``openai.OpenAI`` client (used for image generation).
        model: LLM model string for text generation.
        image_model: OpenAI image model (e.g. ``"dall-e-3"``).
        image_size: Dimensions string (e.g. ``"1024x1024"``).
        image_quality: Quality level — ``"standard"`` or ``"hd"`` (dall-e-3 only).
        output_dir: Directory where the generated PNG is saved.
    """

    name = "GraphicDesignerAgent"

    def __init__(
        self,
        llm_client: Any,
        image_client: Any,
        model: str,
        image_model: str = "dall-e-3",
        image_size: str = "1024x1024",
        image_quality: str = "standard",
        output_dir: Path = Path("./output"),
    ) -> None:
        self._llm_client = llm_client
        self._image_client = image_client
        self._model = model
        self._image_model = image_model
        self._image_size = image_size
        self._image_quality = image_quality
        self._output_dir = Path(output_dir)

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        trend_summary: str,
        caption_style: str = "short and punchy",
        output_filename: str = "generated_image.png",
    ) -> dict[str, str]:
        """
        Generate an image from *trend_summary* and save it to disk.

        Args:
            trend_summary: Text output from the Market Research Agent.
            caption_style: Style hint for the caption (e.g. ``"short and punchy"``).
            output_filename: Filename for the saved PNG (inside ``output_dir``).

        Returns:
            Dict with keys ``image_path``, ``prompt``, ``caption``.
        """
        logger.info("[%s] Generating image prompt and caption", self.name)

        # Step 1: Ask the LLM to write an image prompt + caption
        prompt, caption = self._generate_prompt_and_caption(trend_summary, caption_style)
        logger.debug("[%s] Image prompt: %s", self.name, prompt)

        # Step 2: Generate the image
        logger.info("[%s] Calling image generation API", self.name)
        image_path = self._generate_and_save_image(prompt, output_filename)

        return {
            "image_path": str(image_path),
            "prompt": prompt,
            "caption": caption,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_prompt_and_caption(
        self, trend_summary: str, caption_style: str
    ) -> tuple[str, str]:
        """Call the LLM to produce an image generation prompt + marketing caption."""
        system = (
            "You are a visual marketing assistant. "
            "Based on trend insights you write creative image-generation prompts "
            "and short marketing captions."
        )
        user = f"""
Trend insights:
{trend_summary}

Please output a JSON object with exactly two keys:
- "prompt": A vivid, descriptive prompt (2–4 sentences) to guide an AI image generation model.
            Focus on photographic style, lighting, sunglasses styles, and summer mood.
- "caption": A marketing caption in style: {caption_style} (max 15 words).

Respond with raw JSON only — no markdown fences or extra text.
""".strip()

        response = self._llm_client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content.strip()

        try:
            # Be tolerant of markdown fences the model may add
            match = re.search(r"\{.*\}", content, re.DOTALL)
            parsed: dict[str, str] = json.loads(match.group(0)) if match else {}
            return parsed.get("prompt", content), parsed.get("caption", "")
        except (json.JSONDecodeError, AttributeError):
            logger.warning("[%s] Could not parse JSON from LLM; using raw output as prompt", self.name)
            return content, ""

    def _generate_and_save_image(self, prompt: str, filename: str) -> Path:
        """Call the OpenAI Images API and save the result as a PNG."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir / filename

        response = self._image_client.images.generate(
            model=self._image_model,
            prompt=prompt,
            size=self._image_size,
            quality=self._image_quality,
            n=1,
            response_format="b64_json",
        )

        b64_data = response.data[0].b64_json
        img_bytes = base64.b64decode(b64_data)

        # Use Pillow to verify / re-save as PNG
        try:
            from PIL import Image  # lazy import — optional at import time

            img = Image.open(BytesIO(img_bytes))
            img.save(output_path)
        except ImportError:
            output_path.write_bytes(img_bytes)

        logger.info("[%s] Image saved to %s", self.name, output_path)
        return output_path
