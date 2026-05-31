"""
AWS Bedrock client for the reflection pattern demo.

Provides a thin wrapper around the Bedrock Converse API so that both
text-only and multimodal (text + image) calls share a consistent interface.

Configuration (via environment variables):
    AWS_PROFILE  — boto3 credential profile name (optional)
    AWS_REGION   — override default region (defaults to us-east-1)
"""

import mimetypes
import os
from pathlib import Path

import boto3

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

_region = os.environ.get("AWS_REGION", "us-east-1")
_profile = os.environ.get("AWS_PROFILE")

_session = boto3.Session(profile_name=_profile, region_name=_region) if _profile else boto3.Session(region_name=_region)
_bedrock = _session.client("bedrock-runtime")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def generate_text(model_id: str, prompt: str, max_tokens: int = 2000, temperature: float = 0.0) -> str:
    """Call a Bedrock model with a plain-text prompt and return the response text."""
    response = _bedrock.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    )
    return _extract_text(response)


def generate_with_image(
    model_id: str,
    prompt: str,
    image_path: str,
    max_tokens: int = 2000,
    temperature: float = 0.0,
) -> str:
    """Call a multimodal Bedrock model with a prompt and an image file.

    The image format is inferred from the file extension.
    """
    image_bytes, image_format = _read_image(image_path)

    response = _bedrock.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                    {
                        "image": {
                            "format": image_format,
                            "source": {"bytes": image_bytes},
                        }
                    },
                ],
            }
        ],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    )
    return _extract_text(response)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_text(converse_response: dict) -> str:
    """Pull the first text block out of a Bedrock Converse response."""
    output_message = converse_response["output"]["message"]
    for block in output_message["content"]:
        if "text" in block:
            return block["text"].strip()
    return ""


def _read_image(image_path: str) -> tuple[bytes, str]:
    """Return raw bytes and Bedrock-compatible format string for an image file."""
    path = Path(image_path)
    mime, _ = mimetypes.guess_type(path)
    # Bedrock Converse accepts: jpeg, png, gif, webp
    format_map = {
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
    }
    image_format = format_map.get(mime or "", "png")
    with open(path, "rb") as f:
        return f.read(), image_format
