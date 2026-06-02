"""
AWS Bedrock client for the reflection pattern demo and research pipeline.

Provides a thin wrapper around the Bedrock Converse API so that both
text-only and multimodal (text + image) calls share a consistent interface.
Also exposes a tool-calling loop suitable for agentic research workflows.

Configuration (via environment variables):
    AWS_PROFILE  — boto3 credential profile name (optional)
    AWS_REGION   — override default region (defaults to us-east-1)
"""

import json
import mimetypes
import os
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

_region = os.environ.get("AWS_REGION", "us-east-1")
_profile = os.environ.get("AWS_PROFILE")

# Increase read_timeout to 300s — large generations can take >60s (boto3 default).
_config = Config(read_timeout=300, connect_timeout=10, retries={"max_attempts": 1})

_session = boto3.Session(profile_name=_profile, region_name=_region) if _profile else boto3.Session(region_name=_region)
_bedrock = _session.client("bedrock-runtime", config=_config)


def _handle_client_error(exc: ClientError, context: str = "") -> None:
    """Re-raise ClientError with a human-friendly message for common cases."""
    code = exc.response["Error"]["Code"]
    if code == "ExpiredTokenException":
        raise RuntimeError(
            "AWS credentials have expired. Please refresh your credentials "
            "(e.g. `aws sso login` or renew your session) and re-run."
        ) from exc
    if code == "ThrottlingException":
        raise RuntimeError(
            f"Bedrock throttling{' (' + context + ')' if context else ''}. "
            "Wait a moment and retry, or reduce request rate."
        ) from exc
    raise


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


def generate_text_with_system(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2000,
    temperature: float = 0.0,
) -> str:
    """
    Call a Bedrock model with separate system and user prompts.

    Uses converse_stream so that large responses don't time out —
    the connection stays alive as tokens arrive instead of waiting
    for the entire response before returning.
    """
    try:
        response = _bedrock.converse_stream(
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": user_prompt}],
                }
            ],
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        )
    except ClientError as exc:
        _handle_client_error(exc, context="generate_text_with_system")

    # Collect streamed text chunks
    chunks = []
    for event in response["stream"]:
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                chunks.append(delta["text"])
    return "".join(chunks).strip()


def run_tool_loop(
    model_id: str,
    system_prompt: str,
    initial_prompt: str,
    tools: list[dict],
    tool_mapping: dict,
    max_turns: int = 10,
    max_tokens: int = 4000,
    temperature: float = 0.7,
) -> tuple[str, list[dict]]:
    """
    Run an agentic tool-calling loop against a Bedrock model.

    The loop continues until the model returns a plain-text response (no
    toolUse blocks) or max_turns is exhausted.

    Parameters
    ----------
    model_id      : Bedrock model ID to call.
    system_prompt : System instructions passed once for the whole conversation.
    initial_prompt: First user message that starts the conversation.
    tools         : List of Bedrock toolSpec dicts.
    tool_mapping  : Dict mapping tool name → Python callable.
    max_turns     : Safety ceiling for the number of Converse calls.
    max_tokens    : Max tokens per Converse call.
    temperature   : Sampling temperature.

    Returns
    -------
    final_text : The last plain-text assistant response.
    messages   : Full conversation history (user + assistant turns).
    """
    messages: list[dict] = [
        {"role": "user", "content": [{"text": initial_prompt}]}
    ]

    tool_config = {"tools": tools, "toolChoice": {"auto": {}}}
    final_text = ""

    for turn in range(max_turns):
        try:
            response = _bedrock.converse(
                modelId=model_id,
                system=[{"text": system_prompt}],
                messages=messages,
                toolConfig=tool_config,
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                },
            )
        except ClientError as exc:
            _handle_client_error(exc, context=f"run_tool_loop turn {turn}")

        stop_reason = response["stopReason"]
        assistant_content = response["output"]["message"]["content"]

        # Record assistant turn
        messages.append({"role": "assistant", "content": assistant_content})

        if stop_reason == "end_turn":
            # Model finished — collect the final text
            for block in assistant_content:
                if "text" in block:
                    final_text = block["text"].strip()
                    break
            print("✅ Final answer received.")
            break

        if stop_reason == "tool_use":
            # Execute every tool the model requested and collect results
            tool_results = []
            for block in assistant_content:
                if "toolUse" not in block:
                    continue
                tool_use = block["toolUse"]
                tool_name = tool_use["name"]
                tool_input = tool_use["input"]
                tool_use_id = tool_use["toolUseId"]

                print(f"🛠️  {tool_name}({tool_input})")

                try:
                    func = tool_mapping[tool_name]
                    result = func(**tool_input)
                except Exception as e:
                    result = {"error": str(e)}

                tool_results.append(
                    {
                        "toolUseId": tool_use_id,
                        "content": [{"text": json.dumps(result)}],
                    }
                )

            # Feed all results back as a single user turn
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"toolResult": tr} for tr in tool_results
                    ],
                }
            )
        else:
            # Unexpected stop reason — treat remaining content as final
            for block in assistant_content:
                if "text" in block:
                    final_text = block["text"].strip()
                    break
            break

    return final_text, messages


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
