"""
Agentic tool-calling loop for AWS Bedrock.

Converts plain Python functions into Bedrock tool specs automatically
(using their docstrings and type annotations), then runs a multi-turn
agentic loop until the model produces a final answer.

Usage:
    from agent import run_with_tools
    result = run_with_tools("What time is it?", tools=[get_current_time], model="...")
"""

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import boto3
import os

_region  = os.environ.get("AWS_REGION", "us-east-1")
_profile = os.environ.get("AWS_PROFILE")
_session = (
    boto3.Session(profile_name=_profile, region_name=_region)
    if _profile
    else boto3.Session(region_name=_region)
)
_bedrock = _session.client("bedrock-runtime")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    tool_name: str
    args:      dict
    result:    str


@dataclass
class AgentResult:
    prompt:     str
    answer:     str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Function → Bedrock tool spec
# ---------------------------------------------------------------------------

_PY_TO_JSON_TYPE: dict[Any, str] = {
    str:   "string",
    int:   "integer",
    float: "number",
    bool:  "boolean",
}


def _func_to_tool_spec(func: Callable) -> dict:
    """
    Build a Bedrock-compatible tool spec from a Python function.

    Uses the function's name, docstring, and type-annotated parameters.
    The first line of the docstring becomes the tool description.
    Annotated parameters are listed in the JSON schema; all are optional
    unless they have no default value.
    """
    doc = inspect.cleandoc(func.__doc__ or "")
    # Collect lines until a blank line or an Args/Returns/Raises section header
    first_para_lines = []
    for line in doc.splitlines():
        stripped = line.strip()
        if not stripped or stripped.rstrip(":") in ("Args", "Returns", "Raises", "Note", "Notes"):
            break
        first_para_lines.append(stripped)
    description = " ".join(first_para_lines) if first_para_lines else func.__name__

    sig = inspect.signature(func)
    properties: dict[str, dict] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        json_type = _PY_TO_JSON_TYPE.get(param.annotation, "string")
        properties[name] = {"type": json_type, "description": name}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "toolSpec": {
            "name":        func.__name__,
            "description": description,
            "inputSchema": {
                "json": {
                    "type":       "object",
                    "properties": properties,
                    "required":   required,
                }
            },
        }
    }


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def run_with_tools(
    prompt:    str,
    tools:     list[Callable],
    model:     str,
    max_turns: int = 10,
) -> AgentResult:
    """
    Run an agentic loop: prompt the model, execute any tool calls it requests,
    feed results back, repeat until the model gives a final answer.

    Args:
        prompt:    User question or instruction.
        tools:     List of Python callables the model may invoke.
        model:     Bedrock model ID.
        max_turns: Safety cap on the number of LLM calls.

    Returns:
        AgentResult with the final answer and a trace of every tool call.
    """
    tool_specs = [_func_to_tool_spec(f) for f in tools]
    tool_map   = {f.__name__: f for f in tools}

    messages: list[dict] = [{"role": "user", "content": [{"text": prompt}]}]
    result = AgentResult(prompt=prompt)

    for _ in range(max_turns):
        response = _bedrock.converse(
            modelId=model,
            messages=messages,
            toolConfig={"tools": tool_specs},
        )

        stop_reason  = response["stopReason"]
        output_msg   = response["output"]["message"]
        messages.append(output_msg)

        # Final answer — collect all text blocks
        if stop_reason == "end_turn":
            parts = [
                block["text"]
                for block in output_msg["content"]
                if "text" in block
            ]
            result.answer = "\n".join(parts)
            break

        # Tool calls — execute each one and collect results
        if stop_reason == "tool_use":
            tool_results = []

            for block in output_msg["content"]:
                if "toolUse" not in block:
                    continue

                tool_use    = block["toolUse"]
                tool_name   = tool_use["name"]
                tool_args   = tool_use.get("input", {})
                tool_use_id = tool_use["toolUseId"]

                func         = tool_map.get(tool_name)
                tool_output  = func(**tool_args) if func else f"Unknown tool: {tool_name}"

                result.tool_calls.append(
                    ToolCall(tool_name=tool_name, args=tool_args, result=str(tool_output))
                )

                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content":   [{"text": str(tool_output)}],
                    }
                })

            messages.append({"role": "user", "content": tool_results})

    return result
