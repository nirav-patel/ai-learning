"""
Bedrock agentic loop that routes tool calls through the MCP protocol.

Unlike the direct-call agent in email-tools-demo (which invokes Python
functions in-process), this agent:
  - Fetches tool specs dynamically from the MCP server at runtime
  - Executes every tool call via EmailMCPClient.call_tool()

Usage:
    async with EmailMCPClient() as client:
        result = await run_with_mcp_tools(
            prompt="Check unread emails from boss...",
            client=client,
            model="us.anthropic.claude-sonnet-4-6",
        )
"""

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import boto3

if TYPE_CHECKING:
    from mcp_client import EmailMCPClient

_region  = os.environ.get("AWS_REGION", "us-east-1")
_profile = os.environ.get("AWS_PROFILE")
_session = (
    boto3.Session(profile_name=_profile, region_name=_region)
    if _profile
    else boto3.Session(region_name=_region)
)
_bedrock = _session.client("bedrock-runtime")


# ── Result containers ─────────────────────────────────────────────────────────

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


# ── Agentic loop ──────────────────────────────────────────────────────────────

async def run_with_mcp_tools(
    prompt:    str,
    client:    "EmailMCPClient",
    model:     str,
    max_turns: int = 10,
) -> AgentResult:
    """
    Run an agentic loop using MCP-sourced tools and AWS Bedrock for reasoning.

    Tool specs are fetched dynamically from the MCP server. Each tool call
    requested by the model is forwarded to the MCP server for execution.

    Args:
        prompt:    Natural-language instruction for the agent.
        client:    Connected EmailMCPClient instance.
        model:     Bedrock model ID.
        max_turns: Safety cap on the number of LLM calls.

    Returns:
        AgentResult with the final answer and a trace of every tool call.
    """
    tool_specs = await client.list_tools_as_bedrock_specs()
    messages: list[dict] = [{"role": "user", "content": [{"text": prompt}]}]
    result = AgentResult(prompt=prompt)

    for _ in range(max_turns):
        response = _bedrock.converse(
            modelId=model,
            messages=messages,
            toolConfig={"tools": tool_specs},
        )

        stop_reason = response["stopReason"]
        output_msg  = response["output"]["message"]
        messages.append(output_msg)

        if stop_reason == "end_turn":
            parts = [
                block["text"]
                for block in output_msg["content"]
                if "text" in block
            ]
            result.answer = "\n".join(parts)
            break

        if stop_reason == "tool_use":
            tool_results = []

            for block in output_msg["content"]:
                if "toolUse" not in block:
                    continue

                tool_use    = block["toolUse"]
                tool_name   = tool_use["name"]
                tool_args   = tool_use.get("input", {})
                tool_use_id = tool_use["toolUseId"]

                # Execute via MCP protocol (not direct Python call)
                tool_output = await client.call_tool(tool_name, tool_args)

                result.tool_calls.append(
                    ToolCall(tool_name=tool_name, args=tool_args, result=tool_output)
                )

                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content":   [{"text": tool_output}],
                    }
                })

            messages.append({"role": "user", "content": tool_results})

    return result
