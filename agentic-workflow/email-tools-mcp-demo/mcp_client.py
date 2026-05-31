"""
MCP client that connects to the email MCP server via stdio transport.

EmailMCPClient is an async context manager that:
  - Spawns the MCP server subprocess (email_mcp_server.py) over stdio
  - Fetches the tool list and converts it to Bedrock-compatible tool specs
  - Executes tool calls by forwarding them through the MCP protocol

Usage:
    async with EmailMCPClient() as client:
        specs = await client.list_tools_as_bedrock_specs()
        result = await client.call_tool("search_emails", {"query": "lunch"})
"""

import json
import os
import sys
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# Path to the MCP server entry point
_MCP_SERVER_MODULE = "mcp_server.email_mcp_server"

# Map MCP JSON schema types to Bedrock-compatible JSON schema types
_TYPE_MAP: dict[str, str] = {
    "string":  "string",
    "integer": "integer",
    "number":  "number",
    "boolean": "boolean",
    "array":   "array",
    "object":  "object",
}


class EmailMCPClient:
    """
    Async context manager wrapping an MCP ClientSession connected to the
    email MCP server over stdio transport.

    Provides two high-level methods used by the Bedrock agentic loop:
      - list_tools_as_bedrock_specs(): discover tools at runtime
      - call_tool(name, args): execute a tool via the MCP protocol
    """

    def __init__(self, email_server_url: str = "http://127.0.0.1:5001") -> None:
        self._email_server_url = email_server_url
        self._session: ClientSession | None = None
        self._exit_stack = None

    async def __aenter__(self) -> "EmailMCPClient":
        from contextlib import AsyncExitStack

        env = {**os.environ, "EMAIL_SERVER_URL": self._email_server_url}
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", _MCP_SERVER_MODULE],
            env=env,
        )

        self._exit_stack = AsyncExitStack()
        read, write = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._exit_stack:
            await self._exit_stack.aclose()

    async def list_tools_as_bedrock_specs(self) -> list[dict]:
        """
        Fetch the tool list from the MCP server and convert each tool's
        JSON schema into a Bedrock-compatible toolSpec dict.
        """
        result = await self._session.list_tools()
        return [_to_bedrock_spec(tool) for tool in result.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a named tool on the MCP server and return its output as a string.

        FastMCP may return list results as multiple content blocks (one JSON
        object per block). This method reassembles them into a single JSON
        array string so the LLM receives clean, parseable output.
        """
        result = await self._session.call_tool(name, arguments)
        parts = [
            block.text if hasattr(block, "text") else str(block)
            for block in result.content
        ]

        if len(parts) <= 1:
            return parts[0] if parts else ""

        # Multiple blocks — try to reassemble as a JSON array
        items = []
        for part in parts:
            try:
                items.append(json.loads(part))
            except json.JSONDecodeError:
                # Not JSON — fall back to plain joined text
                return "\n".join(parts)
        return json.dumps(items, indent=2)


# ── Conversion helper ─────────────────────────────────────────────────────────

def _to_bedrock_spec(tool) -> dict:
    """Convert an MCP Tool object into a Bedrock toolSpec dict."""
    schema = tool.inputSchema or {}
    raw_props = schema.get("properties", {})
    required   = schema.get("required", [])

    properties = {
        name: {
            "type":        _TYPE_MAP.get(prop.get("type", "string"), "string"),
            "description": prop.get("description", name),
        }
        for name, prop in raw_props.items()
    }

    return {
        "toolSpec": {
            "name":        tool.name,
            "description": tool.description or tool.name,
            "inputSchema": {
                "json": {
                    "type":       "object",
                    "properties": properties,
                    "required":   required,
                }
            },
        }
    }
