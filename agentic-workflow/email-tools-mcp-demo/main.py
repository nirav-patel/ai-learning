"""
Email Tools MCP Demo — entry point.

Starts the FastAPI email simulation server, then connects to the email MCP
server to run four demo scenarios. Every tool call travels through the MCP
protocol rather than being invoked as a direct Python function.

Scenarios:
  1. Multi-step workflow — unread → mark read → send reply (via MCP)
  2. Tool gap           — delete request with full tools (agent finds & deletes)
  3. Delete by sender   — search sender + delete (after inbox reset)
  4. Delete by subject  — search keyword + delete (after inbox reset)
"""

import asyncio
import os
import subprocess
import sys
import time

import requests

from agent import run_with_mcp_tools
from display import print_agent_result
from mcp_client import EmailMCPClient

# ── Configuration ─────────────────────────────────────────────────────────────

EMAIL_SERVER_PORT = 5001
EMAIL_SERVER_URL  = f"http://127.0.0.1:{EMAIL_SERVER_PORT}"
HEALTH_URL        = f"{EMAIL_SERVER_URL}/health"

SMART_MODEL = "us.anthropic.claude-sonnet-4-6"

SYSTEM_CONTEXT = (
    "You are an AI assistant specialized in managing emails. "
    "Use the provided tools to interact with the email system. "
    "Never ask the user for confirmation before performing an action. "
    "My email address is 'you@email.com'."
)


def build_prompt(request: str) -> str:
    return f"{SYSTEM_CONTEXT}\n\n{request.strip()}"


# ── Email server lifecycle ────────────────────────────────────────────────────

def start_email_server() -> subprocess.Popen:
    """Launch the FastAPI email server as a background subprocess."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        "email_server.email_service:app",
        "--port", str(EMAIL_SERVER_PORT),
        "--log-level", "warning",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.path.dirname(__file__),
    )


def wait_for_server(timeout: int = 15) -> None:
    """Poll /health until the server is ready or timeout is reached."""
    for _ in range(timeout * 2):
        try:
            if requests.get(HEALTH_URL, timeout=1).status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Email server did not start within {timeout}s")


# ── Demo scenarios ────────────────────────────────────────────────────────────

async def demo_1_read_and_reply(client: EmailMCPClient) -> None:
    """
    Multi-step workflow via MCP:
    Find unread email from boss → mark as read → send polite follow-up.
    """
    prompt = build_prompt(
        "Check for unread emails from boss@email.com, mark them as read, "
        "and send a polite follow-up reply."
    )
    result = await run_with_mcp_tools(prompt, client, SMART_MODEL, max_turns=10)
    print_agent_result(result, title="Demo 1 — Read & Reply (MCP)")


async def demo_2_delete_by_sender(client: EmailMCPClient) -> None:
    """
    Search by sender and delete via MCP.
    The agent searches for alice's email, finds the ID, then deletes it.
    """
    requests.get(f"{EMAIL_SERVER_URL}/reset_database")

    prompt = build_prompt("Delete alice@work.com email.")
    result = await run_with_mcp_tools(prompt, client, SMART_MODEL, max_turns=5)
    print_agent_result(result, title="Demo 2 — Delete by Sender (MCP)")


async def demo_3_delete_by_subject(client: EmailMCPClient) -> None:
    """
    Search by subject keyword and delete via MCP.
    The agent searches for 'happy hour', identifies the email, and deletes it.
    """
    requests.get(f"{EMAIL_SERVER_URL}/reset_database")

    prompt = build_prompt("Delete the happy hour email.")
    result = await run_with_mcp_tools(prompt, client, SMART_MODEL, max_turns=5)
    print_agent_result(result, title="Demo 3 — Delete by Subject (MCP)")


async def demo_4_summarise_unread(client: EmailMCPClient) -> None:
    """
    List all unread emails and produce a concise summary via MCP.
    """
    requests.get(f"{EMAIL_SERVER_URL}/reset_database")

    prompt = build_prompt(
        "List all unread emails and give me a brief summary of each one, "
        "including sender and key point."
    )
    result = await run_with_mcp_tools(prompt, client, SMART_MODEL, max_turns=5)
    print_agent_result(result, title="Demo 4 — Summarise Unread (MCP)")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    async with EmailMCPClient(email_server_url=EMAIL_SERVER_URL) as client:
        # Show discovered tools on startup
        specs = await client.list_tools_as_bedrock_specs()
        tool_names = [s["toolSpec"]["name"] for s in specs]
        print(f"✅ MCP server connected — {len(tool_names)} tools discovered:")
        for name in tool_names:
            print(f"   • {name}")
        print()

        await demo_1_read_and_reply(client)
        await demo_2_delete_by_sender(client)
        await demo_3_delete_by_subject(client)
        await demo_4_summarise_unread(client)


if __name__ == "__main__":
    print("Starting email simulation server…")
    server = start_email_server()
    try:
        wait_for_server()
        print(f"✅ Email server ready on port {EMAIL_SERVER_PORT}\n")
        asyncio.run(main())
    finally:
        server.terminate()
        server.wait()
        print("Email server stopped.")
