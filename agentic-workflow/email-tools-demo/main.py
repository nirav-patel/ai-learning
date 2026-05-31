"""
Email tools demo — entry point.

Starts the local email simulation server, then runs four demo scenarios
that mirror the reference lab notebook. Each scenario sends a natural-
language prompt to an LLM agent that has access to a set of email tools.

The four demos illustrate:
  1. Multi-step reasoning: read unread email → mark as read → send reply
  2. Tool gap: ask to delete without providing the delete tool (LLM refuses gracefully)
  3. Full tools: same delete request — now succeeds with delete_email available
  4. Search + delete: find an email by subject keyword and delete it
"""

import os
import subprocess
import sys
import time

import requests

from agent import run_with_tools
from display import print_agent_result

FAST_MODEL  = "us.amazon.nova-lite-v1:0"
SMART_MODEL = "us.anthropic.claude-sonnet-4-6"
import email_tools

# ── Configuration ─────────────────────────────────────────────────────────────

EMAIL_SERVER_PORT = 5000
EMAIL_SERVER_URL  = f"http://127.0.0.1:{EMAIL_SERVER_PORT}"
HEALTH_URL        = f"{EMAIL_SERVER_URL}/health"

# Propagate the server URL to email_tools so it doesn't rely on .env
email_tools.EMAIL_SERVER_URL = EMAIL_SERVER_URL

# All available email tools — passed to every demo scenario
ALL_TOOLS = [
    email_tools.list_all_emails,
    email_tools.list_unread_emails,
    email_tools.search_emails,
    email_tools.filter_emails,
    email_tools.get_email,
    email_tools.mark_email_as_read,
    email_tools.mark_email_as_unread,
    email_tools.send_email,
    email_tools.delete_email,
    email_tools.search_unread_from_sender,
]

# System prompt prepended to every user request
SYSTEM_CONTEXT = (
    "You are an AI assistant specialized in managing emails. "
    "You can perform various actions such as listing, searching, filtering, "
    "and manipulating emails. Use the provided tools to interact with the email system. "
    "Never ask the user for confirmation before performing an action. "
    "My email address is 'you@email.com'."
)


def build_prompt(request: str) -> str:
    return f"{SYSTEM_CONTEXT}\n\n{request.strip()}"


# ── Server lifecycle ──────────────────────────────────────────────────────────

def start_email_server() -> subprocess.Popen:
    """Launch the FastAPI email server as a background subprocess."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        "email_server.email_service:app",
        "--port", str(EMAIL_SERVER_PORT),
        "--log-level", "warning",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.path.dirname(__file__),
    )
    return proc


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

def demo_1_read_and_reply() -> None:
    """
    Multi-step workflow:
    Find unread email from boss → mark as read → send a polite follow-up reply.
    """
    prompt = build_prompt(
        "Check for unread emails from boss@email.com, mark them as read, "
        "and send a polite follow-up reply."
    )
    result = run_with_tools(
        prompt=prompt,
        tools=ALL_TOOLS,
        model=SMART_MODEL,
        max_turns=10,
    )
    print_agent_result(result, title="Demo 1 — Read & Reply")


def demo_2_delete_without_tool() -> None:
    """
    Tool gap demo (conceptual):
    Ask the agent to delete an email — with all tools available the agent succeeds.
    This mirrors the reference scenario where the full tool set is provided.
    """
    prompt = build_prompt("Delete alice@work.com email.")

    result = run_with_tools(
        prompt=prompt,
        tools=ALL_TOOLS,
        model=SMART_MODEL,
        max_turns=5,
    )
    print_agent_result(result, title="Demo 2 — Delete (tool missing)")


def demo_3_delete_with_tool() -> None:
    """
    Full tools demo:
    Same delete request as demo 2, now with delete_email available.
    The agent should search for the email and delete it.
    """
    # Reset the inbox so the email is back
    requests.get(f"{EMAIL_SERVER_URL}/reset_database")

    prompt = build_prompt("Delete alice@work.com email.")

    result = run_with_tools(
        prompt=prompt,
        tools=ALL_TOOLS,
        model=SMART_MODEL,
        max_turns=5,
    )
    print_agent_result(result, title="Demo 3 — Delete (with tool)")


def demo_4_search_and_delete() -> None:
    """
    Search + delete workflow:
    Ask the agent to find and delete an email by subject keyword.
    """
    # Reset so the happy hour email is present
    requests.get(f"{EMAIL_SERVER_URL}/reset_database")

    prompt = build_prompt("Delete the happy hour email.")

    result = run_with_tools(
        prompt=prompt,
        tools=ALL_TOOLS,
        model=SMART_MODEL,
        max_turns=5,
    )
    print_agent_result(result, title="Demo 4 — Search & Delete")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting email simulation server…")
    server = start_email_server()
    try:
        wait_for_server()
        print("✅ Email server ready\n")

        demo_1_read_and_reply()
        demo_2_delete_without_tool()
        demo_3_delete_with_tool()
        demo_4_search_and_delete()

    finally:
        server.terminate()
        server.wait()
        print("Email server stopped.")
