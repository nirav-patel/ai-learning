"""
MCP server for the email simulation system.

Exposes all email operations as MCP tools using FastMCP's @mcp.tool()
decorator. Each tool calls the local FastAPI email REST server internally.

Run directly (stdio transport — used by EmailMCPClient):
    python -m mcp_server.email_mcp_server

The EMAIL_SERVER_URL environment variable controls which REST server to target
(default: http://127.0.0.1:5001).
"""

import os
import requests
from mcp.server.fastmcp import FastMCP

EMAIL_SERVER_URL = os.environ.get("EMAIL_SERVER_URL", "http://127.0.0.1:5001")

mcp = FastMCP("email-tools")


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def list_all_emails() -> list:
    """Fetch all emails stored in the system, ordered from newest to oldest."""
    return requests.get(f"{EMAIL_SERVER_URL}/emails").json()


@mcp.tool()
def list_unread_emails() -> list:
    """Fetch all unread emails only, ordered from newest to oldest."""
    return requests.get(f"{EMAIL_SERVER_URL}/emails/unread").json()


@mcp.tool()
def search_emails(query: str) -> list:
    """Search emails containing the query in subject, body, or sender.

    Args:
        query: Keyword or phrase to search for.
    """
    return requests.get(f"{EMAIL_SERVER_URL}/emails/search", params={"q": query}).json()


@mcp.tool()
def filter_emails(
    recipient: str = "",
    date_from: str = "",
    date_to: str = "",
) -> list:
    """Filter emails by recipient and/or date range.

    Args:
        recipient: Email address to filter by (leave empty to skip).
        date_from: Start date in YYYY-MM-DD format (leave empty to skip).
        date_to:   End date in YYYY-MM-DD format (leave empty to skip).
    """
    params = {}
    if recipient:
        params["recipient"] = recipient
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    return requests.get(f"{EMAIL_SERVER_URL}/emails/filter", params=params).json()


@mcp.tool()
def get_email(email_id: int) -> dict:
    """Retrieve a specific email by its integer ID.

    Args:
        email_id: The unique ID of the email to fetch.
    """
    return requests.get(f"{EMAIL_SERVER_URL}/emails/{email_id}").json()


@mcp.tool()
def mark_email_as_read(email_id: int) -> dict:
    """Mark a specific email as read.

    Args:
        email_id: The ID of the email to mark as read.
    """
    return requests.patch(f"{EMAIL_SERVER_URL}/emails/{email_id}/read").json()


@mcp.tool()
def mark_email_as_unread(email_id: int) -> dict:
    """Mark a specific email as unread.

    Args:
        email_id: The ID of the email to mark as unread.
    """
    return requests.patch(f"{EMAIL_SERVER_URL}/emails/{email_id}/unread").json()


@mcp.tool()
def send_email(recipient: str, subject: str, body: str) -> dict:
    """Send an email. The sender is automatically set to you@email.com.

    Args:
        recipient: The email address of the recipient.
        subject:   Subject line of the email.
        body:      Message body content.
    """
    payload = {"recipient": recipient, "subject": subject, "body": body}
    return requests.post(f"{EMAIL_SERVER_URL}/send", json=payload).json()


@mcp.tool()
def delete_email(email_id: int) -> dict:
    """Permanently delete an email by its ID.

    Args:
        email_id: The ID of the email to delete.
    """
    return requests.delete(f"{EMAIL_SERVER_URL}/emails/{email_id}").json()


@mcp.tool()
def search_unread_from_sender(sender: str) -> list:
    """Return all unread emails from a specific sender address.

    Args:
        sender: Email address of the sender to filter by.
    """
    unread = requests.get(f"{EMAIL_SERVER_URL}/emails/unread").json()
    return [e for e in unread if e["sender"].lower() == sender.lower()]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
