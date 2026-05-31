"""
Email tool functions for the LLM agent.

Each function calls the local email simulation server via REST.
These are passed as tools to run_with_tools() so the LLM can
invoke them automatically in response to natural-language prompts.
"""

import os
import requests

EMAIL_SERVER_URL = os.environ.get("EMAIL_SERVER_URL", "http://127.0.0.1:5000")


def list_all_emails() -> list:
    """
    Fetch all emails stored in the system, ordered from newest to oldest.

    Returns:
        List of all emails including read and unread. Each email has keys:
        id, sender, recipient, subject, body, timestamp, read.
    """
    return requests.get(f"{EMAIL_SERVER_URL}/emails").json()


def list_unread_emails() -> list:
    """
    Fetch all unread emails only, ordered from newest to oldest.

    Returns:
        List of unread emails (read == False). Same structure as list_all_emails.
    """
    return requests.get(f"{EMAIL_SERVER_URL}/emails/unread").json()


def search_emails(query: str) -> list:
    """
    Search emails containing the query in subject, body, or sender.

    Args:
        query: A keyword or phrase to search for.

    Returns:
        List of emails matching the query string.
    """
    return requests.get(f"{EMAIL_SERVER_URL}/emails/search", params={"q": query}).json()


def filter_emails(recipient: str = None, date_from: str = None, date_to: str = None) -> list:
    """
    Filter emails by recipient and/or a date range.

    Args:
        recipient: Email address to filter by (optional).
        date_from: Start date in YYYY-MM-DD format (optional).
        date_to: End date in YYYY-MM-DD format (optional).

    Returns:
        List of emails matching the given filters.
    """
    params = {}
    if recipient:
        params["recipient"] = recipient
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    return requests.get(f"{EMAIL_SERVER_URL}/emails/filter", params=params).json()


def get_email(email_id: int) -> dict:
    """
    Retrieve a specific email by its ID.

    Args:
        email_id: The unique integer ID of the email to fetch.

    Returns:
        A single email record dict, or an error if not found.
    """
    return requests.get(f"{EMAIL_SERVER_URL}/emails/{email_id}").json()


def mark_email_as_read(email_id: int) -> dict:
    """
    Mark a specific email as read.

    Args:
        email_id: The ID of the email to mark as read.

    Returns:
        The updated email record with read set to true.
    """
    return requests.patch(f"{EMAIL_SERVER_URL}/emails/{email_id}/read").json()


def mark_email_as_unread(email_id: int) -> dict:
    """
    Mark a specific email as unread.

    Args:
        email_id: The ID of the email to mark as unread.

    Returns:
        The updated email record with read set to false.
    """
    return requests.patch(f"{EMAIL_SERVER_URL}/emails/{email_id}/unread").json()


def send_email(recipient: str, subject: str, body: str) -> dict:
    """
    Send an email. The sender is set automatically to you@email.com.

    Args:
        recipient: The email address of the recipient.
        subject: The subject line of the email.
        body: The message body content.

    Returns:
        The created email record.
    """
    payload = {"recipient": recipient, "subject": subject, "body": body}
    return requests.post(f"{EMAIL_SERVER_URL}/send", json=payload).json()


def delete_email(email_id: int) -> dict:
    """
    Permanently delete an email by its ID.

    Args:
        email_id: The ID of the email to delete.

    Returns:
        Confirmation dict with key 'message'.
    """
    return requests.delete(f"{EMAIL_SERVER_URL}/emails/{email_id}").json()


def search_unread_from_sender(sender: str) -> list:
    """
    Return all unread emails from a specific sender address.

    Args:
        sender: The email address of the sender to filter by.

    Returns:
        List of unread emails whose sender matches the given address.
    """
    unread = list_unread_emails()
    return [e for e in unread if e["sender"].lower() == sender.lower()]
