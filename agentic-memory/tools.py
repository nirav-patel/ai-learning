"""
Placeholder email and calendar tools for the response agent.

These are plain Python functions. LangGraph/LangChain converts them to tools
automatically via @tool or by passing them directly to create_react_agent.

Memory tools (manage_memory, search_memory) come from LangMem —
see email_assistant.py.
"""
from __future__ import annotations


def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email to the specified recipient."""
    return f"Email sent to {to} with subject '{subject}'"


def schedule_meeting(
    attendees: list, subject: str, duration_minutes: int, preferred_day: str
) -> str:
    """Schedule a calendar meeting with the given attendees."""
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"


def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"
