"""
Pydantic model and LangGraph state definition.

Mirrors the sample notebook's schemas.py, adapted for LangGraph v1.
"""
from __future__ import annotations

from typing import Optional
from typing_extensions import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(description="Step-by-step reasoning behind the classification.")
    classification: Literal["ignore", "respond", "notify"] = Field(
        description=(
            "The classification of an email: 'ignore' for irrelevant emails, "
            "'notify' for important information that doesn't need a response, "
            "'respond' for emails that need a reply."
        )
    )


class State(TypedDict):
    email_input: dict
    messages: Annotated[list[AnyMessage], add_messages]
    classification: NotRequired[str]
    reasoning: NotRequired[str]
