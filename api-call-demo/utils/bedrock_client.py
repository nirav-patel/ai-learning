"""
AWS Bedrock client setup and LLM call helpers
=============================================
This module is the single source of truth for:
  - MODEL_ID                      : the Claude model used across all demos
  - create_client()               : builds an authenticated boto3 bedrock-runtime client
  - get_completion()              : simple one-shot Q&A (no system message)
  - get_completion_sys_role()     : call with a system role + message(s)
  - get_completion_and_token_count(): like get_completion_sys_role but also returns token usage
"""
from __future__ import annotations

import os

import boto3
import certifi

# The Claude model to use for all demos.
# Change this one constant to switch models project-wide.
MODEL_ID = "us.anthropic.claude-sonnet-4-6"


def create_client():
    """
    Create and return an AWS Bedrock runtime client.
    Reads AWS_REGION from the environment (defaults to us-west-2).
    """
    return boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-west-2"),
        verify=certifi.where(),
    )


def get_completion(client, question: str) -> str:
    """
    Simple one-shot completion: send a single user question and return the answer.
    No system message — good for quick/simple queries.
    """
    response = client.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": question}]}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.0},
    )
    return response["output"]["message"]["content"][0]["text"]


def get_completion_sys_role(client, role: str, question_or_messages) -> str:
    """
    Completion with a system role.

    Args:
        client:               Bedrock runtime client.
        role:                 System message text (sets the model's persona/behavior).
        question_or_messages: Either a plain string (single user turn) or a
                              list of message dicts for multi-turn conversations.
    Returns:
        The model's response text.
    """
    system = [{"text": role}]
    if isinstance(question_or_messages, str):
        messages = [{"role": "user", "content": [{"text": question_or_messages}]}]
    else:
        messages = question_or_messages
    response = client.converse(
        modelId=MODEL_ID,
        system=system,
        messages=messages,
        inferenceConfig={"maxTokens": 512, "temperature": 0.0},
    )
    return response["output"]["message"]["content"][0]["text"]


def get_completion_and_token_count(client, system_message: str, user_message: str) -> tuple[str, dict]:
    """
    Like get_completion_sys_role but also returns token usage metrics.

    Returns:
        (response_text, token_dict) where token_dict has keys:
          prompt_tokens, completion_tokens, total_tokens
    """
    system = [{"text": system_message}]
    messages = [{"role": "user", "content": [{"text": user_message}]}]
    response = client.converse(
        modelId=MODEL_ID,
        system=system,
        messages=messages,
        inferenceConfig={"maxTokens": 512, "temperature": 0.0},
    )
    content = response["output"]["message"]["content"][0]["text"]
    usage = response["usage"]
    token_dict = {
        "prompt_tokens": usage["inputTokens"],
        "completion_tokens": usage["outputTokens"],
        "total_tokens": usage["totalTokens"],
    }
    return content, token_dict
