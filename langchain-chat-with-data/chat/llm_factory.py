"""LLM factory — AWS Bedrock ChatBedrock."""
from __future__ import annotations

import boto3
import certifi

from .config import AppConfig


def make_llm(config: AppConfig):
    """Return a ChatBedrock instance configured for the given AppConfig."""
    from langchain_aws import ChatBedrock

    client = boto3.client(
        "bedrock-runtime",
        region_name=config.aws_region,
        verify=certifi.where(),
    )
    return ChatBedrock(
        client=client,
        model_id=config.llm_model_id,
        model_kwargs={"temperature": 0.0},
    )
