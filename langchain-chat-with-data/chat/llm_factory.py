"""LLM factory — AWS Bedrock ChatBedrock."""
from __future__ import annotations

import boto3
import certifi
from botocore.exceptions import ClientError, NoCredentialsError

from .config import AppConfig


def _validate_aws_session(config: AppConfig) -> None:
    """Fail fast with an actionable message when AWS credentials are unusable."""
    sts_client = boto3.client(
        "sts",
        region_name=config.aws_region,
        verify=certifi.where(),
    )
    try:
        sts_client.get_caller_identity()
    except NoCredentialsError as exc:
        raise RuntimeError(
            "AWS credentials were not found. Configure or refresh your AWS session before starting the app."
        ) from exc
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code == "ExpiredTokenException":
            raise RuntimeError(
                "AWS credentials are expired. Refresh your AWS session, then restart the app."
            ) from exc
        raise RuntimeError(
            f"AWS credential validation failed: {error_code or str(exc)}"
        ) from exc


def make_llm(config: AppConfig):
    """Return a ChatBedrock instance configured for the given AppConfig."""
    from langchain_aws import ChatBedrock

    _validate_aws_session(config)

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
