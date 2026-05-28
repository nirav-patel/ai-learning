"""providers/llm.py — LLM factory: bedrock | openai | ollama.

Select the provider via AppConfig.llm_provider (or LLM_PROVIDER env-var).
Only the active provider's package needs to be installed.
"""
from __future__ import annotations

from ..config import AppConfig


def make_llm(config: AppConfig):
    """Return a ChatModel for the configured LLM provider.

    Supported providers
    -------------------
    bedrock  : AWS Bedrock (ChatBedrock) — requires boto3, langchain-aws
    openai   : OpenAI ChatCompletion    — requires langchain-openai, OPENAI_API_KEY
    ollama   : Local Ollama server      — requires langchain-ollama, running Ollama
    """
    provider = config.llm_provider.lower().strip()
    if provider == "bedrock":
        return _bedrock(config)
    if provider == "openai":
        return _openai(config)
    if provider == "ollama":
        return _ollama(config)
    raise ValueError(
        f"Unknown LLM provider '{config.llm_provider}'. "
        "Supported: bedrock | openai | ollama"
    )


def _bedrock(config: AppConfig):
    import boto3
    import certifi
    from botocore.exceptions import ClientError, NoCredentialsError
    from langchain_aws import ChatBedrock

    _validate_aws(config)
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


def _openai(config: AppConfig):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=config.llm_model_id, temperature=0.0, streaming=True)


def _ollama(config: AppConfig):
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=config.llm_model_id,
        base_url=config.ollama_base_url,
        temperature=0.0,
    )


def _validate_aws(config: AppConfig) -> None:
    """Fail fast with an actionable message when AWS credentials are unusable."""
    import boto3
    import certifi
    from botocore.exceptions import ClientError, NoCredentialsError

    try:
        boto3.client("sts", region_name=config.aws_region, verify=certifi.where()).get_caller_identity()
    except NoCredentialsError as exc:
        raise RuntimeError(
            "AWS credentials not found. Configure or refresh your AWS session before starting the app."
        ) from exc
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code == "ExpiredTokenException":
            raise RuntimeError("AWS credentials are expired. Refresh your session and restart.") from exc
        raise RuntimeError(f"AWS credential validation failed: {code or str(exc)}") from exc
