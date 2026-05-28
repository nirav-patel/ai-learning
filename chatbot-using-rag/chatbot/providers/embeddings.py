"""providers/embeddings.py — Embedding factory: huggingface | openai | ollama.

Select the provider via AppConfig.embed_provider (or EMBED_PROVIDER env-var).

HuggingFace notes:
  nomic-embed-text-v1.5 requires trust_remote_code=True, batch_size=8, and
  task-specific prefixes ("search_document: " / "search_query: ").
  all-MiniLM-L6-v2 requires no special configuration.
"""
from __future__ import annotations

from ..config import AppConfig


def make_embeddings(config: AppConfig):
    """Return an embeddings object for the configured provider.

    Supported providers
    -------------------
    huggingface : HuggingFaceEmbeddings (local, no API key)
    openai      : OpenAIEmbeddings — requires langchain-openai, OPENAI_API_KEY
    ollama      : OllamaEmbeddings — requires langchain-ollama, running Ollama
    """
    provider = config.embed_provider.lower().strip()
    if provider == "huggingface":
        return _huggingface(config)
    if provider == "openai":
        return _openai(config)
    if provider == "ollama":
        return _ollama(config)
    raise ValueError(
        f"Unknown embedding provider '{config.embed_provider}'. "
        "Supported: huggingface | openai | ollama"
    )


def _huggingface(config: AppConfig):
    from langchain_huggingface import HuggingFaceEmbeddings

    if "nomic" in config.embed_model_name.lower():

        class _NomicEmbeddings(HuggingFaceEmbeddings):
            """Adds required task prefixes for nomic-embed-text-v1.5."""

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return super().embed_documents(["search_document: " + t for t in texts])

            def embed_query(self, text: str) -> list[float]:
                return super().embed_query("search_query: " + text)

        return _NomicEmbeddings(
            model_name=config.embed_model_name,
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"batch_size": 8},
        )

    return HuggingFaceEmbeddings(model_name=config.embed_model_name)


def _openai(config: AppConfig):
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=config.embed_model_name)


def _ollama(config: AppConfig):
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=config.embed_model_name, base_url=config.ollama_base_url)
