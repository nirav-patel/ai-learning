"""
Shared pytest fixtures for the multi-agent test suite.

All external services (LLM client, OpenAI images, Tavily) are replaced with
lightweight fakes so that unit tests run offline with zero API cost.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.inventory import create_inventory_dataframe
from tools.catalog_tool import ProductCatalogTool
from tools.registry import ToolRegistry
from tools.search_tool import TavilySearchTool


# ─── LLM response helpers ─────────────────────────────────────────────────────


def _make_llm_message(content: str, tool_calls: list | None = None) -> SimpleNamespace:
    """Return an object mimicking ``response.choices[0].message``."""
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return msg


def _make_llm_response(content: str, tool_calls: list | None = None) -> SimpleNamespace:
    """Return an object mimicking the full LLM completion response."""
    return SimpleNamespace(choices=[SimpleNamespace(message=_make_llm_message(content, tool_calls))])


def _make_tool_call(name: str, arguments: dict[str, Any], call_id: str = "call_001") -> SimpleNamespace:
    """Return an object mimicking an LLM tool-call entry."""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


# ─── LLM client fixture ───────────────────────────────────────────────────────


@pytest.fixture()
def mock_llm_client():
    """
    MagicMock of an aisuite client.

    By default each call returns a simple final text answer.
    Override ``mock_llm_client.chat.completions.create.return_value`` in
    individual tests to simulate tool calls or specific content.
    """
    client = MagicMock()
    client.chat.completions.create.return_value = _make_llm_response(
        "This is a mock LLM response."
    )
    return client


@pytest.fixture()
def mock_image_client():
    """MagicMock of an ``openai.OpenAI`` client for image generation."""
    # Create a 1×1 white PNG as a minimal valid PNG bytes
    import struct
    import zlib

    def _create_minimal_png() -> bytes:
        def _chunk(ctype: bytes, data: bytes) -> bytes:
            c = struct.pack(">I", len(data)) + ctype + data
            return c + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)

        header = b"\x89PNG\r\n\x1a\n"
        ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        raw = b"\x00\xff\xff\xff"  # filter byte + 1 RGB pixel
        idat = _chunk(b"IDAT", zlib.compress(raw))
        iend = _chunk(b"IEND", b"")
        return header + ihdr + idat + iend

    b64_png = base64.b64encode(_create_minimal_png()).decode()
    client = MagicMock()
    client.images.generate.return_value = SimpleNamespace(
        data=[SimpleNamespace(b64_json=b64_png)]
    )
    return client


# ─── Tool fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_search_tool():
    """TavilySearchTool with a patched internal client so no real HTTP call is made."""
    with patch("tools.search_tool.TavilyClient") as MockClient:
        MockClient.return_value.search.return_value = {
            "results": [
                {
                    "title": "Top Sunglasses Trends 2025",
                    "content": "Oversized and wraparound styles dominate.",
                    "url": "https://example.com/trends",
                }
            ],
            "images": [],
        }
        tool = TavilySearchTool(api_key="fake-key")
        yield tool


@pytest.fixture()
def catalog_tool():
    """ProductCatalogTool backed by the real in-memory DataFrame."""
    return ProductCatalogTool(seed=42)


@pytest.fixture()
def tool_registry(mock_search_tool, catalog_tool):
    """ToolRegistry pre-loaded with mock search tool + real catalog tool."""
    return ToolRegistry().register(mock_search_tool).register(catalog_tool)


# ─── Data fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def inventory_df() -> pd.DataFrame:
    """Fresh inventory DataFrame with a fixed seed."""
    return create_inventory_dataframe(seed=42)


# ─── Filesystem fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def tmp_output(tmp_path: Path) -> Path:
    """A temporary directory for output files (images, reports)."""
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture()
def sample_image(tmp_output: Path) -> Path:
    """Write a minimal PNG file to tmp_output and return its path."""
    import struct
    import zlib

    def _create_minimal_png() -> bytes:
        def _chunk(ctype: bytes, data: bytes) -> bytes:
            c = struct.pack(">I", len(data)) + ctype + data
            return c + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)

        header = b"\x89PNG\r\n\x1a\n"
        ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        raw = b"\x00\xff\xff\xff"
        idat = _chunk(b"IDAT", zlib.compress(raw))
        iend = _chunk(b"IEND", b"")
        return header + ihdr + idat + iend

    img_path = tmp_output / "test_image.png"
    img_path.write_bytes(_create_minimal_png())
    return img_path


# ─── Re-export helpers for test files ────────────────────────────────────────

__all__ = [
    "_make_llm_response",
    "_make_tool_call",
]
