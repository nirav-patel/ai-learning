"""
Research tools for the tool-calling pipeline.

Provides three search tools:
  - arxiv_search_tool     — academic papers via the arXiv API
  - tavily_search_tool    — general web search via the Tavily API
  - wikipedia_search_tool — encyclopaedic summaries via the Wikipedia REST API

Each tool ships with two flavours of metadata:
  - OpenAI-style function definitions  (arxiv_tool_def / tavily_tool_def / wikipedia_tool_def)
  - Bedrock-style toolSpec definitions (ARXIV_BEDROCK_TOOL_DEF / TAVILY_BEDROCK_TOOL_DEF /
                                        WIKIPEDIA_BEDROCK_TOOL_DEF)

TOOL_MAPPING maps tool names (strings) to the actual Python callables so that
an agent loop can look up and execute a tool by name.
"""

# ================================
# Standard library imports
# ================================
import json
import os
import xml.etree.ElementTree as ET

# ================================
# Third-party imports
# ================================
import requests
from dotenv import load_dotenv
from tavily import TavilyClient

# ================================
# Environment setup
# ================================
load_dotenv()

_WIKI_UA = "ResearchAgent/1.0 (mailto:dev@example.com)"

_session = requests.Session()
_session.headers.update(
    {"User-Agent": "LF-ADP-Agent/1.0 (mailto:your.email@example.com)"}
)


# ---------------------------------------------------------------------------
# Tool: arXiv search
# ---------------------------------------------------------------------------

def arxiv_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """
    Search arXiv for research papers matching the given query.

    Returns a list of dicts with keys:
        title, authors, published, url, summary, link_pdf
    """
    url = (
        f"https://export.arxiv.org/api/query"
        f"?search_query=all:{query}&start=0&max_results={max_results}"
    )

    try:
        response = _session.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return [{"error": str(e)}]

    try:
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        results = []
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns).text.strip()
            authors = [
                author.find("atom:name", ns).text
                for author in entry.findall("atom:author", ns)
            ]
            published = entry.find("atom:published", ns).text[:10]
            url_abstract = entry.find("atom:id", ns).text
            summary = entry.find("atom:summary", ns).text.strip()

            link_pdf = None
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    link_pdf = link.attrib.get("href")
                    break

            results.append(
                {
                    "title": title,
                    "authors": authors,
                    "published": published,
                    "url": url_abstract,
                    "summary": summary,
                    "link_pdf": link_pdf,
                }
            )
        return results

    except Exception as e:
        return [{"error": f"Parsing failed: {str(e)}"}]


# ---------------------------------------------------------------------------
# Tool: Tavily web search
# ---------------------------------------------------------------------------

def tavily_search_tool(
    query: str, max_results: int = 5, include_images: bool = False
) -> list[dict]:
    """
    Perform a general-purpose web search using the Tavily API.

    Returns a list of dicts with keys: title, content, url.
    When include_images=True, image URL dicts are appended.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables.")

    api_base_url = os.getenv("DLAI_TAVILY_BASE_URL")
    client = TavilyClient(api_key=api_key, api_base_url=api_base_url)

    try:
        response = client.search(
            query=query, max_results=max_results, include_images=include_images
        )

        results = []
        for r in response.get("results", []):
            results.append(
                {
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                }
            )

        if include_images:
            for img_url in response.get("images", []):
                results.append({"image_url": img_url})

        return results

    except Exception as e:
        return [{"error": str(e)}]


# ---------------------------------------------------------------------------
# Tool: Wikipedia search
# ---------------------------------------------------------------------------

def wikipedia_search_tool(query: str, sentences: int = 5) -> list[dict]:
    """
    Search Wikipedia for a summary of the given query using the Wikipedia REST API.

    Returns a list with a single dict containing: title, summary, url.
    On error returns [{"error": "<message>"}].
    """
    _wiki_headers = {"User-Agent": _WIKI_UA}
    try:
        # Step 1: find the best matching page title
        search_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1,
            },
            headers=_wiki_headers,
            timeout=15,
        )
        search_resp.raise_for_status()
        results = search_resp.json().get("query", {}).get("search", [])
        if not results:
            return [{"error": f"No Wikipedia results found for: {query}"}]

        page_title = results[0]["title"]

        # Step 2: fetch the page extract via the REST summary endpoint
        summary_resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(page_title)}",
            headers=_wiki_headers,
            timeout=15,
        )
        summary_resp.raise_for_status()
        data = summary_resp.json()

        extract = data.get("extract", "")
        # Trim to requested number of sentences
        truncated = ". ".join(extract.split(". ")[:sentences])

        return [
            {
                "title": data.get("title", page_title),
                "summary": truncated,
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            }
        ]
    except Exception as e:
        return [{"error": str(e)}]


# ---------------------------------------------------------------------------
# Tool definitions — OpenAI format (kept for reference)
# ---------------------------------------------------------------------------

arxiv_tool_def = {
    "type": "function",
    "function": {
        "name": "arxiv_search_tool",
        "description": "Searches for research papers on arXiv by query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for research papers.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

tavily_tool_def = {
    "type": "function",
    "function": {
        "name": "tavily_search_tool",
        "description": "Performs a general-purpose web search using the Tavily API.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for retrieving information from the web.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
                "include_images": {
                    "type": "boolean",
                    "description": "Whether to include image results.",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    },
}

wikipedia_tool_def = {
    "type": "function",
    "function": {
        "name": "wikipedia_search_tool",
        "description": "Searches for a Wikipedia article summary by query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for the Wikipedia article.",
                },
                "sentences": {
                    "type": "integer",
                    "description": "Number of sentences in the summary.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# Tool definitions — AWS Bedrock toolSpec format
# ---------------------------------------------------------------------------

ARXIV_BEDROCK_TOOL_DEF = {
    "toolSpec": {
        "name": "arxiv_search_tool",
        "description": (
            "Searches arXiv for academic research papers matching the query. "
            "Returns titles, authors, publication dates, abstracts, and URLs."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords for research papers.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 5).",
                    },
                },
                "required": ["query"],
            }
        },
    }
}

TAVILY_BEDROCK_TOOL_DEF = {
    "toolSpec": {
        "name": "tavily_search_tool",
        "description": (
            "Performs a general-purpose web search using the Tavily API. "
            "Returns titles, content snippets, and source URLs."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords for web search.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 5).",
                    },
                    "include_images": {
                        "type": "boolean",
                        "description": "Whether to include image URL results.",
                    },
                },
                "required": ["query"],
            }
        },
    }
}

WIKIPEDIA_BEDROCK_TOOL_DEF = {
    "toolSpec": {
        "name": "wikipedia_search_tool",
        "description": (
            "Searches Wikipedia for an encyclopaedic article summary. "
            "Returns the article title, a plain-text summary, and the page URL."
        ),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords for the Wikipedia article.",
                    },
                    "sentences": {
                        "type": "integer",
                        "description": "Number of sentences to include in the summary (default 5).",
                    },
                },
                "required": ["query"],
            }
        },
    }
}


# ---------------------------------------------------------------------------
# Tool mapping — name → callable
# ---------------------------------------------------------------------------

TOOL_MAPPING: dict[str, callable] = {
    "arxiv_search_tool": arxiv_search_tool,
    "tavily_search_tool": tavily_search_tool,
    "wikipedia_search_tool": wikipedia_search_tool,
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def parse_input(text_or_messages) -> str:
    """
    Accept either a plain string or a messages list (as returned by a tool-calling
    loop) and return the last assistant text content as a plain string.
    """
    if isinstance(text_or_messages, list):
        text_report = None
        for m in reversed(text_or_messages):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            content = (
                m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            )
            if role == "assistant" and content:
                # Bedrock messages have content as a list of blocks
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            text_report = block["text"]
                            break
                else:
                    text_report = str(content)
            if text_report:
                break
        if not text_report:
            raise ValueError("No assistant text found in messages.")
        return text_report
    else:
        return str(text_or_messages)
