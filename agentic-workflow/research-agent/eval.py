"""
Component-level evaluation for the research pipeline.

Evaluates the quality of web search results by comparing URLs returned
by the research step against a predefined list of preferred domains.

This is an objective, per-example evaluation (upper-left quadrant):
  - Each URL is compared against TOP_DOMAINS.
  - A PASS/FAIL flag is returned based on the ratio of preferred URLs.

Two evaluation functions are provided:

  evaluate_report_sources
      Extract and evaluate URLs from plain-text or Markdown report output.
      Mirrors the approach used in the reference M4 ungraded lab.

  evaluate_tool_call_sources
      Extract and evaluate Tavily URLs directly from the Bedrock
      tool-result message history — more precise because it checks
      *what the search tool actually returned*, not just what the model
      chose to cite.

Usage (standalone):
    python eval.py "path/to/report.txt"

Usage (programmatic):
    from eval import evaluate_report_sources, TOP_DOMAINS

    flag, report = evaluate_report_sources(my_report_text)
    print(report)
"""

# ================================
# Standard library imports
# ================================
import json
import re
import sys

# ---------------------------------------------------------------------------
# Preferred domains — ground truth for source quality
# ---------------------------------------------------------------------------

TOP_DOMAINS: set[str] = {
    # General reference / institutions / publishers
    "wikipedia.org", "nature.com", "science.org", "sciencemag.org", "cell.com",
    "mit.edu", "stanford.edu", "harvard.edu", "nasa.gov", "noaa.gov", "europa.eu",

    # CS / AI venues & indexes
    "arxiv.org", "acm.org", "ieee.org", "neurips.cc", "icml.cc", "openreview.net",

    # Other reputable scientific outlets
    "elifesciences.org", "pnas.org", "jmlr.org", "springer.com", "sciencedirect.com",

    # Space / astronomy
    "chandra.harvard.edu", "stsci.edu", "eso.org", "iau.org",
    "aanda.org", "iopscience.iop.org", "space.com",

    # Extra academic / government
    "pbs.org", "nih.gov", "cdc.gov", "nsf.gov", "energy.gov",

    # Tech reference
    "codecademy.com", "datacamp.com",
}

# Compiled once at import time
_URL_RE = re.compile(r"https?://[^\s\]\)>\}<>\"']+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _domain_from_url(url: str) -> str:
    """Extract hostname from a URL, stripping 'www.' prefix."""
    parts = url.split("/")
    host = parts[2] if len(parts) > 2 else url
    return host[4:] if host.startswith("www.") else host


# ---------------------------------------------------------------------------
# Evaluation 1 — plain-text / Markdown report
# ---------------------------------------------------------------------------

def evaluate_report_sources(
    report_text: str,
    top_domains: set[str] = TOP_DOMAINS,
    min_ratio: float = 0.4,
) -> tuple[bool, str]:
    """
    Evaluate whether URLs cited in a research report come from preferred domains.

    Extracts all URLs from the report text, compares each domain against
    ``top_domains``, and computes the ratio of preferred vs. total sources.

    This mirrors the evaluation approach demonstrated in the M4 ungraded lab:
    for each URL found in the research output, we check whether its domain
    is in the preferred list and report a PASS/FAIL based on the ratio.

    Parameters
    ----------
    report_text : Plain-text or Markdown research report (may contain URLs).
    top_domains : Set of trusted domain strings used for partial matching,
                  e.g., 'arxiv.org' matches 'export.arxiv.org'.
    min_ratio   : Minimum ratio of preferred URLs required to PASS (default 0.4).

    Returns
    -------
    tuple[bool, str]
        flag           — True if PASS, False if FAIL.
        markdown_report — Markdown-formatted evaluation summary.
    """
    print("\n🔍 Evaluating report sources against preferred domains...", report_text)
    urls = _URL_RE.findall(report_text)

    if not urls:
        return False, (
            "### Evaluation — Source Quality (Component-Level)\n"
            "⚠️  No URLs detected in the research report.\n"
            "Consider instructing the model to include full source URLs.\n"
        )

    total = len(urls)
    preferred_count = 0
    details = []

    for url in urls:
        domain = _domain_from_url(url)
        preferred = any(td in domain for td in top_domains)
        if preferred:
            preferred_count += 1
        label = "✅ PREFERRED" if preferred else "❌ NOT PREFERRED"
        details.append(f"- {url} → {label}")

    ratio = preferred_count / total if total > 0 else 0.0
    flag = ratio >= min_ratio
    status = "✅ PASS" if flag else "❌ FAIL"

    report = (
        f"\n### Evaluation — Source Quality (Component-Level)\n"
        f"- Total URLs found : {total}\n"
        f"- Preferred URLs   : {preferred_count}\n"
        f"- Ratio            : {ratio:.2%}\n"
        f"- Threshold        : {min_ratio:.0%}\n"
        f"- Status           : {status}\n"
        f"\n**Details:**\n"
        + "\n".join(details)
        + "\n"
    )
    return flag, report


# ---------------------------------------------------------------------------
# Evaluation 2 — Bedrock tool-result message history (Tavily-specific)
# ---------------------------------------------------------------------------

def evaluate_tool_call_sources(
    messages: list[dict],
    tool_name: str = "tavily_search_tool",
    top_domains: set[str] = TOP_DOMAINS,
    min_ratio: float = 0.4,
) -> tuple[bool, str]:
    """
    Evaluate whether Tavily results in the Bedrock message history come from
    preferred domains.

    Scans user messages that contain ``toolResult`` blocks, matches them against
    the tool calls made by the assistant, and evaluates URLs from those results.

    This is more precise than ``evaluate_report_sources`` because it checks
    *what the search tool returned* rather than what the model chose to cite.

    Parameters
    ----------
    messages   : Full message history from ``bedrock_client.run_tool_loop``.
    tool_name  : Name of the tool whose results to evaluate
                 (default: 'tavily_search_tool').
    top_domains: Set of preferred domain strings.
    min_ratio  : Minimum ratio for PASS (default 0.4).

    Returns
    -------
    tuple[bool, str]
        flag           — True if PASS, False if FAIL.
        markdown_report — Markdown-formatted evaluation summary.
    """
    # Step 1: Collect toolUseId values that belong to the target tool
    tool_use_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if "toolUse" not in block:
                continue
            tu = block["toolUse"]
            if tu.get("name") == tool_name:
                tool_use_ids.add(tu["toolUseId"])

    print(f"\n🔍 Evaluating {tool_use_ids} results against preferred domains...")
    # Step 2: x result items from matching toolResult blocks
    items: list[dict] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        for block in msg.get("content", []):
            if "toolResult" not in block:
                continue
            tr = block["toolResult"]
            if tr.get("toolUseId") not in tool_use_ids:
                continue
            for content_block in tr.get("content", []):
                if "text" not in content_block:
                    continue
                try:
                    data = json.loads(content_block["text"])
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "url" in item:
                                items.append(item)
                except (json.JSONDecodeError, ValueError):
                    pass

    if not items:
        return False, (
            f"### Evaluation — Tool Source Quality ({tool_name})\n"
            f"No {tool_name} results found in message history.\n"
        )

    total = len(items)
    preferred_count = 0
    details = []

    for item in items:
        url = item.get("url", "")
        title = item.get("title") or url
        domain = _domain_from_url(url)
        preferred = any(td in domain for td in top_domains)
        if preferred:
            preferred_count += 1
        label = "✅ PREFERRED" if preferred else "❌ NOT PREFERRED"
        details.append(f"- [{title}]({url}) — `{domain}` → {label}")

    ratio = preferred_count / total if total > 0 else 0.0
    flag = ratio >= min_ratio
    status = "✅ PASS" if flag else "❌ FAIL"

    report = (
        f"\n### Evaluation — Tool Source Quality ({tool_name})\n"
        f"- Total results    : {total}\n"
        f"- Preferred results: {preferred_count}\n"
        f"- Ratio            : {ratio:.2%}\n"
        f"- Threshold        : {min_ratio:.0%}\n"
        f"- Status           : {status}\n"
        f"\n**Details:**\n"
        + "\n".join(details)
        + "\n"
    )
    return flag, report


# ---------------------------------------------------------------------------
# CLI entry point — quick eval of a saved report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval.py <report_file_or_text>")
        sys.exit(1)

    arg = sys.argv[1]
    try:
        with open(arg, encoding="utf-8") as fh:
            text = fh.read()
    except FileNotFoundError:
        text = arg

    ok, md = evaluate_report_sources(text)
    print(md)
    sys.exit(0 if ok else 1)
