"""
Tools Demo — entry point.

Demonstrates four tool-calling scenarios that mirror the reference lab
(M3_UGL_1.md), adapted for AWS Bedrock:

  Demo 1 — Single tool, no args     : ask for the current time
  Demo 2 — Single tool, external API: ask for today's weather
  Demo 3 — Single tool with args    : create a reminder text file
  Demo 4 — Multiple tools in sequence: QR code + weather note in one prompt

Usage:
    python main.py
"""

from agent import run_with_tools
from display import print_agent_result
from tools import (
    generate_qr_code,
    get_current_time,
    get_weather_from_ip,
    write_txt_file,
)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL = "us.anthropic.claude-sonnet-4-6"

ALL_TOOLS = [get_current_time, get_weather_from_ip, write_txt_file, generate_qr_code]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_demo(title: str, prompt: str, tools: list) -> None:
    print("=" * 56)
    print(f"  {title}")
    print("=" * 56)
    print(f"  Prompt: {prompt}\n")
    result = run_with_tools(prompt, tools=tools, model=MODEL, max_turns=10)
    print_agent_result(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  Tools Demo — Turning Functions into LLM Tools")
    print("=" * 56)
    print(f"  Model: {MODEL}\n")

    # ── Demo 1: Single no-arg tool ────────────────────────────────────────
    _run_demo(
        "Demo 1 — Current Time",
        prompt="What time is it right now?",
        tools=ALL_TOOLS,
    )

    # ── Demo 2: External API tool ─────────────────────────────────────────
    _run_demo(
        "Demo 2 — Weather Lookup",
        prompt="Can you get the weather for my current location?",
        tools=ALL_TOOLS,
    )

    # ── Demo 3: Tool with arguments ───────────────────────────────────────
    _run_demo(
        "Demo 3 — Write a Reminder File",
        prompt=(
            "Can you make a txt note for me at output/reminders.txt "
            "that reminds me to call Daniel tomorrow at 7PM?"
        ),
        tools=ALL_TOOLS,
    )

    # ── Demo 4: Multiple tools in sequence ────────────────────────────────
    _run_demo(
        "Demo 4 — Multi-Tool: QR Code + Weather Note",
        prompt=(
            "Can you do two things: "
            "1) Generate a QR code for https://www.github.com and save it as output/github_qr.png. "
            "2) Write a txt file at output/weather_note.txt with today's current weather."
        ),
        tools=ALL_TOOLS,
    )
