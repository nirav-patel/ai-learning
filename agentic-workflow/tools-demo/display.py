"""
Console display helpers for the tools demo.

Terminal equivalent of the reference lab's display_functions.py —
prints the tool call trace, responses, and tool sequence in a readable
format without requiring Jupyter / IPython.
"""

import json
from agent import AgentResult


def print_agent_result(result: AgentResult) -> None:
    """
    Print a formatted trace of an AgentResult to the terminal.

    Shows each tool call (name + args), its output, the final answer,
    and a summary tool-call sequence — mirroring the reference lab's
    pretty_print_chat_completion layout.
    """
    sep = "─" * 56

    for call in result.tool_calls:
        args_str = json.dumps(call.args, indent=2) if call.args else "{}"
        print(f"\n  🧠 LLM Action : {call.tool_name}")
        print(f"     Args       :\n{_indent(args_str, 5)}")
        print(f"\n  🔧 Tool Result: {call.tool_name}")
        print(f"     {call.result}")

    print(f"\n  ✅ Final Answer:")
    print(f"     {result.answer}")

    if result.tool_calls:
        sequence = " → ".join(c.tool_name for c in result.tool_calls)
        print(f"\n  🧭 Tool Sequence: {sequence}")

    print()


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())
