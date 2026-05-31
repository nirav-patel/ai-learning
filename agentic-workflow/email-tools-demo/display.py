"""
Terminal display for the email agent demo.

Prints the agentic loop trace (tool calls + results) and the final answer
in a readable, colour-coded format.
"""

import json
from agent import AgentResult


_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_GREEN  = "\033[32m"
_DIM    = "\033[2m"
_BLUE   = "\033[34m"


def _json_or_str(value: str) -> str:
    """Pretty-print a string if it is valid JSON, otherwise return as-is."""
    try:
        parsed = json.loads(value)
        return json.dumps(parsed, indent=2)
    except (json.JSONDecodeError, TypeError):
        return value


def print_agent_result(result: AgentResult, title: str = "") -> None:
    """
    Print a full agentic run trace to the terminal.

    Shows:
    - The prompt
    - Each tool call with its args and result
    - The tool invocation sequence
    - The final model answer
    """
    width = 70
    header = f" {title} " if title else " Agent Run "
    print(f"\n{_BOLD}{_CYAN}{'─' * width}{_RESET}")
    print(f"{_BOLD}{_CYAN}{header.center(width)}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'─' * width}{_RESET}")

    # Prompt
    print(f"\n{_BOLD}Prompt:{_RESET}")
    print(f"  {result.prompt}\n")

    # Tool calls
    if result.tool_calls:
        print(f"{_BOLD}Tool calls:{_RESET}")
        for i, tc in enumerate(result.tool_calls, 1):
            args_str = json.dumps(tc.args, indent=4) if tc.args else "(no args)"
            result_str = _json_or_str(tc.result)

            print(f"\n  {_YELLOW}[{i}] 🧠 {tc.tool_name}{_RESET}")
            print(f"  {_DIM}Args:{_RESET}")
            for line in args_str.splitlines():
                print(f"      {line}")
            print(f"  {_BLUE}Result:{_RESET}")
            for line in result_str.splitlines()[:15]:   # cap noisy list output
                print(f"      {line}")
            if len(result_str.splitlines()) > 15:
                print(f"      {_DIM}… ({len(result_str.splitlines())} lines total){_RESET}")

        # Tool sequence summary
        sequence = " → ".join(tc.tool_name for tc in result.tool_calls)
        print(f"\n  {_DIM}Sequence: {sequence}{_RESET}")

    # Final answer
    print(f"\n{_BOLD}{_GREEN}✅ Final Answer:{_RESET}")
    for line in result.answer.splitlines():
        print(f"  {line}")
    print(f"\n{_CYAN}{'─' * width}{_RESET}\n")
