"""
LangChain: Agents — Amazon Bedrock Edition
==========================================
Based on the DeepLearning.AI "LangChain for LLM Application Development" L6
course notebook, adapted to use AWS Bedrock (Claude) instead of OpenAI.

════════════════════════════════════════════════════════════════
HOW THE AGENT LOOP WORKS
════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────┐
  │  User Question  →  AgentExecutor                        │
  │                          │                              │
  │          ┌───────────────▼──────────────────┐           │
  │          │   ReAct Prompt (Thought/Action)   │           │
  │          └───────────────┬──────────────────┘           │
  │                          │                              │
  │          ┌───────────────▼──────────────────┐           │
  │          │     LLM (Claude on Bedrock)       │           │
  │          │  Decides: Action + Action Input   │           │
  │          └───────────────┬──────────────────┘           │
  │                          │                              │
  │          ┌───────────────▼──────────────────┐           │
  │          │  Tool Execution                  │           │
  │          │  (Wikipedia / Calculator /       │           │
  │          │   Python REPL / Date)            │           │
  │          └───────────────┬──────────────────┘           │
  │                          │                              │
  │          ┌───────────────▼──────────────────┐           │
  │          │  Observation fed back to LLM      │           │
  │          │  → Thought → Final Answer?        │           │
  │          └─────── loop until done ──────────┘           │
  └─────────────────────────────────────────────────────────┘

AGENT PATTERN: ReAct (Reason + Act)
  The LLM alternates between:
    Thought:  "I need to look this up …"
    Action:   <tool name>
    Action Input: <input string>
    Observation: <tool result>
  …until it emits "Final Answer: <answer>".

TOOLS IN THIS DEMO
  1. Calculator    — llm-math style: LLM translates question → expression → numexpr eval
  2. Wikipedia     — custom @tool over MediaWiki REST API with certifi SSL
  3. Python REPL   — executes arbitrary Python code (⚠ use in sandbox only)
  4. today_date    — custom @tool: returns today's date string

DEMO RUNS
  A. Math question   — "What is 25% of 300?"
  B. Wikipedia       — "What book did Tom M. Mitchell write?"
  C. Python REPL     — Sort a customer list by last name then first name
  D. Custom tool     — "What's today's date?"

RUN
---
    cd lang-chain-demo
    python l6_agents_demo.py

DEPENDENCIES (see ../requirements.txt)
---------------------------------------
    langchain, langchain-aws, langchain-core, langchain-classic,
    langchain-community, langchain-experimental, numexpr,
    requests, boto3, certifi, python-dotenv
"""

from __future__ import annotations

import logging
import os
import sys
import textwrap
import warnings

# ── Suppress noisy third-party warnings BEFORE importing them ─────────────────
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import certifi
import boto3
import numexpr
import requests
from datetime import date
from dotenv import load_dotenv, find_dotenv

from langchain_aws import ChatBedrock
from langchain_classic.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_experimental.tools.python.tool import PythonREPLTool

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv(find_dotenv())

# ── Configuration ─────────────────────────────────────────────────────────────
LLM_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
AWS_REGION   = os.getenv("AWS_REGION", "us-west-1")

# ── ReAct system prompt ───────────────────────────────────────────────────────
# Standard ReAct template; {tools}, {tool_names}, {input}, {agent_scratchpad}
# are the four variables expected by create_react_agent.
_REACT_TEMPLATE = """\
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

REACT_PROMPT = PromptTemplate.from_template(_REACT_TEMPLATE)


# ── Helper builders ───────────────────────────────────────────────────────────

def _make_llm() -> ChatBedrock:
    client = boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        verify=certifi.where(),
    )
    return ChatBedrock(
        client=client,
        model_id=LLM_MODEL_ID,
        model_kwargs={"temperature": 0.0},
    )


# ── Tool definitions ──────────────────────────────────────────────────────────

_WIKI_API = "https://en.wikipedia.org/w/api.php"


def _make_llm_math_tool(llm: ChatBedrock):
    """
    Implements the llm-math pattern from the original notebook:
      1. The LLM translates the user’s natural-language question into a
         plain Python math expression.
      2. numexpr evaluates the expression and returns the numeric result.

    Why not load_tools(["llm-math"], llm=llm)?
    The underlying LLMMathChain uses a Pydantic v1 class whose __annotate__
    method contains `dict[str, Any]` evaluated as a runtime call.  Python 3.14
    changed how inspect.signature resolves annotations and this triggers
    `TypeError: 'function' object is not subscriptable`.  The manual shim below
    is functionally identical but avoids that incompatibility.
    """
    _math_prompt = PromptTemplate.from_template(
        "Translate the following math question into a single Python expression "
        "using only numbers and basic operators (+, -, *, /, **). "
        "Output ONLY the expression, no words or explanation.\n\n"
        "Question: {question}\n"
        "Expression:"
    )
    _math_chain = _math_prompt | llm

    @tool
    def Calculator(question: str) -> str:  # noqa: N802
        """
        Useful for answering math questions.
        Input should be a natural-language math question.
        """
        try:
            msg        = _math_chain.invoke({"question": question})
            expression = msg.content.strip()
            result     = numexpr.evaluate(expression)
            return f"Answer: {float(result)}"
        except Exception as exc:
            return f"Calculator error: {exc}"

    return Calculator


def _make_wikipedia_tool():
    """
    Custom Wikipedia tool that calls the MediaWiki REST API directly with
    certifi-verified SSL.

    Root cause of the WikipediaQueryRun / wikipedia-package JSONDecodeError:
    the `wikipedia` PyPI package does not pass verify=certifi.where() to its
    internal requests session, so on some macOS systems the TLS handshake
    fails silently and returns an empty body — causing json.loads to raise
    "Expecting value: line 1 column 1 (char 0)".

    Fix: bypass the package entirely and call the API with requests + certifi.
    """
    @tool
    def wikipedia(query: str) -> str:
        """Search Wikipedia and return a short article summary for the query."""
        try:
            # Step 1: find the best-matching page title
            search_resp = requests.get(
                _WIKI_API,
                params={
                    "action": "query",
                    "list":   "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": 1,
                },
                verify=certifi.where(),
                timeout=10,
            )
            search_resp.raise_for_status()
            results = search_resp.json().get("query", {}).get("search", [])
            if not results:
                return "No Wikipedia results found."

            title = results[0]["title"]

            # Step 2: fetch the plain-text intro extract (≈3 sentences)
            extract_resp = requests.get(
                _WIKI_API,
                params={
                    "action":      "query",
                    "prop":        "extracts",
                    "exintro":     True,
                    "explaintext": True,
                    "exsentences": 3,
                    "titles":      title,
                    "format":      "json",
                },
                verify=certifi.where(),
                timeout=10,
            )
            extract_resp.raise_for_status()
            pages = extract_resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                return page.get("extract", "No extract available.")
            return "No result found."
        except Exception as exc:
            return f"Wikipedia lookup failed: {exc}"

    return wikipedia


def _make_python_repl_tool():
    """
    Python REPL tool from langchain-experimental.
    ⚠ WARNING: executes arbitrary Python code in the current process.
       Only use in a trusted / sandboxed environment.
    """
    return PythonREPLTool()


@tool
def today_date(text: str) -> str:
    """
    Returns today's date.
    Use this for any questions about today's date or the current date.
    The input should always be an empty string.
    """
    return str(date.today())


# ── Agent builder ─────────────────────────────────────────────────────────────

def _build_agent(llm: ChatBedrock, tools: list) -> AgentExecutor:
    """
    Wire the LLM and tools into a ReAct AgentExecutor.

    create_react_agent:
      Builds a Runnable that formats the ReAct prompt, calls the LLM,
      and parses its Thought/Action output.

    AgentExecutor:
      Orchestrates the loop: run agent → call tool → feed observation back →
      repeat until the agent emits a Final Answer.
    """
    agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )


# ── Demo A: Math ──────────────────────────────────────────────────────────────

def demo_math(executor: AgentExecutor) -> None:
    """
    Ask the agent a simple arithmetic question.
    The LLM-math tool internally asks the LLM to translate the question into
    a math expression, evaluates it, and returns the numeric answer.
    Expected flow: Thought → Action: Calculator → Observation → Final Answer.
    """
    print("\n" + "=" * 60)
    print("DEMO A — Math: 'What is 25% of 300?'")
    print("=" * 60)
    result = executor.invoke({"input": "What is the 25% of 300?"})
    print("\n>>> Final output:", result.get("output"))


# ── Demo B: Wikipedia ─────────────────────────────────────────────────────────

def demo_wikipedia(executor: AgentExecutor) -> None:
    """
    Ask a factual question answered by Wikipedia.
    Expected flow: Thought → Action: wikipedia → Observation → Final Answer.
    """
    print("\n" + "=" * 60)
    print("DEMO B — Wikipedia: Tom M. Mitchell's book")
    print("=" * 60)
    question = (
        "Tom M. Mitchell is an American computer scientist and the Founders "
        "University Professor at Carnegie Mellon University (CMU). "
        "What book did he write?"
    )
    result = executor.invoke({"input": question})
    print("\n>>> Final output:", result.get("output"))


# ── Demo C: Python REPL ───────────────────────────────────────────────────────

def demo_python_repl(executor: AgentExecutor) -> None:
    """
    Ask the agent to sort a customer list using the Python REPL tool.
    The LLM writes a small Python snippet, the REPL executes it, and the
    printed output is returned as the Observation.

    ⚠ The Python REPL executes code in-process — keep inputs trusted.
    """
    print("\n" + "=" * 60)
    print("DEMO C — Python REPL: Sort customer list")
    print("=" * 60)
    customer_list = [
        ["Harrison", "Chase"],
        ["Lang",     "Chain"],
        ["Dolly",    "Too"],
        ["Elle",     "Elem"],
        ["Geoff",    "Fusion"],
        ["Trance",   "Former"],
        ["Jen",      "Ayai"],
    ]
    prompt = (
        f"Sort these customers by last name and then first name "
        f"and print the output: {customer_list}"
    )
    result = executor.invoke({"input": prompt})
    print("\n>>> Final output:", result.get("output"))


# ── Demo D: Custom date tool ──────────────────────────────────────────────────

def demo_custom_date_tool(executor: AgentExecutor) -> None:
    """
    Shows how a custom @tool decorator integrates into the ReAct loop.
    The agent calls today_date() to answer "What's today's date?".
    """
    print("\n" + "=" * 60)
    print("DEMO D — Custom tool: 'What's today's date?'")
    print("=" * 60)
    result = executor.invoke({"input": "What's today's date?"})
    print("\n>>> Final output:", result.get("output"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(textwrap.dedent("""\
        ╔══════════════════════════════════════════════════════════╗
        ║   LangChain: Agents  –  AWS Bedrock Edition              ║
        ╚══════════════════════════════════════════════════════════╝
    """))

    # Initialise LLM
    try:
        llm = _make_llm()
    except Exception as exc:
        print(f"[ERROR] Could not initialise Bedrock LLM: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Build tool sets ───────────────────────────────────────────────────────
    llm_math     = _make_llm_math_tool(llm)
    wikipedia    = _make_wikipedia_tool()
    python_repl  = _make_python_repl_tool()

    general_tools = [llm_math, wikipedia, today_date]
    repl_tools    = [python_repl]

    # ── Agents (two: one for general Q&A, one for code tasks) ────────────────
    print("Building agents …")
    general_agent = _build_agent(llm, general_tools)
    repl_agent    = _build_agent(llm, repl_tools)
    print("Agents ready.\n")

    # ── Run demos ─────────────────────────────────────────────────────────────
    try:
        demo_math(general_agent)
    except Exception as exc:
        print(f"  [SKIP] Math demo failed: {exc}")

    try:
        demo_wikipedia(general_agent)
    except Exception as exc:
        print(f"  [SKIP] Wikipedia demo failed: {exc}")

    try:
        demo_python_repl(repl_agent)
    except Exception as exc:
        print(f"  [SKIP] Python REPL demo failed: {exc}")

    try:
        demo_custom_date_tool(general_agent)
    except Exception as exc:
        print(f"  [SKIP] Date demo failed: {exc}")

    print("\n" + "=" * 60)
    print("All demos complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
