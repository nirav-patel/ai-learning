"""
Workflow functions for the research agent.

Contains two independent pipelines:

Essay reflection pipeline (original)
-------------------------------------
1. Draft   — a fast LLM writes an initial essay on the given topic.
2. Reflect — a reasoning-capable LLM critiques the draft (structure,
             clarity, argument strength, writing style).
3. Revise  — the first LLM rewrites the essay incorporating the feedback.

Research pipeline (new — tool-calling)
--------------------------------------
1. Search  — LLM calls arXiv + Tavily tools to gather sources.
2. Reflect — LLM produces a structured JSON critique + revised report.
3. Format  — LLM converts the revised report to styled HTML.
"""

import ast
import json
import re
from dataclasses import dataclass, field

import bedrock_client
import eval as research_eval
import research_tools


DEFAULT_GENERATION_MODEL = "us.anthropic.claude-sonnet-4-6"
DEFAULT_REFLECTION_MODEL = "us.anthropic.claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WorkflowResult:
    topic:    str
    draft:    str = ""
    feedback: str = ""
    revised:  str = ""


# ---------------------------------------------------------------------------
# Stage 1 — Generate initial essay draft
# ---------------------------------------------------------------------------

def generate_draft(topic: str, model: str) -> str:
    """
    Ask the LLM to write a complete essay draft on the given topic.
    Returns the raw essay text.
    """
    prompt = f"""
You are an expert essay writer.

Write a well-structured, thoughtful essay on the following topic.
Include an introduction, at least two body paragraphs with supporting
arguments, and a conclusion.

Topic: {topic}
"""
    return bedrock_client.generate_text(model, prompt, max_tokens=1500, temperature=0.7)


# ---------------------------------------------------------------------------
# Stage 2 — Reflect on the draft
# ---------------------------------------------------------------------------

def reflect_on_draft(draft: str, model: str) -> str:
    """
    Ask the LLM to critically evaluate the essay draft.

    The feedback covers structure, clarity, argument strength, and writing
    style. It does NOT rewrite the essay — critique only.
    Returns the feedback as plain text.
    """
    prompt = f"""
You are a critical writing coach reviewing an essay draft.

Provide constructive feedback on the essay below. Your critique should
address each of the following dimensions:

1. **Structure** — Is the essay logically organised? Are transitions clear?
2. **Clarity**   — Is the language precise and easy to follow?
3. **Argument strength** — Are claims well-supported with evidence or reasoning?
4. **Writing style** — Is the tone appropriate and the prose engaging?

Be specific: point to concrete passages that could be improved and explain
why. Do NOT rewrite the essay — only analyse and advise.

Essay:
{draft}
"""
    return bedrock_client.generate_text(model, prompt, max_tokens=1000, temperature=0.2)


# ---------------------------------------------------------------------------
# Stage 3 — Revise the draft using reflection feedback
# ---------------------------------------------------------------------------

def revise_draft(original_draft: str, reflection: str, model: str) -> str:
    """
    Rewrite the essay, applying all improvements suggested in the reflection.

    Returns only the final revised essay text (no meta-commentary).
    """
    prompt = f"""
You are an expert essay editor.

Below is an original essay draft followed by detailed feedback from a
writing coach. Rewrite the essay so that it addresses every point raised
in the feedback. Improve clarity, coherence, argument strength, and overall
flow. Return ONLY the revised essay — no preamble, no commentary.

--- Original Draft ---
{original_draft}

--- Feedback ---
{reflection}
"""
    return bedrock_client.generate_text(model, prompt, max_tokens=1500, temperature=0.4)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_essay_workflow(
    topic:            str,
    model_generation: str,
    model_reflection: str,
) -> WorkflowResult:
    """
    End-to-end reflective essay pipeline.

    Steps:
      1  Generate draft        — model_generation
      2  Reflect on draft      — model_reflection
      3  Revise with feedback  — model_generation
    """
    result = WorkflowResult(topic=topic)

    # ── Step 1: Draft ─────────────────────────────────────────────────────
    print("Step 1: Generating essay draft… 📝")
    result.draft = generate_draft(topic, model_generation)
    print(f"\n{result.draft}\n")

    # ── Step 2: Reflect ───────────────────────────────────────────────────
    print("Step 2: Reflecting on draft… 🧠")
    result.feedback = reflect_on_draft(result.draft, model_reflection)
    print(f"\n{result.feedback}\n")

    # ── Step 3: Revise ────────────────────────────────────────────────────
    print("Step 3: Revising essay with feedback… ✍️")
    result.revised = revise_draft(result.draft, result.feedback, model_generation)
    print(f"\n{result.revised}\n")

    print("Workflow complete.")
    return result


# ===========================================================================
# Research pipeline — tool-calling workflow
# ===========================================================================

@dataclass
class ResearchResult:
    topic:          str
    report:         str = ""
    reflection:     str = ""
    revised_report: str = ""
    html:           str = ""
    messages:       list = field(default_factory=list)
    eval_report:    str = ""   # component-level source-quality evaluation
    plan_steps:     list[str] = field(default_factory=list)
    execution_log:  list[dict] = field(default_factory=list)
    remediated:     bool = False


@dataclass
class AgentExecutionState:
    """Shared state passed between planner/executor agent steps."""

    topic: str
    plan_steps: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    current_report: str = ""
    reflection: str = ""
    revised_report: str = ""
    html: str = ""
    eval_passed: bool = False
    eval_report: str = ""
    remediated: bool = False


def _default_plan(topic: str) -> list[str]:
    """Fallback plan used if planner output cannot be parsed."""
    return [
        f"Research authoritative sources on: {topic} and produce a cited draft report.",
        "Draft a coherent markdown report using the gathered evidence and links.",
        "Review the draft critically and improve clarity, structure, and citation quality.",
        "Generate the final research report as markdown.",
    ]


def planner_agent(topic: str, model: str = DEFAULT_REFLECTION_MODEL) -> list[str]:
    """Create an executable plan as a Python list of step strings."""
    user_prompt = f"""You are a planning agent for a multi-agent research system.

Return ONLY a valid Python list of strings.
Each item must be one executable step.
Available capabilities:
- research agent (tool-based retrieval and synthesis)
- writer agent (drafting/structuring)
- editor agent (reflection and revision)

Topic: {topic}
"""
    raw = bedrock_client.generate_text_with_system(
        model_id=model,
        system_prompt="You generate concise execution plans.",
        user_prompt=user_prompt,
        max_tokens=1200,
        temperature=0.2,
    )

    try:
        parsed = ast.literal_eval(raw.strip())
        if isinstance(parsed, list) and len(parsed) >= 3 and all(isinstance(s, str) for s in parsed):
            return parsed
    except (ValueError, SyntaxError):
        pass

    return _default_plan(topic)


def route_step_to_agent(
    step: str,
    model: str = DEFAULT_REFLECTION_MODEL,
) -> tuple[str, str]:
    """Choose the best agent for a step and return (agent_name, normalized_task)."""
    routing_prompt = f"""Choose one agent for the instruction below.

Return ONLY JSON with this schema:
{{"agent": "research_agent|writer_agent|editor_agent", "task": "normalized instruction"}}

Instruction: {step}
"""
    raw = bedrock_client.generate_text_with_system(
        model_id=model,
        system_prompt="You are an execution router for a research workflow.",
        user_prompt=routing_prompt,
        max_tokens=300,
        temperature=0,
    ).strip()

    try:
        data = json.loads(raw)
        agent = str(data.get("agent", "")).strip()
        task = str(data.get("task", step)).strip() or step
        if agent in {"research_agent", "writer_agent", "editor_agent"}:
            return agent, task
    except json.JSONDecodeError:
        pass

    # Heuristic fallback routing if model output is malformed.
    low = step.lower()
    if any(k in low for k in ["search", "research", "sources", "citations", "evidence"]):
        return "research_agent", step
    if any(k in low for k in ["edit", "critique", "revise", "improve", "review"]):
        return "editor_agent", step
    return "writer_agent", step


def _evaluate_source_quality(report_text: str, messages: list[dict]) -> tuple[bool, str]:
    """Evaluate source quality, preferring tool-level evidence when available."""
    tool_flag, tool_eval = research_eval.evaluate_tool_call_sources(messages)
    report_flag, report_eval = research_eval.evaluate_report_sources(report_text)

    if "No tavily_search_tool results" not in tool_eval:
        return tool_flag, tool_eval
    return report_flag, report_eval


def research_agent(task: str, state: AgentExecutionState, model: str) -> str:
    """Run tool-based retrieval/synthesis with prior-step context."""
    prior_context = "\n\n".join(
        f"Step {i + 1} ({h['agent']}):\n{h['output'][:800]}"
        for i, h in enumerate(state.history[-3:])
    )
    prompt = f"""Task: {task}

Topic: {state.topic}

Prior context (may be empty):
{prior_context}

Requirements:
- Use tools as needed.
- Prioritize authoritative sources and include URLs.
- Produce a comprehensive report with citations.
"""
    report, messages = generate_research_report_with_tools(prompt=prompt, model=model, max_turns=10)
    state.current_report = report
    state.messages = messages
    return report


def writer_agent(task: str, state: AgentExecutionState, model: str) -> str:
    """Draft or restructure report content using existing research context."""
    base_text = state.revised_report or state.current_report
    if not base_text:
        base_text = "No report exists yet. Create a first complete draft from the task."

    user_prompt = f"""You are writing the next version of a research report.

Task: {task}
Topic: {state.topic}

Current draft:
{base_text}

Return the full updated markdown report only.
"""
    out = bedrock_client.generate_text_with_system(
        model_id=model,
        system_prompt="You are a technical research writer.",
        user_prompt=user_prompt,
        max_tokens=4000,
        temperature=0.5,
    )
    state.current_report = out.strip()
    return state.current_report


def editor_agent(task: str, state: AgentExecutionState, model: str) -> str:
    """Critique and revise the current report using the existing editor routine."""
    base_report = state.current_report or state.revised_report
    if not base_report:
        base_report = f"Topic: {state.topic}\n\nTask: {task}"

    out = reflection_and_rewrite(base_report, model=model)
    state.reflection = out["reflection"]
    state.revised_report = out["revised_report"]
    state.current_report = state.revised_report
    return state.revised_report


def run_autonomous_research_pipeline(
    topic: str,
    generation_model: str = DEFAULT_GENERATION_MODEL,
    reflection_model: str = DEFAULT_REFLECTION_MODEL,
    limit_steps: bool = True,
    max_steps: int = 4,
) -> ResearchResult:
    """Planner + executor workflow with dynamic routing and eval-gated remediation."""
    state = AgentExecutionState(topic=topic)
    plan_steps = planner_agent(topic, model=reflection_model)
    if limit_steps:
        plan_steps = plan_steps[: min(len(plan_steps), max_steps)]
    state.plan_steps = plan_steps

    print("Step 0: Planning autonomous workflow… 🗺️")
    for idx, step in enumerate(plan_steps, start=1):
        print(f"  {idx}. {step}")

    for idx, step in enumerate(plan_steps, start=1):
        agent_name, task = route_step_to_agent(step, model=reflection_model)
        print(f"\nStep {idx}: {agent_name} executing → {task}")

        if agent_name == "research_agent":
            output = research_agent(task, state, model=generation_model)
        elif agent_name == "editor_agent":
            output = editor_agent(task, state, model=reflection_model)
        else:
            output = writer_agent(task, state, model=generation_model)

        state.history.append({"step": step, "agent": agent_name, "task": task, "output": output})

    # Quality gate and autonomous remediation if quality is below threshold.
    eval_flag, eval_report = _evaluate_source_quality(state.current_report, state.messages)
    state.eval_passed = eval_flag
    state.eval_report = eval_report

    if not state.eval_passed:
        print("\nSource quality failed threshold; running autonomous remediation…")
        remediation_task = (
            "Re-run research prioritizing authoritative domains (arxiv.org, nature.com, "
            "science.org, nasa.gov, major universities), include explicit URLs, and improve citation quality."
        )
        research_agent(remediation_task, state, model=generation_model)
        editor_agent("Revise report after remediation and strengthen evidence quality.", state, model=reflection_model)
        state.remediated = True
        state.eval_passed, state.eval_report = _evaluate_source_quality(state.current_report, state.messages)

    final_report = state.revised_report or state.current_report
    final_html = convert_report_to_html(final_report, model=generation_model)

    result = ResearchResult(
        topic=topic,
        report=state.current_report,
        reflection=state.reflection,
        revised_report=final_report,
        html=final_html,
        messages=state.messages,
        eval_report=state.eval_report,
        plan_steps=state.plan_steps,
        execution_log=state.history,
        remediated=state.remediated,
    )

    print("\nAutonomous research pipeline complete. ✅")
    return result


# ---------------------------------------------------------------------------
# Step 1 — Generate research report using external tools
# ---------------------------------------------------------------------------

def generate_research_report_with_tools(
    prompt: str,
    model: str = DEFAULT_GENERATION_MODEL,
    max_turns: int = 10,
) -> tuple[str, list[dict]]:
    """
    Generate a detailed research report by letting the LLM call arXiv and
    Tavily search tools to gather sources before writing.

    Parameters
    ----------
    prompt    : Research topic or question.
    model     : Bedrock model ID to use.
    max_turns : Maximum number of tool-calling iterations.

    Returns
    -------
    tuple[str, list[dict]]
        final_text — Final research report text.
        messages   — Full conversation history (for downstream evaluation).
    """
    system_prompt = (
        "You are a research assistant that can search the web and arXiv to write "
        "detailed, accurate, and properly sourced research reports.\n\n"
        "🔍 Use the available tools when appropriate (e.g., to find scientific papers "
        "or current web content).\n"
        "📚 Cite sources whenever relevant. Do NOT omit citations for brevity.\n"
        "🌐 When possible, include full URLs (arXiv links, web sources, etc.).\n"
        "✍️  Use an academic tone, organise output into clearly labelled sections, "
        "and include inline citations or footnotes as needed.\n"
        "🚫 Do not include placeholder text such as '(citation needed)' or "
        "'(citations omitted)'."
    )

    tools = [
        research_tools.ARXIV_BEDROCK_TOOL_DEF,
        research_tools.TAVILY_BEDROCK_TOOL_DEF,
        research_tools.WIKIPEDIA_BEDROCK_TOOL_DEF,
    ]

    final_text, messages = bedrock_client.run_tool_loop(
        model_id=model,
        system_prompt=system_prompt,
        initial_prompt=prompt,
        tools=tools,
        tool_mapping=research_tools.TOOL_MAPPING,
        max_turns=max_turns,
        max_tokens=4000,
        temperature=0.7,
    )

    return final_text, messages


# ---------------------------------------------------------------------------
# Step 2 — Structured reflection + rewrite
# ---------------------------------------------------------------------------

def reflection_and_rewrite(
    report,
    model: str = DEFAULT_REFLECTION_MODEL,
    temperature: float = 0.3,
) -> dict:
    """
    Produce a structured critique and an improved version of the report.

    Accepts raw text OR the messages list returned by
    ``generate_research_report_with_tools``.

    Uses two separate Bedrock calls to avoid token budget issues and fragile
    JSON-inside-JSON escaping:
      1. Generate reflection text only.
      2. Generate revised report using the original + reflection.

    Returns
    -------
    dict with keys:
        "reflection"    — structured critique covering Strengths, Limitations,
                          Suggestions, and Opportunities.
        "revised_report"— improved version of the input report.
    """
    report_text = research_tools.parse_input(report)

    # ── Call 1: reflection only ───────────────────────────────────────────
    print("  Step 2a: Generating reflection… 🔎")
    reflection_prompt = f"""You are an academic reviewer.

Analyse the research report below and write a structured critique.

Use exactly these four headings (in this order):
- Strengths
- Limitations
- Suggestions
- Opportunities

Be specific: point to concrete passages and explain how to improve them.
Do NOT rewrite the report — critique only.

Research report:
{report_text}
"""
    reflection = bedrock_client.generate_text_with_system(
        model_id=model,
        system_prompt="You are an academic reviewer and editor.",
        user_prompt=reflection_prompt,
        max_tokens=2000,
        temperature=temperature,
    )

    # ── Call 2: revised report using the reflection ───────────────────────
    print("  Step 2b: Rewriting report with feedback… ✍️")
    rewrite_prompt = f"""You are an expert research editor.

Below is an original research report followed by a structured critique.
Rewrite the report so that it addresses every point raised in the critique.
Improve clarity, depth, and academic tone. Keep all citations and URLs.
Return ONLY the full revised report — no preamble, no commentary.

--- Original Report ---
{report_text}

--- Critique ---
{reflection}
"""
    revised_report = bedrock_client.generate_text_with_system(
        model_id=model,
        system_prompt="You are an expert research editor.",
        user_prompt=rewrite_prompt,
        max_tokens=4000,
        temperature=temperature,
    )

    return {
        "reflection": reflection.strip(),
        "revised_report": revised_report.strip(),
    }


# ---------------------------------------------------------------------------
# Step 3 — Convert report to styled HTML
# ---------------------------------------------------------------------------

def convert_report_to_html(
    report,
    model: str = DEFAULT_GENERATION_MODEL,
    temperature: float = 0.5,
) -> str:
    """
    Convert a plaintext research report into a well-structured, styled HTML page.

    Accepts raw text OR the messages list from the tool-calling step.

    Returns
    -------
    str : A complete HTML document string.
    """
    report_text = research_tools.parse_input(report)

    system_prompt = "You convert plaintext research reports into full clean HTML documents."

    user_prompt = f"""Convert the research report below into a complete, well-structured HTML document.

Requirements:
- Return ONLY valid HTML — no markdown, no code fences, no commentary.
- Include a proper <html>, <head> (with <meta charset="UTF-8"> and a <style> block), and <body>.
- Use semantic tags: <h1> for the title, <h2> for section headers, <p> for paragraphs.
- All URLs must be wrapped in <a href="..."> tags so they are clickable.
- Preserve all citations and references from the original report.
- Add a clean, readable CSS style (e.g., max-width, line-height, font-family, link colours).
- Do NOT truncate or omit any content from the report.

Research report:
{report_text}
"""

    html = bedrock_client.generate_text_with_system(
        model_id=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=6000,
        temperature=temperature,
    )

    # Strip any accidental markdown code fences
    html = re.sub(r"^```(?:html)?\s*", "", html.strip(), flags=re.MULTILINE)
    html = re.sub(r"\s*```$", "", html.strip(), flags=re.MULTILINE)

    return html.strip()


# ---------------------------------------------------------------------------
# Orchestrator — full research pipeline
# ---------------------------------------------------------------------------

def run_research_pipeline(
    topic: str,
    generation_model: str = DEFAULT_GENERATION_MODEL,
    reflection_model: str = DEFAULT_REFLECTION_MODEL,
    autonomous: bool = False,
) -> ResearchResult:
    """
    End-to-end research pipeline.

    Steps:
      1  Search + generate report   — generation_model (with tools)
      2  Reflect + rewrite          — reflection_model
      3  Convert to HTML            — generation_model
    """
    if autonomous:
        return run_autonomous_research_pipeline(
            topic=topic,
            generation_model=generation_model,
            reflection_model=reflection_model,
        )

    result = ResearchResult(topic=topic)

    # ── Step 1: Research with tools ───────────────────────────────────────
    print("Step 1: Generating research report with tools… 🔍")
    result.report, result.messages = generate_research_report_with_tools(
        topic, generation_model
    )

    # ── Component-level eval: source quality ──────────────────────────────
    print("  📊 Evaluating source quality…")
    # Evaluate against raw Tavily tool results (more precise)
    tool_flag, tool_eval = research_eval.evaluate_tool_call_sources(result.messages)
    # Also evaluate cited URLs in the final report text
    report_flag, report_eval = research_eval.evaluate_report_sources(result.report)

    # Prefer tool-call eval if Tavily results were found; fall back to report eval
    if "No tavily_search_tool results" not in tool_eval:
        result.eval_report = tool_eval
        eval_flag = tool_flag
    else:
        result.eval_report = report_eval
        eval_flag = report_flag

    status = "✅ PASS" if eval_flag else "❌ FAIL"
    print(f"  Source quality eval: {status}")
    print(result.eval_report)

    # ── Step 2: Reflect + rewrite ─────────────────────────────────────────
    print("Step 2: Reflecting on report… 🧠")
    reflection_output = reflection_and_rewrite(result.report, reflection_model)
    result.reflection = reflection_output["reflection"]
    result.revised_report = reflection_output["revised_report"]
    print(f"\nReflection:\n{result.reflection[:300]}…\n")

    # ── Step 3: Convert to HTML ───────────────────────────────────────────
    print("Step 3: Converting report to HTML… 🌐")
    result.html = convert_report_to_html(result.revised_report, generation_model)
    print(f"\nHTML preview (first 300 chars):\n{result.html[:300]}…\n")

    print("Research pipeline complete. ✅")
    return result
