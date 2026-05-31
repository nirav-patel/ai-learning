"""
Core essay reflection workflow.

Implements the three-step reflective writing pattern from the reference lab
(C1M2_Assignment.md), adapted for AWS Bedrock instead of aisuite/OpenAI.

Workflow stages
---------------
1. Draft   — a fast LLM writes an initial essay on the given topic.
2. Reflect — a reasoning-capable LLM critiques the draft (structure,
             clarity, argument strength, writing style).
3. Revise  — the first LLM rewrites the essay incorporating the feedback.
"""

from dataclasses import dataclass

import bedrock_client


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
