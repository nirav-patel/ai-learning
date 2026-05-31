"""
Reflection Pattern Demo — Research Agent (Essay Improvement).

Demonstrates the three-step reflective writing workflow:
  - A fast LLM (Amazon Nova Lite) writes an initial essay draft.
  - A reasoning LLM (Claude Sonnet) critiques the draft for structure,
    clarity, argument strength, and writing style.
  - The first LLM rewrites the essay incorporating all feedback.

Usage:
    python main.py
"""

from workflow import run_essay_workflow

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Fast model for draft generation and revision
GENERATION_MODEL = "us.amazon.nova-2-lite-v1:0"

# Reasoning model for critique — benefits from stronger analytical capability
REFLECTION_MODEL = "us.anthropic.claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    topic = "Should social media platforms be regulated by the government?"

    print("=" * 60)
    print("  Reflection Pattern Demo — Research Agent")
    print("=" * 60)
    print(f"\nTopic            : {topic}")
    print(f"Generation model : {GENERATION_MODEL}")
    print(f"Reflection model : {REFLECTION_MODEL}")
    print()

    result = run_essay_workflow(
        topic=topic,
        model_generation=GENERATION_MODEL,
        model_reflection=REFLECTION_MODEL,
    )

    # ── Summary ──────────────────────────────────────────────────────────
    sep = "-" * 60

    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)

    print(f"\n📝 Step 1 — Draft:\n{sep}\n{result.draft}")
    print(f"\n🧠 Step 2 — Reflection:\n{sep}\n{result.feedback}")
    print(f"\n✍️  Step 3 — Revised:\n{sep}\n{result.revised}")

    # ── Unit tests ───────────────────────────────────────────────────────
    from tests import test_generate_draft, test_reflect_on_draft, test_revise_draft

    print("\n" + "=" * 60)
    print("  Unit Tests")
    print("=" * 60 + "\n")

    print("─" * 50)
    print("Testing generate_draft:")
    test_generate_draft(result.draft)

    print("─" * 50)
    print("Testing reflect_on_draft:")
    test_reflect_on_draft(result.feedback)

    print("─" * 50)
    print("Testing revise_draft:")
    test_revise_draft(result.revised)
    print()
