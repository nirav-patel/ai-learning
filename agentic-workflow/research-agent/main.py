"""
Research Agent — Entry Point

Runs two independent workflows:

1. Essay Reflection Pipeline (original)
   - A fast LLM (Amazon Nova Lite) writes an initial essay draft.
   - A reasoning LLM (Claude Sonnet) critiques the draft for structure,
     clarity, argument strength, and writing style.
   - The first LLM rewrites the essay incorporating all feedback.

2. Research Pipeline with Tools (new)
   - LLM calls arXiv and Tavily tools to gather sources.
   - LLM produces a structured JSON critique and revised report.
   - LLM converts the revised report to styled HTML.

Usage:
    python main.py                  # runs both pipelines
    python main.py --research-only  # skip essay, run research pipeline only
    python main.py --essay-only     # run essay pipeline only
"""

import sys

from workflow import run_essay_workflow, run_research_pipeline

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Fast model for draft generation and revision
GENERATION_MODEL = "us.anthropic.claude-sonnet-4-6" #"us.amazon.nova-2-lite-v1:0"

# Reasoning model for critique — benefits from stronger analytical capability
REFLECTION_MODEL = "us.anthropic.claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    run_essay    = "--research-only" not in args
    run_research = "--essay-only"    not in args

    # ──────────────────────────────────────────────────────────────────────
    # Pipeline 1 — Essay reflection (original)
    # ──────────────────────────────────────────────────────────────────────
    if run_essay:
        topic = "Should social media platforms be regulated by the government?"

        print("=" * 60)
        print("  Pipeline 1 — Essay Reflection")
        print("=" * 60)
        print(f"\nTopic            : {topic}")
        print(f"Generation model : {GENERATION_MODEL}")
        print(f"Reflection model : {REFLECTION_MODEL}")
        print()

        essay_result = run_essay_workflow(
            topic=topic,
            model_generation=GENERATION_MODEL,
            model_reflection=REFLECTION_MODEL,
        )

        sep = "-" * 60
        print("\n" + "=" * 60)
        print("  Essay Results Summary")
        print("=" * 60)
        print(f"\n📝 Step 1 — Draft:\n{sep}\n{essay_result.draft}")
        print(f"\n🧠 Step 2 — Reflection:\n{sep}\n{essay_result.feedback}")
        print(f"\n✍️  Step 3 — Revised:\n{sep}\n{essay_result.revised}")

        from tests import test_generate_draft, test_reflect_on_draft, test_revise_draft
        print("\n" + "=" * 60)
        print("  Essay Unit Tests")
        print("=" * 60 + "\n")
        print("─" * 50)
        print("Testing generate_draft:")
        test_generate_draft(essay_result.draft)
        print("─" * 50)
        print("Testing reflect_on_draft:")
        test_reflect_on_draft(essay_result.feedback)
        print("─" * 50)
        print("Testing revise_draft:")
        test_revise_draft(essay_result.revised)
        print()

    # ──────────────────────────────────────────────────────────────────────
    # Pipeline 2 — Research with tools
    # ──────────────────────────────────────────────────────────────────────
    if run_research:
        research_topic = "recent developments in black hole science"

        print("=" * 60)
        print("  Pipeline 2 — Research with Tools")
        print("=" * 60)
        print(f"\nTopic : {research_topic}")
        print(f"Model : {GENERATION_MODEL}")
        print()

        research_result = run_research_pipeline(
            topic=research_topic,
            generation_model=GENERATION_MODEL,
            reflection_model=REFLECTION_MODEL,
        )

        sep = "-" * 60
        print("\n" + "=" * 60)
        print("  Research Results Summary")
        print("=" * 60)
        print(f"\n📄 Report (first 500 chars):\n{sep}\n{research_result.report[:500]}…")
        print(f"\n📊 Source Quality Eval:\n{sep}\n{research_result.eval_report}")
        print(f"\n🧠 Reflection:\n{sep}\n{research_result.reflection[:500]}…")
        print(f"\n✍️  Revised Report (first 500 chars):\n{sep}\n{research_result.revised_report[:500]}…")
        print(f"\n🌐 HTML Preview (first 500 chars):\n{sep}\n{research_result.html[:500]}…")

        # Save HTML to file
        html_path = "research_report.html"
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(research_result.html)
        print(f"\n💾 Full HTML saved to: {html_path}")

        from tests import (
            test_generate_research_report_with_tools,
            test_reflection_and_rewrite,
            test_convert_report_to_html,
            test_evaluate_report_sources,
        )
        from eval import evaluate_report_sources
        print("\n" + "=" * 60)
        print("  Research Pipeline Unit Tests")
        print("=" * 60 + "\n")
        print("─" * 50)
        print("Testing generate_research_report_with_tools:")
        test_generate_research_report_with_tools(research_result.report)
        print("─" * 50)
        print("Testing reflection_and_rewrite:")
        test_reflection_and_rewrite(
            {"reflection": research_result.reflection,
             "revised_report": research_result.revised_report}
        )
        print("─" * 50)
        print("Testing convert_report_to_html:")
        test_convert_report_to_html(research_result.html)
        print("─" * 50)
        print("Testing evaluate_report_sources (component-level eval):")
        flag, report = evaluate_report_sources(research_result.report)
        test_evaluate_report_sources(flag, report)
        print()
