"""
Reflection Pattern Demo — SQL Query Improvement.

Demonstrates the reflection design pattern applied to SQL generation:
  - A fast/light LLM (Amazon Nova Lite) generates an initial SQL query.
  - A robust LLM (Claude Sonnet) evaluates the *actual query output* and
    produces a refined SQL that correctly answers the question.

Usage:
    python main.py
"""

from workflow import run_sql_workflow
import utils

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Fast, lightweight model for initial SQL generation
GENERATION_MODEL = "us.amazon.nova-2-lite-v1:0"

# Robust reasoning model for reflection and query refinement
REFLECTION_MODEL = "us.anthropic.claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH = "products.db"

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create (or recreate) the synthetic transactions database
    utils.create_transactions_db(db_name=DB_PATH)

    question = "Which color of product has the highest total sales revenue?"

    print("=" * 60)
    print("  Reflection Pattern Demo — SQL Query Improvement")
    print("=" * 60)
    print(f"\nQuestion         : {question}")
    print(f"Generation model : {GENERATION_MODEL}")
    print(f"Reflection model : {REFLECTION_MODEL}")
    print()

    result = run_sql_workflow(
        db_path=DB_PATH,
        question=question,
        model_generation=GENERATION_MODEL,
        model_evaluation=REFLECTION_MODEL,
    )

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)

    print(f"\nQuestion : {result.question}")
    print(f"\nSQL V1   :\n  {result.sql_v1}")

    if result.v1_has_error:
        print(f"\n⚠ V1 error: {result.df_v1['error'].iloc[0]}")
    else:
        print(f"\nV1 output:\n{result.df_v1.to_string(index=False)}")

    print(f"\nReflection feedback:\n  {result.feedback}")
    print(f"\nSQL V2   :\n  {result.sql_v2}")

    if result.v2_has_error:
        print(f"\n⚠ V2 error: {result.df_v2['error'].iloc[0]}")
    else:
        print(f"\nV2 output (final answer):\n{result.df_v2.to_string(index=False)}")

    print()
