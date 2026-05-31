"""
Reflection Pattern Demo — entry point.

Runs a complete reflection workflow where:
  - A fast/light LLM (Amazon Nova Lite) generates matplotlib chart code.
  - A robust multimodal LLM (Claude Sonnet) critiques the chart image AND
    produces improved code in a single reflection call.

All generated artifacts (code + charts) are saved for side-by-side review.

Usage:
    python main.py
"""

from workflow import run_workflow

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Fast, lightweight model for code generation
GENERATION_MODEL = "us.amazon.nova-2-lite-v1:0"

# Robust multimodal model for visual reflection + code refinement
REFLECTION_MODEL = "us.anthropic.claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Run  (mirrors reference lab's run_workflow signature)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    user_instructions = "Create a plot comparing Q1 coffee sales in 2024 and 2025 using the data in coffee_sales.csv."
    generation_model  = GENERATION_MODEL
    reflection_model  = REFLECTION_MODEL
    image_basename    = "chart"

    print("=" * 60)
    print("  Reflection Pattern Demo — Chart Visualization")
    print("=" * 60)
    print(f"\nInstruction      : {user_instructions}")
    print(f"Generation model : {generation_model}")
    print(f"Reflection model : {reflection_model}")
    print()

    result = run_workflow(
        dataset_path=     "coffee_sales.csv",
        user_instructions=user_instructions,
        generation_model= generation_model,
        reflection_model= reflection_model,
        image_basename=   image_basename,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)

    if result.feedback:
        print(f"\nReflection feedback:\n  {result.feedback}")

    print("\nArtifacts saved:")
    print(f"  Chart V1    : {result.chart_v1_path}")
    print(f"  Chart V2    : {result.chart_v2_path}")
    print(f"  Comparison  : {result.comparison_path}")
    print(f"  Code V1     : code_output/{image_basename}_v1.py")
    print(f"  Code V2     : code_output/{image_basename}_v2.py")

    if result.v1_exec_error:
        print(f"\n⚠ V1 execution error: {result.v1_exec_error}")
    if result.v2_exec_error:
        print(f"\n⚠ V2 execution error: {result.v2_exec_error}")

    print()
