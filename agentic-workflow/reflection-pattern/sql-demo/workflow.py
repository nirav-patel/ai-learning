"""
Core SQL reflection workflow.

Mirrors the reference lab (agent_sql / M2_UGL_2.md) but adapted for a
Python-script context using AWS Bedrock instead of aisuite/OpenAI.

Workflow stages
---------------
1. Generate V1   — fast LLM converts a natural-language question to SQL.
2. Execute V1    — run the query and capture the results as a DataFrame.
3. Reflect + V2  — robust LLM evaluates the *actual output* and proposes
                   a refined SQL that truly answers the question.
4. Execute V2    — run the refined query for the final answer.
"""

import json
from dataclasses import dataclass, field

import pandas as pd

import bedrock_client
import utils


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WorkflowResult:
    question:  str
    schema:    str    = ""
    sql_v1:    str    = ""
    sql_v2:    str    = ""
    feedback:  str    = ""
    df_v1:     pd.DataFrame = field(default_factory=pd.DataFrame)
    df_v2:     pd.DataFrame = field(default_factory=pd.DataFrame)
    v1_has_error: bool = False
    v2_has_error: bool = False


# ---------------------------------------------------------------------------
# Stage 1 — Generate initial SQL (V1)
# ---------------------------------------------------------------------------

def generate_sql(question: str, schema: str, model: str) -> str:
    """
    Ask an LLM to translate a natural-language question into a SQLite query.
    Returns the raw SQL string.
    """
    prompt = f"""
You are a SQL assistant. Given the schema and the user's question, write a
SQL query for SQLite.

Schema:
{schema}

User question:
{question}

Respond with the SQL only — no markdown fences, no explanation.
"""
    return bedrock_client.generate_text(model, prompt, max_tokens=512, temperature=0.0)


# ---------------------------------------------------------------------------
# Stage 3 — Reflect on V1 output and produce refined SQL (V2)
# ---------------------------------------------------------------------------

def refine_sql_with_feedback(
    question:   str,
    sql_query:  str,
    df_result:  pd.DataFrame,
    schema:     str,
    model:      str,
) -> tuple[str, str]:
    """
    Evaluate whether the SQL output actually answers the question.
    Uses the *real execution result* as external feedback, so the model can
    detect semantic issues (e.g. negative totals from signed qty_delta) that
    are invisible from query text alone.

    Returns (feedback, refined_sql).
    """
    prompt = f"""
You are a SQL reviewer and refiner.

User asked:
{question}

Original SQL:
{sql_query}

Actual SQL output (first rows):
{df_result.to_markdown(index=False)}

Table Schema:
{schema}

Step 1: Briefly evaluate whether the SQL output fully and correctly answers
        the user's question. Look for semantic issues such as negative totals,
        wrong aggregation logic, missing filters, or incorrect grouping.

Step 2: If improvement is needed, provide a refined SQL query for SQLite.
        If the original SQL is already correct, return it unchanged.

Return STRICT JSON with exactly two fields:
{{
  "feedback": "<1-3 sentences explaining the gap or confirming correctness>",
  "refined_sql": "<final SQL to run — no markdown fences>"
}}
"""
    content = bedrock_client.generate_text(model, prompt, max_tokens=1024, temperature=0.0)

    # Strip markdown JSON fences if the model wraps its response
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[-1] if cleaned.count("```") >= 2 else cleaned
        cleaned = cleaned.strip().removeprefix("json").strip()

    try:
        obj = json.loads(cleaned)
        feedback    = str(obj.get("feedback", "")).strip()
        refined_sql = str(obj.get("refined_sql", sql_query)).strip()
        if not refined_sql:
            refined_sql = sql_query
    except Exception:
        # Fallback: keep original SQL, surface raw model output as feedback
        feedback    = content.strip()
        refined_sql = sql_query

    return feedback, refined_sql


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_sql_workflow(
    db_path:           str,
    question:          str,
    model_generation:  str,
    model_evaluation:  str,
) -> WorkflowResult:
    """
    End-to-end SQL reflection pipeline.

    Steps:
      1  Extract schema
      2  Generate SQL (V1)         — model_generation
      3  Execute V1 → DataFrame
      4  Reflect on V1 output      — model_evaluation
      5  Execute V2 → final answer
    """
    result = WorkflowResult(question=question)

    # ── Step 1: Schema ────────────────────────────────────────────────────
    print("Step 1: Extracting database schema… 📋")
    schema = utils.get_schema(db_path)
    result.schema = schema
    print(schema)

    # ── Step 2: Generate V1 ───────────────────────────────────────────────
    print("\nStep 2: Generating SQL (V1)… 🧠")
    sql_v1 = generate_sql(question, schema, model_generation)
    result.sql_v1 = sql_v1
    print(f"\n  SQL V1:\n  {sql_v1}")

    # ── Step 3: Execute V1 ────────────────────────────────────────────────
    print("\nStep 3: Executing SQL V1… 💻")
    df_v1 = utils.execute_sql(sql_v1, db_path)
    result.df_v1 = df_v1
    result.v1_has_error = "error" in df_v1.columns
    if result.v1_has_error:
        print(f"  ⚠ V1 execution error: {df_v1['error'].iloc[0]}")
    else:
        print(f"\n  V1 output:\n{df_v1.to_string(index=False)}")

    # ── Step 4: Reflect on V1 → V2 ───────────────────────────────────────
    print("\nStep 4: Reflecting on V1 output and refining SQL… 🔁")
    feedback, sql_v2 = refine_sql_with_feedback(
        question=question,
        sql_query=sql_v1,
        df_result=df_v1,
        schema=schema,
        model=model_evaluation,
    )
    result.feedback = feedback
    result.sql_v2   = sql_v2
    print(f"\n  Feedback: {feedback}")
    print(f"\n  SQL V2:\n  {sql_v2}")

    # ── Step 5: Execute V2 ────────────────────────────────────────────────
    print("\nStep 5: Executing refined SQL V2… ✅")
    df_v2 = utils.execute_sql(sql_v2, db_path)
    result.df_v2 = df_v2
    result.v2_has_error = "error" in df_v2.columns
    if result.v2_has_error:
        print(f"  ⚠ V2 execution error: {df_v2['error'].iloc[0]}")
    else:
        print(f"\n  V2 output (final answer):\n{df_v2.to_string(index=False)}")

    print("\nWorkflow complete.")
    return result
