"""
Core reflection workflow for data visualization improvement.

Mirrors the structure from the reference lab (agent_chart_plot) but adapted
to use AWS Bedrock instead of OpenAI / Anthropic direct APIs.

Workflow stages
---------------
1. Generate V1   — fast LLM writes matplotlib code from a natural-language instruction.
2. Execute V1    — run the generated code to produce chart_v1.png.
3. Reflect + V2  — robust multimodal LLM critiques the chart AND returns improved code
                   in a single call (matching the reference lab pattern).
4. Execute V2    — run the improved code to produce chart_v2.png.
5. Compare       — stitch both charts into a side-by-side comparison image.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import bedrock_client
import utils


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WorkflowResult:
    instruction: str
    code_v1: str = ""
    code_v2: str = ""
    chart_v1_path: str = ""
    chart_v2_path: str = ""
    comparison_path: str = ""
    feedback: str = ""
    v1_exec_error: str = ""
    v2_exec_error: str = ""


# ---------------------------------------------------------------------------
# Step 1 — Generate initial chart code (V1)
# Prompt kept as close as possible to the reference lab.
# ---------------------------------------------------------------------------

def generate_chart_code(instruction: str, model: str, out_path_v1: str) -> str:
    """Generate Python code to make a plot with matplotlib using tag-based wrapping."""

    prompt = f"""
    You are a data visualization expert.

    Return your answer *strictly* in this format:

    <execute_python>
    # valid python code here
    </execute_python>

    Do not add explanations, only the tags and the code.

    The code should create a visualization from a DataFrame 'df' with these columns:
    - date   (datetime64 — already parsed; use df['date'].dt.year, df['date'].dt.month, etc.)
    - time   (string, HH:MM — do NOT concatenate or combine with the date column)
    - cash_type (string: 'card' or 'cash')
    - card (string)
    - price (number)
    - coffee_name (string)
    - quarter (int, 1–4 — already computed, use directly)
    - month  (int, 1–12 — already computed, use directly)
    - year   (int, e.g. 2024 — already computed, use directly)

    User instruction: {instruction}

    Requirements for the code:
    1. Assume the DataFrame is already loaded as 'df'.
    2. Use matplotlib for plotting.
    3. Add clear title, axis labels, and legend if needed.
    4. Save the figure as '{out_path_v1}' with dpi=300.
    5. Do not call plt.show().
    6. Close all plots with plt.close().
    7. Add all necessary import python statements
    8. CRITICAL: 'date' is datetime64 — never use string concatenation on it.
       Filter by year/quarter using the 'year' and 'quarter' integer columns.

    Return ONLY the code wrapped in <execute_python> tags.
    """

    return bedrock_client.generate_text(model, prompt, max_tokens=2000, temperature=0.0)


# ---------------------------------------------------------------------------
# Step 3 — Reflect on chart image and regenerate improved code (V2)
# Combined into one call, matching the reference lab's reflect_on_image_and_regenerate.
# ---------------------------------------------------------------------------

def reflect_on_image_and_regenerate(
    chart_path: str,
    instruction: str,
    model_name: str,
    out_path_v2: str,
    code_v1: str,
) -> tuple[str, str]:
    """
    Critique the chart IMAGE and the original code against the instruction,
    then return refined matplotlib code.

    Returns (feedback, refined_code_with_tags).
    """

    prompt = f"""
    You are a data visualization expert.
    Your task: critique the attached chart and the original code against the given instruction,
    then return improved matplotlib code.

    Original code (for context):
    {code_v1}

    OUTPUT FORMAT (STRICT):
    1) First line: a valid JSON object with ONLY the "feedback" field.
    Example: {{"feedback": "The legend is unclear and the axis labels overlap."}}

    2) After a newline, output ONLY the refined Python code wrapped in:
    <execute_python>
    ...
    </execute_python>

    3) Import all necessary libraries in the code. Don't assume any imports from the original code.

    HARD CONSTRAINTS:
    - Do NOT include Markdown, backticks, or any extra prose outside the two parts above.
    - Use pandas/matplotlib only (no seaborn).
    - Assume df already exists; do not read from files.
    - Save to '{out_path_v2}' with dpi=300.
    - Always call plt.close() at the end (no plt.show()).
    - Include all necessary import statements.

    IMPORTANT: The 'date' column is already a pandas datetime64 type.
    - Do NOT concatenate 'date' with 'time' using string operations.
    - To filter by year/quarter, use: df[df['year'] == 2024] or df['date'].dt.year == 2024
    - The 'quarter' and 'year' columns already exist as integers; use them directly.

    Schema (columns available in df):
    - date   (datetime64 — already parsed; use df['date'].dt.year, etc.)
    - time   (string, HH:MM — do NOT concatenate with date)
    - cash_type (string: 'card' or 'cash')
    - card   (string)
    - price  (float)
    - coffee_name (string)
    - quarter (int, 1–4)
    - month  (int, 1–12)
    - year   (int)

    CRITICAL TYPE RULE: 'date' is already datetime64.
    - NEVER do: df['date'] + ' ' + df['time']  ← this will crash
    - ALWAYS filter by year/quarter using the integer columns: df[df['year'] == 2024]

    Instruction:
    {instruction}
    """

    content = bedrock_client.generate_with_image(
        model_name, prompt, chart_path, max_tokens=2000, temperature=0.0
    )

    # --- Parse ONLY the first JSON line (feedback) ---
    lines = content.strip().splitlines()
    json_line = lines[0].strip() if lines else ""

    try:
        obj = json.loads(json_line)
    except Exception as e:
        # Fallback: find the first {...} block anywhere in the response
        m_json = re.search(r"\{.*?\}", content, flags=re.DOTALL)
        if m_json:
            try:
                obj = json.loads(m_json.group(0))
            except Exception as e2:
                obj = {"feedback": f"Failed to parse JSON: {e2}"}
        else:
            obj = {"feedback": f"Failed to find JSON: {e}"}

    # --- Extract refined code from <execute_python>...</execute_python> ---
    m_code = re.search(r"<execute_python>([\s\S]*?)</execute_python>", content)
    refined_code_body = m_code.group(1).strip() if m_code else ""
    refined_code = f"<execute_python>\n{refined_code_body}\n</execute_python>"

    feedback = str(obj.get("feedback", "")).strip()
    return feedback, refined_code


# ---------------------------------------------------------------------------
# Orchestrator — signature mirrors the reference lab's run_workflow
# ---------------------------------------------------------------------------

def run_workflow(
    dataset_path: str,
    user_instructions: str,
    generation_model: str,
    reflection_model: str,
    image_basename: str = "chart",
) -> WorkflowResult:
    """End-to-end reflection pipeline.

    Stages:
        1  Generate V1 code          (generation_model)
        2  Execute V1 → chart_v1.png
        3  Reflect + Generate V2     (reflection_model, multimodal — one call)
        4  Execute V2 → chart_v2.png
        5  Create side-by-side comparison image

    Artifacts saved:
        images/{image_basename}_v1.png
        images/{image_basename}_v2.png
        images/{image_basename}_comparison.png
        code_output/{image_basename}_v1.py
        code_output/{image_basename}_v2.py
    """

    result = WorkflowResult(instruction=user_instructions)

    Path("images").mkdir(exist_ok=True)
    Path("code_output").mkdir(exist_ok=True)

    out_v1 = f"images/{image_basename}_v1.png"
    out_v2 = f"images/{image_basename}_v2.png"
    out_comparison = f"images/{image_basename}_comparison.png"
    code_v1_file = f"code_output/{image_basename}_v1.py"
    code_v2_file = f"code_output/{image_basename}_v2.py"

    result.chart_v1_path = out_v1
    result.chart_v2_path = out_v2
    result.comparison_path = out_comparison

    df = utils.load_and_prepare_data(dataset_path)

    # ------------------------------------------------------------------
    # Stage 1 — Generate V1
    # ------------------------------------------------------------------
    print("Step 1: Generating chart code (V1)… 📈")
    code_v1_raw = generate_chart_code(
        instruction=user_instructions,
        model=generation_model,
        out_path_v1=out_v1,
    )
    code_v1 = utils.extract_python_code(code_v1_raw)
    if not code_v1:
        raise RuntimeError("Generator model returned no executable code.")
    result.code_v1 = code_v1
    utils.save_code(code_v1, code_v1_file)
    print(f"  Code saved → {code_v1_file}")

    # ------------------------------------------------------------------
    # Stage 2 — Execute V1
    # ------------------------------------------------------------------
    print("Step 2: Executing chart code (V1)… 💻")
    v1_ok, v1_err = utils.execute_chart_code(code_v1, df)
    if not v1_ok:
        result.v1_exec_error = v1_err
        print(f"  ⚠ V1 execution failed: {v1_err}")
    else:
        print(f"  Chart saved → {out_v1}")

    # ------------------------------------------------------------------
    # Stage 3 — Reflect on V1 (image + code) → feedback + V2 code
    # ------------------------------------------------------------------
    print("Step 3: Reflecting on V1 (image + code) and generating improvements… 🔁")
    if v1_ok:
        feedback, code_v2_raw = reflect_on_image_and_regenerate(
            chart_path=out_v1,
            instruction=user_instructions,
            model_name=reflection_model,
            out_path_v2=out_v2,
            code_v1=code_v1,
        )
        result.feedback = feedback
        print(f"  Reflection feedback: {feedback}")
    else:
        print("  Skipping reflection — V1 chart was not produced.")
        feedback, code_v2_raw = "", code_v1_raw

    code_v2 = utils.extract_python_code(code_v2_raw)
    if not code_v2:
        raise RuntimeError("Reflection model returned no executable code.")
    result.code_v2 = code_v2
    utils.save_code(code_v2, code_v2_file)
    print(f"  Code saved → {code_v2_file}")

    # ------------------------------------------------------------------
    # Stage 4 — Execute V2
    # ------------------------------------------------------------------
    print("Step 4: Executing refined chart code (V2)… 🖼️")
    v2_ok, v2_err = utils.execute_chart_code(code_v2, df)
    if not v2_ok:
        result.v2_exec_error = v2_err
        print(f"  ⚠ V2 execution failed: {v2_err}")
    else:
        print(f"  Chart saved → {out_v2}")

    # ------------------------------------------------------------------
    # Stage 5 — Side-by-side comparison
    # ------------------------------------------------------------------
    print("Step 5: Creating comparison image… 🖼️")
    if v1_ok and v2_ok:
        utils.create_comparison_image(out_v1, out_v2, out_comparison)
        print(f"  Comparison saved → {out_comparison}")
    else:
        print("  Skipping comparison — one or both charts are missing.")

    print("\nWorkflow complete.")
    return result
