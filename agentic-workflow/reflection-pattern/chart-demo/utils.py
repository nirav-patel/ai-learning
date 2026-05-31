"""
Utility helpers for the reflection pattern demo.

Covers:
  - Loading and preparing the coffee sales dataset
  - Extracting Python code from LLM responses
  - Executing generated chart code safely
  - Saving generated code artifacts for review
  - Creating a side-by-side image comparison
"""

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load coffee_sales.csv and derive common date columns used in charts."""
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["quarter"] = df["date"].dt.quarter
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
    return df


def get_schema_description(df: pd.DataFrame) -> str:
    """Return a human-readable column schema for prompt injection."""
    return "\n".join(f"  - {col}: {dtype}" for col, dtype in df.dtypes.items())


# ---------------------------------------------------------------------------
# Code extraction & execution
# ---------------------------------------------------------------------------

def extract_python_code(llm_response: str) -> str | None:
    """Return the Python code inside <execute_python>…</execute_python> tags.

    Returns None if no such block is found.
    """
    match = re.search(r"<execute_python>([\s\S]*?)</execute_python>", llm_response)
    if match:
        return match.group(1).strip()
    # Fallback: strip plain markdown fences if model forgot the tags
    fenced = re.search(r"```(?:python)?\s*([\s\S]*?)```", llm_response)
    if fenced:
        return fenced.group(1).strip()
    return None


def execute_chart_code(code: str, df: pd.DataFrame) -> tuple[bool, str]:
    """Execute chart-generating code in a sandboxed namespace.

    The DataFrame `df` is injected into the execution context so the generated
    code can reference it directly without reloading the CSV.

    Returns:
        (success, error_message) — error_message is empty on success.
    """
    exec_globals = {
        "df": df,
        "pd": pd,
        "plt": plt,
    }
    try:
        exec(code, exec_globals)  # noqa: S102
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_code(code: str, output_path: str) -> None:
    """Write generated Python code to a .py file for later review."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")


def create_comparison_image(img1_path: str, img2_path: str, output_path: str) -> None:
    """Stitch two chart images side-by-side and save the result."""
    left = Image.open(img1_path)
    right = Image.open(img2_path)

    # Normalise heights so both images share the same canvas
    target_height = max(left.height, right.height)
    left = _resize_to_height(left, target_height)
    right = _resize_to_height(right, target_height)

    canvas = Image.new("RGB", (left.width + right.width, target_height), color="white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resize_to_height(img: Image.Image, height: int) -> Image.Image:
    if img.height == height:
        return img
    ratio = height / img.height
    new_width = int(img.width * ratio)
    return img.resize((new_width, height), Image.LANCZOS)
