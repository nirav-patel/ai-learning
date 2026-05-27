"""
Product catalog helpers
=======================
Loads products_list.json and categories.json from the api-call-demo directory
and provides helper functions for looking up and formatting product data.

Key functions:
  get_product_by_name()      – return one product dict by exact name
  get_products_by_category() – return all products for a given category
  build_allowed_products_text() – build the prompt-injectable product listing
  build_category_names()     – comma-separated category names for prompts
  read_string_to_list()      – parse model output (Python-literal string) → list
  generate_output_string()   – expand a category/products list into full JSON product details
"""
from __future__ import annotations

import ast
import json
import os

# Resolve the directory that contains the JSON data files (api-call-demo/).
# This module lives in api-call-demo/utils/, so we go one level up.
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))

with open(os.path.join(_BASE_DIR, "products_list.json")) as f:
    products = json.load(f)

with open(os.path.join(_BASE_DIR, "categories.json")) as f:
    categories = json.load(f)


# ---------------------------------------------------------------------------
# Product lookups
# ---------------------------------------------------------------------------

def get_product_by_name(name: str) -> dict | None:
    """Return a single product dict by exact product name, or None if not found."""
    return products.get(name, None)


def get_products_by_category(category: str) -> list[dict]:
    """Return all products that belong to the given category."""
    return [p for p in products.values() if p["category"] == category]


# ---------------------------------------------------------------------------
# Prompt-building helpers (inject product/category context into prompts)
# ---------------------------------------------------------------------------

def build_allowed_products_text() -> str:
    """
    Build a human-readable product listing grouped by category.
    Used to inject into prompts so the model knows which products exist.

    Example output:
        Computers and Laptops category:
        TechPro Ultrabook
        BlueWave Gaming Laptop
        ...
    """
    lines = []
    for category, product_list in categories.items():
        lines.append(f"{category} category:")
        lines.extend(product_list)
        lines.append("")
    return "\n".join(lines).strip()


def build_category_names() -> str:
    """Return a comma-separated string of all category names (for injection into prompts)."""
    return ", ".join(categories.keys())


# ---------------------------------------------------------------------------
# Parsing helpers (parse model output back into Python structures)
# ---------------------------------------------------------------------------

def read_string_to_list(input_string: str) -> list | None:
    """
    Parse a Python-literal string (possibly wrapped in markdown code fences)
    into a Python list. Claude sometimes wraps JSON/Python in ```...``` blocks.

    Returns the parsed list, or None on parse failure.
    """
    if input_string is None:
        return None
    raw = input_string.strip()
    # Strip markdown code block if the model wraps output in ```python``` or ```json```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith(("python", "json")):
            raw = raw.split("\n", 1)[1]
        raw = raw.strip()
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError) as e:
        print(f"Error: Could not parse string to list: {e}")
        return None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def generate_output_string(data_list: list) -> str:
    """
    Given a list of {category: ...} or {products: [...]} dicts (as returned by
    find_category_and_product), look up each product and return a JSON string
    with full details of all relevant products.

    This bridges the extraction step and the response-generation step:
    the model extracts names, we look up rich details, then inject them into the prompt.
    """
    output_string = ""
    if data_list is None:
        return output_string
    for data in data_list:
        try:
            if "products" in data:
                for product_name in data["products"]:
                    product = get_product_by_name(product_name)
                    if product:
                        output_string += json.dumps(product, indent=4) + "\n"
                    else:
                        print(f"Error: Product '{product_name}' not found")
            elif "category" in data:
                for product in get_products_by_category(data["category"]):
                    output_string += json.dumps(product, indent=4) + "\n"
            else:
                print("Error: Invalid object format in data_list item")
        except Exception as e:
            print(f"Error: {e}")
    return output_string
