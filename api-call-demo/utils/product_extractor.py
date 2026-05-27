"""
Product/category extraction from customer queries
==================================================
Uses the LLM to parse a free-form customer message and identify which products
or categories the customer is asking about.

Key function:
  find_category_and_product(client, user_input)
    → raw string (Python list literal) to be parsed with read_string_to_list()

This is the "extraction" step used by:
  - l5_chaining_prompts.py  (step 1 of the chaining pipeline)
  - l6_check_outputs.py     (generate a response to then validate)
  - end_to_end_demo.py      (step 2 of the 7-step pipeline)
  - evaluate_with_rubric.py (generate a response to evaluate)
"""
import textwrap

from .bedrock_client import get_completion_sys_role
from .product_catalog import build_allowed_products_text, build_category_names


def find_category_and_product(client, user_input: str) -> str:
    """
    Ask the LLM to extract which products or categories the user mentioned.

    The model returns a Python list of dicts, each with either:
      - {'category': '<name>'}             (user asked about a whole category)
      - {'products': ['name1', 'name2']}   (user asked about specific products)

    Returns:
        Raw string output from the model — pass it through read_string_to_list()
        before use.
    """
    delimiter = "####"
    allowed_products = build_allowed_products_text()
    category_names = build_category_names()

    system_message = textwrap.dedent(f"""
    You will be provided with customer service queries. \
    The customer service query will be delimited with \
    {delimiter} characters.
    Output a python list of objects, where each object has \
    the following format:
        'category': <one of {category_names}>,
    OR
        'products': <a list of products that must \
        be found in the allowed products below>

    Where the categories and products must be found in \
    the customer service query.
    If a product is mentioned, it must be associated with \
    the correct category in the allowed products list below.
    If no products or categories are found, output an \
    empty list.

    Allowed products:

    {allowed_products}

    Only output the list of objects, with nothing else.
    """).strip()

    return get_completion_sys_role(
        client, system_message, f"{delimiter}{user_input}{delimiter}"
    )
