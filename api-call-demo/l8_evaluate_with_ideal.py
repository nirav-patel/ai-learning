"""
Evaluate LLM Response Against Ideal Answers (Few-Shot)
======================================================
Demonstrates evaluating the quality of the LLM's product extraction by
comparing its output to hand-crafted ideal (expected) answers.

Two prompt versions are compared:
  find_category_and_product_v1 – 1-shot example (one price-based query)
  find_category_and_product_v2 – 2-shot examples + "JSON only" constraint (improved)

Why two versions?
  - v1 sometimes outputs extra explanation text or gets confused on price queries.
  - v2 adds a second few-shot example and an explicit "no extra text" instruction,
    which improves both format compliance and accuracy.

Evaluation:
  eval_response_with_ideal() scores a single extraction result against the
  ideal (expected) product set per category. Returns a score 0.0–1.0.

  Running all test cases in msg_ideal_pairs_set.json measures overall
  extraction accuracy across a realistic set of queries.

Run: python evaluate_with_ideal.py
"""
import json
import os
import textwrap

from botocore.exceptions import ClientError
from dotenv import load_dotenv, find_dotenv

from utils import (
    MODEL_ID,
    create_client,
    get_completion_sys_role,
    build_allowed_products_text,
    build_category_names,
)

load_dotenv(find_dotenv())

# Load the test set: list of {customer_msg, ideal_answer} dicts
_BASE_DIR = os.path.dirname(__file__)
with open(os.path.join(_BASE_DIR, "msg_ideal_pairs_set.json")) as f:
    msg_ideal_pairs_set = json.load(f)


# ---------------------------------------------------------------------------
# Prompt versions for product/category extraction
# ---------------------------------------------------------------------------

def find_category_and_product_v1(client, user_input: str) -> str:
    """
    Version 1 – basic 1-shot few-shot prompt.
    Instructs the model to list ALL products in a category when a broad
    attribute like "most expensive" is mentioned (since the model shouldn't
    try to rank by price — it doesn't have price info at extraction time).
    """
    delimiter = "####"
    allowed_products = build_allowed_products_text()
    category_names = build_category_names()
    system_message = textwrap.dedent(f"""
    You will be provided with customer service queries. \
    The customer service query will be delimited with {delimiter} characters.
    Output a python list of json objects, where each object has the following format:
        'category': <one of {category_names}>,
    AND
        'products': <a list of products that must be found in the allowed products below>

    Where the categories and products must be found in the customer service query.
    If a product is mentioned, it must be associated with the correct category in the allowed products list below.
    If no products or categories are found, output an empty list.

    List out all products that are relevant to the customer service query based on how closely it relates
    to the product name and product category.
    Do not assume, from the name of the product, any features or attributes such as relative quality or price.

    The allowed products are provided in JSON format.
    The keys of each item represent the category.
    The values of each item is a list of products that are within that category.
    Allowed products: {allowed_products}
    """).strip()

    # 1-shot example: "most expensive computer" → return all computers (model shouldn't guess price)
    few_shot_user_1 = "I want the most expensive computer."
    few_shot_assistant_1 = (
        "[{'category': 'Computers and Laptops', "
        "'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', "
        "'TechPro Desktop', 'BlueWave Chromebook']}]"
    )
    chained_messages = [
        {"role": "user", "content": [{"text": f"{delimiter}{few_shot_user_1}{delimiter}"}]},
        {"role": "assistant", "content": [{"text": few_shot_assistant_1}]},
        {"role": "user", "content": [{"text": f"{delimiter}{user_input}{delimiter}"}]},
    ]
    return get_completion_sys_role(client, system_message, chained_messages)


def find_category_and_product_v2(client, user_input: str) -> str:
    """
    Version 2 – improved 2-shot few-shot prompt.

    Changes vs v1:
      - Added: "Do not output any additional text that is not in JSON format."
      - Added a second example (cheapest computer) so the model sees two price-based
        queries both returning the full product list — reinforcing the pattern.
    """
    delimiter = "####"
    allowed_products = build_allowed_products_text()
    category_names = build_category_names()
    system_message = textwrap.dedent(f"""
    You will be provided with customer service queries. \
    The customer service query will be delimited with {delimiter} characters.
    Output a python list of json objects, where each object has the following format:
        'category': <one of {category_names}>,
    AND
        'products': <a list of products that must be found in the allowed products below>
    Do not output any additional text that is not in JSON format.
    Do not write any explanatory text after outputting the requested JSON.

    Where the categories and products must be found in the customer service query.
    If a product is mentioned, it must be associated with the correct category in the allowed products list below.
    If no products or categories are found, output an empty list.

    List out all products that are relevant to the customer service query based on how closely it relates
    to the product name and product category.
    Do not assume, from the name of the product, any features or attributes such as relative quality or price.

    The allowed products are provided in JSON format.
    The keys of each item represent the category.
    The values of each item is a list of products that are within that category.
    Allowed products: {allowed_products}
    """).strip()

    # 2-shot examples: both price-based queries return the full category list
    few_shot_user_1 = "I want the most expensive computer. What do you recommend?"
    few_shot_assistant_1 = (
        "[{'category': 'Computers and Laptops', "
        "'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', "
        "'TechPro Desktop', 'BlueWave Chromebook']}]"
    )
    few_shot_user_2 = "I want the cheapest computer. What do you recommend?"
    few_shot_assistant_2 = (
        "[{'category': 'Computers and Laptops', "
        "'products': ['TechPro Ultrabook', 'BlueWave Gaming Laptop', 'PowerLite Convertible', "
        "'TechPro Desktop', 'BlueWave Chromebook']}]"
    )
    chained_messages = [
        {"role": "user", "content": [{"text": f"{delimiter}{few_shot_user_1}{delimiter}"}]},
        {"role": "assistant", "content": [{"text": few_shot_assistant_1}]},
        {"role": "user", "content": [{"text": f"{delimiter}{few_shot_user_2}{delimiter}"}]},
        {"role": "assistant", "content": [{"text": few_shot_assistant_2}]},
        {"role": "user", "content": [{"text": f"{delimiter}{user_input}{delimiter}"}]},
    ]
    return get_completion_sys_role(client, system_message, chained_messages)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_response_with_ideal(response: str, ideal: dict, debug: bool = False) -> float:
    """
    Score a model's extraction response against the ideal (expected) answer.

    Args:
        response: Raw string from find_category_and_product_v*
                  (a Python-dict list with 'category' and/or 'products' keys).
        ideal:    Dict mapping category name → list of expected product names.
        debug:    Print intermediate comparison details if True.

    Returns:
        Float 0.0–1.0 — fraction of categories correctly extracted.
        1.0 means every extracted category had exactly the right product set.
    """
    if debug:
        print("response:", response)

    # Claude sometimes returns single-quoted Python dicts; convert to valid JSON
    json_like_str = response.replace("'", '"')
    l_of_d = json.loads(json_like_str)

    # Edge cases
    if l_of_d == [] and ideal == []:
        return 1.0
    elif l_of_d == [] or ideal == []:
        return 0.0

    correct = 0
    if debug:
        print("l_of_d:", l_of_d)

    for d in l_of_d:
        cat = d.get("category")
        prod_l = d.get("products")
        if cat and prod_l:
            prod_set = set(prod_l)
            ideal_cat = ideal.get(cat)
            if not ideal_cat:
                if debug:
                    print(f"Category '{cat}' not found in ideal. ideal={ideal}")
                continue
            prod_set_ideal = set(ideal_cat)
            if debug:
                print("prod_set:", prod_set)
                print("prod_set_ideal:", prod_set_ideal)
            if prod_set == prod_set_ideal:
                if debug:
                    print("correct")
                correct += 1
            else:
                print("incorrect")
                print(f"  prod_set:       {prod_set}")
                print(f"  prod_set_ideal: {prod_set_ideal}")
                if prod_set <= prod_set_ideal:
                    print("  (response is a subset of ideal)")
                elif prod_set >= prod_set_ideal:
                    print("  (response is a superset of ideal)")

    return correct / len(l_of_d)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    client = create_client()

    test_queries = [
        "Which TV can I buy if I'm on a budget?",
        "I need a charger for my smartphone",
        "What computers do you have?",
        "Tell me about the SmartX Pro phone and the FotoSnap camera, the DSLR one. Also, what TVs do you have?",
    ]

    # -----------------------------------------------
    # Side-by-side comparison: v1 vs v2 on the same queries
    # -----------------------------------------------
    for label, fn in [("v1 (1-shot)", find_category_and_product_v1), ("v2 (2-shot)", find_category_and_product_v2)]:
        print("=" * 60)
        print(f"Few-shot demo — {label}")
        print("=" * 60)
        for query in test_queries:
            answer = fn(client, query)
            print(f"Q: {query}")
            print(f"A: {answer}\n")

    # -----------------------------------------------
    # Evaluate a single test case against its ideal answer
    # -----------------------------------------------
    print("=" * 60)
    print("Single test case evaluation")
    print("=" * 60)
    pair = msg_ideal_pairs_set[7]
    print(f"Customer message: {pair['customer_msg']}")
    print(f"Ideal answer: {pair['ideal_answer']}")
    response = find_category_and_product_v2(client, pair["customer_msg"])
    print(f"LLM Response: {response}")
    score = eval_response_with_ideal(response, pair["ideal_answer"])
    print(f"Score: {score}")

    # -----------------------------------------------
    # Full test suite – score all examples and report overall accuracy
    # -----------------------------------------------
    print("=" * 60)
    print("Full test suite evaluation (v2)")
    print("=" * 60)
    score_accum = 0.0
    for i, pair in enumerate(msg_ideal_pairs_set):
        print(f"\nexample {i}: {pair['customer_msg'][:60]}...")
        response = find_category_and_product_v2(client, pair["customer_msg"])
        score = eval_response_with_ideal(response, pair["ideal_answer"], debug=False)
        print(f"  score: {score}")
        score_accum += score

    n_examples = len(msg_ideal_pairs_set)
    fraction_correct = score_accum / n_examples
    print(f"\nFraction correct out of {n_examples}: {fraction_correct:.2f}")


if __name__ == "__main__":
    try:
        main()
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{MODEL_ID}'. Reason: {e}")
        exit(1)
