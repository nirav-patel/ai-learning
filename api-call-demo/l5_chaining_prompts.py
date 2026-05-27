"""
L5: Process Inputs – Chaining Prompts
======================================
Demonstrates breaking a complex task into a sequence of focused prompt calls
("chaining prompts") instead of one large monolithic prompt.

The pipeline for answering a product query has 3 steps:
  Step 1 – Extract: ask the model which categories/products the user mentioned
            (uses find_category_and_product from utils)
  Step 2 – Lookup:  fetch full product details from the catalog
            (uses generate_output_string from utils)
  Step 3 – Answer:  inject the product details as context and generate
            a final customer-service response

Why chain prompts instead of one big prompt?
  - Smaller, focused prompts are easier to debug and test independently.
  - Intermediate results can be validated/filtered (e.g. remove irrelevant products).
  - Token cost is reduced: only the relevant subset of product data is included
    in the expensive response-generation call.
  - Each step can have its own output format (Python list → JSON → natural language).

Run: python l5_chaining_prompts.py
"""
from __future__ import annotations

import textwrap

from botocore.exceptions import ClientError
from dotenv import load_dotenv, find_dotenv

from utils import (
    MODEL_ID,
    create_client,
    get_completion_sys_role,
    read_string_to_list,
    generate_output_string,
    find_category_and_product,
)

load_dotenv(find_dotenv())

# System message for the final response-generation step.
CUSTOMER_SERVICE_SYSTEM_MESSAGE = textwrap.dedent("""
    You are a customer service assistant for a large electronic store.
    Respond in a friendly and helpful tone, with very concise answers.
    Make sure to ask the user relevant follow up questions.
""").strip()


def generate_customer_response(client, user_message: str, product_information: str) -> str:
    """
    Generate a final customer-service response given the user's question and
    pre-fetched product information (from generate_output_string).

    The message chain injects the product context as an assistant turn, then
    asks the model to answer based on that context. This ensures the model only
    uses verified product data — not hallucinated details.
    """
    chained_messages = [
        # Original user question
        {"role": "user", "content": [{"text": user_message}]},
        # Injected product context (pretend the assistant already retrieved it)
        {
            "role": "assistant",
            "content": [{"text": f"Relevant product information:\n{product_information}".strip()}],
        },
        # Ask the model to now answer using that context
        {
            "role": "user",
            "content": [{"text": "Based on the product information above, please answer my original question."}],
        },
    ]
    return get_completion_sys_role(client, CUSTOMER_SERVICE_SYSTEM_MESSAGE, chained_messages)


def main():
    client = create_client()

    user_message = textwrap.dedent("""
    tell me about the smartx pro phone and \
    the fotosnap camera, the dslr one. \
    Also tell me about your tvs""").strip()

    # Step 1: Extract which categories/products the user mentioned
    raw_extraction = find_category_and_product(client, user_message)
    category_and_product_list = read_string_to_list(raw_extraction)
    print("Step 1 – Extracted products/categories:")
    print(category_and_product_list)
    print('\n------\n')

    # Step 2: Look up full product details from the catalog
    product_information = generate_output_string(category_and_product_list)
    print("Step 2 – Product information fetched:")
    print(product_information)
    print('\n------\n')

    # Step 3: Generate a customer-facing response using the product context
    final_response = generate_customer_response(client, user_message, product_information)
    print("Step 3 – Final customer response:")
    print(final_response)
    print('\n------\n')


if __name__ == "__main__":
    try:
        main()
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{MODEL_ID}'. Reason: {e}")
        exit(1)
