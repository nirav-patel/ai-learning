"""
L6: Check Outputs
==================
Demonstrates two output-quality checks that should be applied AFTER the model
generates a response — before it is shown to the user:

1. Content moderation on the response:
   Run the model's output through moderate_content() to ensure the assistant
   hasn't generated harmful content (even if the input was benign).

2. Factual accuracy check:
   Ask a second model call to verify that the response:
     (a) actually answers the customer's question
     (b) only cites facts that appear in the product information that was supplied

   Returns 'Y' (response is good) or 'N' (inaccurate / insufficient).

Why check outputs?
  - Models can hallucinate product details (wrong price, wrong features).
  - A "meta-evaluator" model call is an automated way to catch these issues
    without a human reviewer in the loop.
  - This is the final gate before a response reaches the end user.

Run: python l6_check_outputs.py
"""
import json
import textwrap

from botocore.exceptions import ClientError
from dotenv import load_dotenv, find_dotenv

from utils import (
    MODEL_ID,
    create_client,
    get_completion_sys_role,
    moderate_content,
    read_string_to_list,
    generate_output_string,
    find_category_and_product,
)

load_dotenv(find_dotenv())


def check_factual_accuracy(client, user_message: str, product_information: str, agent_response: str) -> str:
    """
    Ask the model to evaluate whether the agent response is factually grounded
    in the provided product information and sufficiently answers the question.

    Args:
        client:             Bedrock runtime client.
        user_message:       The original customer question.
        product_information: The product data that was used to generate the response.
        agent_response:     The response to evaluate.

    Returns:
        'Y' – response is accurate and sufficient
        'N' – response is inaccurate or does not answer the question
    """
    system_message = textwrap.dedent("""
    You are an assistant that evaluates whether \
    customer service agent responses sufficiently \
    answer customer questions, and also validates that \
    all the facts the assistant cites from the product \
    information are correct.
    The product information and user and customer \
    service agent messages will be delimited by \
    3 backticks, i.e. ```.
    Respond with a Y or N character, with no punctuation:
    Y - if the output sufficiently answers the question \
    AND the response correctly uses product information
    N - otherwise

    Output a single letter only.
    """).strip()

    q_a_pair = textwrap.dedent(f"""
    Customer message: ```{user_message}```
    Product information: ```{product_information}```
    Agent response: ```{agent_response}```

    Does the response use the retrieved information correctly?
    Does the response sufficiently answer the question?

    Output Y or N
    """).strip()

    return get_completion_sys_role(client, system_message, q_a_pair)


def main():
    client = create_client()

    user_message = textwrap.dedent("""
    tell me about the smartx pro phone and the fotosnap camera, the dslr one.
    Also tell me about your tvs""").strip()

    # Generate a response to check (replicating the L5 chaining pipeline)
    raw_extraction = find_category_and_product(client, user_message)
    category_and_product_list = read_string_to_list(raw_extraction)
    product_information = generate_output_string(category_and_product_list)

    final_system_message = textwrap.dedent("""
        You are a customer service assistant for a large electronic store.
        Respond in a friendly and helpful tone, with very concise answers.
        Make sure to ask the user relevant follow up questions.
    """).strip()
    chained_messages = [
        {"role": "user", "content": [{"text": user_message}]},
        {
            "role": "assistant",
            "content": [{"text": f"Relevant product information:\n{product_information}".strip()}],
        },
        {
            "role": "user",
            "content": [{"text": "Based on the product information above, please answer my original question."}],
        },
    ]
    agent_response = get_completion_sys_role(client, final_system_message, chained_messages)
    print(f"Agent response:\n{agent_response}\n")

    # -----------------------------------------------
    # Check 1: Content moderation on the response
    # -----------------------------------------------
    print("=" * 60)
    print("Output Moderation Check")
    print("=" * 60)
    moderation_output = moderate_content(client, agent_response)
    print(json.dumps(moderation_output, indent=2))
    print('\n------\n')

    # -----------------------------------------------
    # Check 2: Factual accuracy check
    # -----------------------------------------------
    print("=" * 60)
    print("Factual Accuracy Check")
    print("=" * 60)

    # Real agent response – should return Y (factually grounded)
    result = check_factual_accuracy(client, user_message, product_information, agent_response)
    print(f"Factually correct (real response): {result}")

    # Completely off-topic response – should return N
    bad_response = "Life is like a box of chocolates"
    result = check_factual_accuracy(client, user_message, product_information, bad_response)
    print(f"Factually correct (bad response): {result}")


if __name__ == "__main__":
    try:
        main()
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{MODEL_ID}'. Reason: {e}")
        exit(1)
