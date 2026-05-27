"""
Evaluate LLM Response with a Rubric and vs. Ideal Answer
=========================================================
Demonstrates two more nuanced approaches to evaluating a full customer-service
response (as opposed to just the extraction step in evaluate_with_ideal.py):

1. eval_with_rubric():
   Ask the model to evaluate the response on multiple structured criteria:
     - Is it grounded in the provided context only? (no hallucination)
     - Does it include information NOT in the context? (hallucination check)
     - Is there factual disagreement with the context?
     - How many questions did the user ask, and were all of them answered?
   Returns a structured breakdown (multiple Y/N + counts).

2. eval_vs_ideal():
   Compare the response to a hand-crafted expert ideal answer and classify
   the relationship into one of 5 options (A–E):
     A – Submitted is a subset of expert, fully consistent
     B – Submitted is a superset of expert, fully consistent
     C – Submitted contains all the same details as expert
     D – There is a factual disagreement
     E – Answers differ but the difference doesn't matter for factuality
   Returns a single letter.

When to use which:
  eval_with_rubric  – when you want a detailed breakdown (good for debugging)
  eval_vs_ideal     – when you want a quick pass/fail or relative quality score

Run: python evaluate_with_rubric.py
"""
import os
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

# A manually curated ideal answer for a specific customer query.
# Used to benchmark the model's actual response against expert-level output.
TEST_SET_IDEAL = {
    "customer_msg": textwrap.dedent("""
        tell me about the smartx pro phone and the fotosnap camera, the dslr one.
        Also, what TVs or TV related products do you have?
    """).strip(),
    "ideal_answer": textwrap.dedent("""
        Of course! The SmartX ProPhone is a powerful smartphone with advanced camera features.
        For instance, it has a 12MP dual camera. Other features include 5G wireless and 128GB storage.
        It also has a 6.1-inch display. The price is $899.99.

        The FotoSnap DSLR Camera is great for capturing stunning photos and videos.
        Some features include 1080p video, 3-inch LCD, a 24.2MP sensor, and interchangeable lenses.
        The price is 599.99.

        For TVs and TV related products, we offer 3 TVs. All TVs offer HDR and Smart TV.

        The CineView 4K TV has vibrant colors and smart features.
        Some of these features include a 55-inch display, 4K resolution. It's priced at 599.

        The CineView 8K TV is a stunning 8K TV.
        Some features include a 65-inch display and 8K resolution. It's priced at 2999.99.

        The CineView OLED TV lets you experience vibrant colors.
        Some features include a 55-inch display and 4K resolution. It's priced at 1499.99.

        We also offer 2 home theater products, both of which include bluetooth.
        The SoundMax Home Theater is a powerful home theater system for an immersive audio experience.
        Its features include 5.1 channel, 1000W output, and wireless subwoofer. It's priced at 399.99.

        The SoundMax Soundbar is a sleek and powerful soundbar.
        Its features include 2.1 channel, 300W output, and wireless subwoofer. It's priced at 199.99.

        Are there any additional questions you may have about these products?
    """).strip(),
}


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def eval_with_rubric(client, test_set: dict, assistant_answer: str) -> str:
    """
    Evaluate the assistant's answer using a structured rubric.
    Checks factual grounding, hallucination, and question coverage.

    Args:
        test_set:         Dict with 'customer_msg' (question) and 'context' (product info used).
        assistant_answer: The response to evaluate.

    Returns:
        Multi-line string with structured Y/N answers and counts.
    """
    cust_msg = test_set["customer_msg"]
    context = test_set["context"]

    system_message = textwrap.dedent("""\
    You are an assistant that evaluates how well the customer service agent \
    answers a user question by looking at the context that the customer service \
    agent is using to generate its response.
    """).strip()

    user_message = textwrap.dedent(f"""\
You are evaluating a submitted answer to a question based on the context \
the agent uses to answer the question.
Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {cust_msg}
    ************
    [Context]: {context}
    ************
    [Submission]: {assistant_answer}
    ************
    [END DATA]

Compare the factual content of the submitted answer with the context. \
Ignore any differences in style, grammar, or punctuation.
Answer the following questions:
    - Is the Assistant response based only on the context provided? (Y or N)
    - Does the answer include information that is not provided in the context? (Y or N)
    - Is there any disagreement between the response and the context? (Y or N)
    - Count how many questions the user asked. (output a number)
    - For each question that the user asked, is there a corresponding answer to it?
      Question 1: (Y or N)
      Question 2: (Y or N)
      ...
      Question N: (Y or N)
    - Of the number of questions asked, how many of these questions were addressed by the answer? (output a number)
""").strip()

    return get_completion_sys_role(client, system_message, user_message)


def eval_vs_ideal(client, test_set: dict, assistant_answer: str) -> str:
    """
    Compare the assistant's answer to an expert ideal answer and classify the relationship.

    Args:
        test_set:         Dict with 'customer_msg' (question) and 'ideal_answer' (expert answer).
        assistant_answer: The response to compare.

    Returns:
        Single letter:
          A – Submitted is a subset of expert, fully consistent
          B – Submitted is a superset of expert, fully consistent
          C – Submitted contains all the same details as expert
          D – There is a factual disagreement between submitted and expert
          E – Answers differ but differences don't matter for factuality
    """
    cust_msg = test_set["customer_msg"]
    ideal = test_set["ideal_answer"]

    system_message = textwrap.dedent("""\
    You are an assistant that evaluates how well the customer service agent \
    answers a user question by comparing the response to the ideal (expert) response.
    Output a single letter and nothing else.
    """).strip()

    user_message = textwrap.dedent(f"""\
You are comparing a submitted answer to an expert answer on a given question. Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {cust_msg}
    ************
    [Expert]: {ideal}
    ************
    [Submission]: {assistant_answer}
    ************
    [END DATA]

Compare the factual content of the submitted answer with the expert answer. \
Ignore any differences in style, grammar, or punctuation.
The submitted answer may either be a subset or superset of the expert answer, or it may conflict with it.
Determine which case applies. Answer the question by selecting one of the following options:
    (A) The submitted answer is a subset of the expert answer and is fully consistent with it.
    (B) The submitted answer is a superset of the expert answer and is fully consistent with it.
    (C) The submitted answer contains all the same details as the expert answer.
    (D) There is a disagreement between the submitted answer and the expert answer.
    (E) The answers differ, but these differences don't matter from the perspective of factuality.
  choice_strings: ABCDE
""").strip()

    return get_completion_sys_role(client, system_message, user_message)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    client = create_client()

    print("=" * 60)
    print("Evaluating LLM response with a rubric")
    print("=" * 60)

    customer_msg = TEST_SET_IDEAL["customer_msg"]

    # Generate a response using the L5 chaining pipeline
    raw_extraction = find_category_and_product(client, customer_msg)
    category_and_product_list = read_string_to_list(raw_extraction)
    product_info = generate_output_string(category_and_product_list)

    final_system_message = textwrap.dedent("""
        You are a customer service assistant for a large electronic store.
        Respond in a friendly and helpful tone, with very concise answers.
        Make sure to ask the user relevant follow up questions.
    """).strip()
    chained_messages = [
        {"role": "user", "content": [{"text": customer_msg}]},
        {
            "role": "assistant",
            "content": [{"text": f"Relevant product information:\n{product_info}".strip()}],
        },
        {
            "role": "user",
            "content": [{"text": "Based on the product information above, please answer my original question."}],
        },
    ]
    assistant_answer = get_completion_sys_role(client, final_system_message, chained_messages)
    print(f"Assistant answer:\n{assistant_answer}\n")

    # -----------------------------------------------
    # Rubric evaluation – structured breakdown
    # -----------------------------------------------
    cust_prod_info = {"customer_msg": customer_msg, "context": product_info}
    rubric_result = eval_with_rubric(client, cust_prod_info, assistant_answer)
    print(f"Rubric evaluation:\n{rubric_result}\n")

    # -----------------------------------------------
    # Vs. ideal evaluation – single letter A–E
    # -----------------------------------------------
    vs_ideal_result = eval_vs_ideal(client, TEST_SET_IDEAL, assistant_answer)
    print(f"Evaluation vs ideal (real answer): {vs_ideal_result}")

    # Confirm the evaluator can detect a clearly wrong answer
    vs_ideal_bad = eval_vs_ideal(client, TEST_SET_IDEAL, "life is like a box of chocolates")
    print(f"Evaluation vs ideal (bad answer): {vs_ideal_bad}")


if __name__ == "__main__":
    try:
        main()
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{MODEL_ID}'. Reason: {e}")
        exit(1)
