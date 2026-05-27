"""
L1: Language Models, the Chat Format and Tokens
================================================
Demonstrates the fundamentals of interacting with an LLM via the Chat API:

  1. Basic one-shot completion (no system message)
  2. Using a system message to set a model persona/style
  3. Combining multiple constraints in one system message
  4. Counting tokens to understand API cost/limits

Key concepts:
  get_completion()                – no system message, single user turn
  get_completion_sys_role()       – set a persona/role for the model
  get_completion_and_token_count()– like get_completion_sys_role, also returns token usage

Run: python l1_language_models.py
"""
from botocore.exceptions import ClientError
from dotenv import load_dotenv, find_dotenv

from utils import (
    MODEL_ID,
    create_client,
    get_completion,
    get_completion_sys_role,
    get_completion_and_token_count,
)

load_dotenv(find_dotenv())


def main():
    client = create_client()

    # -----------------------------------------------
    # Basic completions (no system message)
    # -----------------------------------------------

    # Simple factual question
    answer = get_completion(client, "What is the capital of France?")
    print(answer)

    # Test character-level reasoning (model must work letter-by-letter, not token-by-token)
    answer = get_completion(client, "Take the letters in lollipop and reverse them")
    print(answer)

    # -----------------------------------------------
    # System message: style control
    # -----------------------------------------------

    system = "You are an assistant who responds in the style of Dr Seuss."
    question = "Write me a very short poem about a happy carrot."
    answer = get_completion_sys_role(client, system, question)
    print(answer)
    print('\n------\n')

    # -----------------------------------------------
    # System message: length control
    # -----------------------------------------------

    system = "All your responses must be one sentence long."
    question = "Write me a very short poem about a happy carrot."
    answer = get_completion_sys_role(client, system, question)
    print(answer)
    print('\n------\n')

    # -----------------------------------------------
    # Combining style + length in a single system message
    # -----------------------------------------------

    system = (
        "You are an assistant who responds in the style of Dr Seuss. "
        "All your responses must be one sentence long."
    )
    question = "Write me a very short poem about a happy carrot."
    answer = get_completion_sys_role(client, system, question)
    print(answer)
    print('\n------\n')

    # -----------------------------------------------
    # Token counting: understand how many tokens a request uses
    # (useful for cost estimation and staying within context limits)
    # -----------------------------------------------

    role = "You are an assistant who responds in the style of Dr Seuss."
    question = "write me a very short poem about a happy carrot"
    answer, tokens = get_completion_and_token_count(client, role, question)
    print(answer)
    print(tokens)  # {'prompt_tokens': ..., 'completion_tokens': ..., 'total_tokens': ...}
    print('\n------\n')


if __name__ == "__main__":
    try:
        main()
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{MODEL_ID}'. Reason: {e}")
        exit(1)
