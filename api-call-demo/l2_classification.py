"""
L2: Evaluate Inputs – Classification
=====================================
Demonstrates classifying customer service queries into a structured
primary/secondary category taxonomy using a single system message.

Key concepts:
  - Delimiter-based input isolation: wrap the user query in #### so the model
    treats it as data, not as additional instructions.
  - Structured JSON output: ask the model to return {"primary": ..., "secondary": ...}
    so the result is machine-readable and easy to route.
  - Taxonomy design: a well-defined system message with clear categories
    dramatically improves classification accuracy.

Use cases:
  - Routing tickets to the right support team
  - Prioritizing issues (e.g. billing vs. general inquiry)

Run: python l2_classification.py
"""
import textwrap

from botocore.exceptions import ClientError
from dotenv import load_dotenv, find_dotenv

from utils import MODEL_ID, create_client, get_completion_sys_role

load_dotenv(find_dotenv())

# Delimiter used to isolate the user query from the system instructions.
# Using a distinctive string (####) makes it harder for users to accidentally
# (or intentionally) break out of the input boundary.
DELIMITER = "####"

# System message that defines the full classification taxonomy.
# The model will classify any query into exactly one primary + one secondary category.
CLASSIFICATION_SYSTEM_MESSAGE = textwrap.dedent(f"""
    You will be provided with customer service queries.
    The customer service query will be delimited with {DELIMITER} characters.
    Classify each query into a primary category and a secondary category.
    Provide your output in json format with the keys: primary and secondary.

    Primary categories: Billing, Technical Support, Account Management, or General Inquiry.

    Billing secondary categories:
    Unsubscribe or upgrade
    Add a payment method
    Explanation for charge
    Dispute a charge

    Technical Support secondary categories:
    General troubleshooting
    Device compatibility
    Software updates

    Account Management secondary categories:
    Password reset
    Update personal information
    Close account
    Account security

    General Inquiry secondary categories:
    Product information
    Pricing
    Feedback
    Speak to a human
""").strip()


def classify_query(client, user_query: str) -> str:
    """
    Classify a customer service query into primary and secondary categories.

    Args:
        client:     Bedrock runtime client.
        user_query: Raw customer message text.

    Returns:
        JSON string like: {"primary": "Account Management", "secondary": "Close account"}
    """
    user_message = f"{DELIMITER}{user_query}{DELIMITER}"
    return get_completion_sys_role(client, CLASSIFICATION_SYSTEM_MESSAGE, user_message)


def main():
    client = create_client()

    # Account management query – should classify as "Close account"
    query = "I want you to delete my profile and all of my user data"
    answer = classify_query(client, query)
    print(f"Query: {query}")
    print(f"Classification: {answer}")
    print('\n------\n')

    # General inquiry query – should classify as "Product information"
    query = "Tell me more about your flat screen tvs"
    answer = classify_query(client, query)
    print(f"Query: {query}")
    print(f"Classification: {answer}")
    print('\n------\n')


if __name__ == "__main__":
    try:
        main()
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{MODEL_ID}'. Reason: {e}")
        exit(1)
