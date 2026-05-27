"""
L4: Process Inputs – Chain of Thought Reasoning
================================================
Demonstrates instructing the model to reason step-by-step before answering.

Key concept – Chain of Thought (CoT):
  Instead of asking for a direct answer, instruct the model to work through
  the problem in numbered steps. This helps the model:
    - Identify what type of question is being asked (Step 1)
    - Verify whether mentioned products actually exist (Step 2)
    - Detect assumptions the user is making (Step 3)
    - Check whether those assumptions are correct (Step 4)
    - Give an accurate, polite final answer (Step 5)

Why it matters:
  Without CoT, models may hallucinate product details or miss incorrect
  assumptions. Forcing explicit reasoning steps reduces errors significantly.

Extracting the final answer:
  The response uses a delimiter (####) to separate each step.
  We split on the delimiter and take the last segment to get just the
  customer-facing answer (without the internal reasoning).

Run: python l4_chain_of_thought.py
"""
from __future__ import annotations

import textwrap

from botocore.exceptions import ClientError
from dotenv import load_dotenv, find_dotenv

from utils import MODEL_ID, create_client, get_completion_sys_role

load_dotenv(find_dotenv())

DELIMITER = "####"

# System prompt that instructs the model to follow a 5-step reasoning process.
# The product list is hardcoded here (not loaded from catalog) to make this
# demo entirely self-contained and illustrate the CoT technique clearly.
CHAIN_OF_THOUGHT_SYSTEM_MESSAGE = textwrap.dedent(f"""
Follow these steps to answer the customer queries.
The customer query will be delimited with four hashtags, i.e. {DELIMITER}.

Step 1:{DELIMITER} First decide whether the user is \
asking a question about a specific product or products. \
Product category doesn't count.

Step 2:{DELIMITER} If the user is asking about \
specific products, identify whether \
the products are in the following list.
All available products:
1. Product: TechPro Ultrabook
   Category: Computers and Laptops | Brand: TechPro | Model: TP-UB100
   Warranty: 1 year | Rating: 4.5
   Features: 13.3-inch display, 8GB RAM, 256GB SSD, Intel Core i5
   Price: $799.99

2. Product: BlueWave Gaming Laptop
   Category: Computers and Laptops | Brand: BlueWave | Model: BW-GL200
   Warranty: 2 years | Rating: 4.7
   Features: 15.6-inch display, 16GB RAM, 512GB SSD, NVIDIA RTX 3060
   Price: $1199.99

3. Product: PowerLite Convertible
   Category: Computers and Laptops | Brand: PowerLite | Model: PL-CV300
   Warranty: 1 year | Rating: 4.3
   Features: 14-inch touchscreen, 8GB RAM, 256GB SSD, 360-degree hinge
   Price: $699.99

4. Product: TechPro Desktop
   Category: Computers and Laptops | Brand: TechPro | Model: TP-DT500
   Warranty: 1 year | Rating: 4.4
   Features: Intel Core i7, 16GB RAM, 1TB HDD, NVIDIA GTX 1660
   Price: $999.99

5. Product: BlueWave Chromebook
   Category: Computers and Laptops | Brand: BlueWave | Model: BW-CB100
   Warranty: 1 year | Rating: 4.1
   Features: 11.6-inch display, 4GB RAM, 32GB eMMC, Chrome OS
   Price: $249.99

Step 3:{DELIMITER} If the message contains products \
in the list above, list any assumptions that the \
user is making in their message \
e.g. that Laptop X is bigger than Laptop Y, or that Laptop Z has a 2 year warranty.

Step 4:{DELIMITER}: If the user made any assumptions, \
figure out whether the assumption is true based on your \
product information.

Step 5:{DELIMITER}: First, politely correct the \
customer's incorrect assumptions if applicable. \
Only mention or reference products in the list of \
5 available products, as these are the only 5 \
products that the store sells. \
Answer the customer in a friendly tone.

Use the following format:
Step 1:{DELIMITER} <step 1 reasoning>
Step 2:{DELIMITER} <step 2 reasoning>
Step 3:{DELIMITER} <step 3 reasoning>
Step 4:{DELIMITER} <step 4 reasoning>
Response to user:{DELIMITER} <response to customer>

Make sure to include {DELIMITER} to separate every step.
""").strip()


def ask_with_chain_of_thought(client, user_message: str) -> tuple[str, str]:
    """
    Send a customer query using chain-of-thought prompting.

    Returns:
        (full_response, final_answer)
          full_response – the complete step-by-step reasoning text
          final_answer  – just the customer-facing response (extracted from Step 5)
    """
    full_response = get_completion_sys_role(
        client, CHAIN_OF_THOUGHT_SYSTEM_MESSAGE, user_message
    )
    try:
        # The final answer is always the text after the last delimiter in the response
        final_answer = full_response.split(DELIMITER)[-1].strip()
    except Exception:
        final_answer = "Sorry, I'm having trouble right now, please try asking another question."
    return full_response, final_answer


def main():
    client = create_client()

    # Price comparison query – tests assumption detection (Step 3/4)
    # The user incorrectly assumes the Chromebook costs more than the Desktop
    user_message = "by how much is the BlueWave Chromebook more expensive than the TechPro Desktop"
    full_response, final_answer = ask_with_chain_of_thought(client, user_message)
    print("Full chain-of-thought response:")
    print(full_response)
    print('\n------\n')
    print(f"Final answer only:\n{final_answer}")
    print('\n======\n')

    # Query about a product the store doesn't carry – tests Step 1/2 handling
    user_message = "do you sell TVs?"
    full_response, final_answer = ask_with_chain_of_thought(client, user_message)
    print("Full chain-of-thought response:")
    print(full_response)
    print('\n------\n')
    print(f"Final answer only:\n{final_answer}")


if __name__ == "__main__":
    try:
        main()
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{MODEL_ID}'. Reason: {e}")
        exit(1)
