"""
End-to-End Customer Service Pipeline + Panel UI Chatbot
========================================================
Combines all L1–L6 concepts into a production-style pipeline.

The 7-step process_user_message() pipeline:
  Step 1 – Moderate input       (L3: content moderation)
  Step 2 – Extract products     (L5: chaining prompts — find_category_and_product)
  Step 3 – Lookup product info  (L5: chaining prompts — generate_output_string)
  Step 4 – Generate response    (L5: chaining prompts — inject context + answer)
  Step 5 – Moderate response    (L3: content moderation on output)
  Step 6 – Evaluate quality     (L6: factual accuracy check — Y/N evaluator)
  Step 7 – Return or escalate   (if Y → return response; if N → escalate to human)

Running modes:
  1. CLI single-shot demo:  python end_to_end_demo.py
  2. Interactive CLI chat:  python end_to_end_demo.py  (then type messages)
  3. Panel web UI:          panel serve end_to_end_demo.py
"""
import asyncio
import textwrap

import panel as pn
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

# System message used when generating the customer-service response (Step 4)
_CUSTOMER_SERVICE_SYSTEM_MESSAGE = textwrap.dedent("""
    You are a customer service assistant for a large electronic store.
    Respond in a friendly and helpful tone, with concise answers.
    Make sure to ask the user relevant follow-up questions.
""").strip()

# System message used by the evaluator model (Step 6)
_EVAL_SYSTEM_MESSAGE = textwrap.dedent("""
    You are an assistant that evaluates whether a customer service agent response
    sufficiently answers the customer's question, and that all facts cited from
    product information are correct.
    The customer message and agent response will be delimited by triple backticks.

    Respond with a single letter only, no punctuation:
    Y - if the response sufficiently answers the question AND uses product information correctly
    N - otherwise
""").strip()


def process_user_message(client, user_input: str, all_messages: list, debug: bool = True):
    """
    7-step end-to-end pipeline for processing a customer service query.

    Args:
        client:       Bedrock runtime client.
        user_input:   The customer's raw message text.
        all_messages: Conversation history (list of role/content dicts).
                      Pass [] for the first turn; pass the returned updated_messages
                      on subsequent turns to maintain context.
        debug:        If True, print step-by-step progress messages.

    Returns:
        (response_text, updated_messages)
          response_text    – the assistant's reply to show the user
          updated_messages – conversation history with this turn appended
          On safety/quality failure, returns a safe fallback + the original all_messages.
    """
    delimiter = "```"

    # Step 1: Moderate the user input — reject harmful requests immediately
    moderation_output = moderate_content(client, user_input)
    if moderation_output["flagged"]:
        if debug:
            print("Step 1: Input flagged by Moderation API.")
        return "Sorry, we cannot process this request.", all_messages
    if debug:
        print("Step 1: Input passed moderation check.")

    # Step 2: Extract which categories and products the user mentioned
    category_and_product_response = find_category_and_product(client, user_input)
    category_and_product_list = read_string_to_list(category_and_product_response)
    if debug:
        print("Step 2: Extracted list of products.")

    # Step 3: Look up full product details for the extracted items
    product_information = generate_output_string(category_and_product_list)
    if debug:
        print("Step 3: Looked up product information.")

    # Step 4: Generate a response using the product info as context
    # We build a 3-turn mini-conversation: user question → assistant injects product data
    # → user asks to answer. This keeps the product context scoped to this turn only.
    new_messages = [
        {"role": "user", "content": [{"text": f"{delimiter}{user_input}{delimiter}"}]},
        {
            "role": "assistant",
            "content": [{"text": f"Relevant product information:\n{product_information}".strip()}],
        },
        {
            "role": "user",
            "content": [{"text": "Based on the product information above, please answer my original question."}],
        },
    ]
    final_response = get_completion_sys_role(
        client,
        _CUSTOMER_SERVICE_SYSTEM_MESSAGE,
        all_messages + new_messages,
    )
    if debug:
        print("Step 4: Generated response to user question.")

    # Build updated history: carry only the user question and assistant reply forward.
    # We do NOT include the injected product-context turns to keep history clean.
    updated_messages = all_messages + [
        {"role": "user", "content": [{"text": user_input}]},
        {"role": "assistant", "content": [{"text": final_response}]},
    ]

    # Step 5: Moderate the response — ensure the assistant hasn't generated harmful content
    response_moderation = moderate_content(client, final_response)
    if response_moderation["flagged"]:
        if debug:
            print("Step 5: Response flagged by Moderation API.")
        return "Sorry, we cannot provide this information.", all_messages
    if debug:
        print("Step 5: Response passed moderation check.")

    # Step 6: Evaluate whether the response actually answers the question correctly
    eval_user_message = textwrap.dedent(f"""
        Customer message: {delimiter}{user_input}{delimiter}
        Agent response: {delimiter}{final_response}{delimiter}

        Does the response sufficiently answer the question?
    """).strip()
    evaluation_response = get_completion_sys_role(
        client, _EVAL_SYSTEM_MESSAGE, eval_user_message
    )
    if debug:
        print("Step 6: Model evaluated the response.")

    # Step 7: Return the response if the evaluator approves, otherwise escalate
    if "Y" in evaluation_response:
        if debug:
            print("Step 7: Model approved the response.")
        return final_response, updated_messages
    else:
        if debug:
            print("Step 7: Model disapproved the response.")
        fallback = (
            "I'm unable to provide the information you're looking for. "
            "I'll connect you with a human representative for further assistance."
        )
        return fallback, all_messages


# -----------------------------------------------
# Main – single-shot demo + interactive CLI chatbot
# -----------------------------------------------

def main():
    client = create_client()

    print("=" * 60)
    print("Single-shot demo")
    print("=" * 60)
    user_input = (
        "tell me about the smartx pro phone and the fotosnap camera, "
        "the dslr one. Also tell me about your tvs"
    )
    response, conversation = process_user_message(client, user_input, [], debug=True)
    print(f"\nFinal response:\n{response}\n")

    print("=" * 60)
    print("Interactive chatbot (type 'quit' to exit)")
    print("=" * 60)
    history = []
    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        reply, history = process_user_message(client, user_input, history, debug=False)
        print(f"Assistant: {reply}")


# -----------------------------------------------
# Panel UI – interactive web dashboard
# Run with: panel serve end_to_end_demo.py
# -----------------------------------------------

pn.extension()

_ui_client = create_client()
_chat_column = pn.Column()   # persistent chat history column — append to it directly
_context: list = []

inp = pn.widgets.TextInput(placeholder="Enter text here…")
button_conversation = pn.widgets.Button(name="Chat!", button_type="primary")

# Loading feedback widgets — shown while the LLM pipeline is running
_spinner = pn.indicators.LoadingSpinner(value=False, size=25, visible=False)
_status = pn.pane.Markdown("_Processing…_", visible=False)


async def on_chat_click(event):
    """
    Async button callback: show loading feedback immediately, then run the
    7-step pipeline in a background thread, then append the result to the chat.

    Why async + asyncio.to_thread?
      - process_user_message() is a blocking synchronous function (multiple LLM calls).
      - If we ran it directly on the main thread, Panel's event loop would be blocked
        and the spinner/disabled-button state would never render — including on the first click.
      - asyncio.to_thread() offloads the blocking call to a thread-pool thread, keeping
        the event loop free so Panel can update the UI before and after the LLM call.
    """
    user_input = inp.value_input.strip()
    if not user_input:
        return

    inp.value = ""

    # Show loading state BEFORE the LLM call — works on every click including the first
    button_conversation.disabled = True
    _spinner.value = True
    _spinner.visible = True
    _status.visible = True

    global _context
    try:
        # Run the blocking pipeline in a thread pool so the event loop stays free
        response, _context = await asyncio.to_thread(
            process_user_message, _ui_client, user_input, _context, False
        )
    finally:
        # Always restore UI state, even if the pipeline raises an exception
        button_conversation.disabled = False
        _spinner.value = False
        _spinner.visible = False
        _status.visible = False

    _chat_column.append(
        pn.Row("**User:**", pn.pane.Markdown(user_input, width=600))
    )
    _chat_column.append(
        pn.Row(
            "**Assistant:**",
            pn.pane.Markdown(response, width=600, styles={"background-color": "#F6F6F6"}),
        )
    )


button_conversation.on_click(on_chat_click)

dashboard = pn.Column(
    inp,
    pn.Row(button_conversation, _spinner, _status),
    _chat_column,
)

dashboard.servable()


if __name__ == "__main__":
    try:
        main()
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{MODEL_ID}'. Reason: {e}")
        exit(1)
