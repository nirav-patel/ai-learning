"""
LangChain + Amazon Bedrock — API Calls, Chaining & Response Parsing
====================================================================
Demonstrates using LangChain's ChatBedrock (langchain-aws) instead of
calling boto3 directly, covering:

  1. Basic call      – ChatBedrock.invoke() and raw AIMessage response
  2. StrOutputParser – strip AIMessage down to a plain string
  3. PromptTemplate  – parameterised prompts via ChatPromptTemplate
  4. LCEL chains     – compose prompt | model | parser with the pipe operator
  5. StructuredOutputParser – ResponseSchema-driven extraction from a review
  6. Pydantic parser – type-safe structured output with a Pydantic model

Run:
    pip install langchain langchain-aws langchain-core
    python l10_langchain_bedrock.py

Dependencies (add to requirements.txt):
    langchain
    langchain-aws
    langchain-core
    pydantic
"""

import json
import os
from pydoc import text
import textwrap

from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field

# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv(find_dotenv())

MODEL_ID = "us.anthropic.claude-sonnet-4-6"
AWS_REGION = os.getenv("AWS_REGION", "us-west-1")

# ---------------------------------------------------------------------------
# Shared ChatBedrock instance (reused across all demos)
# ---------------------------------------------------------------------------
chat = ChatBedrock(
    model_id=MODEL_ID,
    region_name=AWS_REGION,
    model_kwargs={"temperature": 0.0, "max_tokens": 512},
)


# ===========================================================================
# 1. Basic call — returns an AIMessage object
# ===========================================================================
def demo_basic_call():
    print("\n" + "=" * 60)
    print("1. Basic call → AIMessage")
    print("=" * 60)

    messages = [
        SystemMessage(content="You are a concise assistant."),
        HumanMessage(content="What is the capital of France?"),
    ]

    response = chat.invoke(messages)

    print(f"Type   : {type(response).__name__}")
    print(f"Content: {response.content}")
    print(f"Usage  : {response.usage_metadata}")


# ===========================================================================
# 2. StrOutputParser — convert AIMessage → plain string
# ===========================================================================
def demo_str_output_parser():
    print("\n" + "=" * 60)
    print("2. StrOutputParser → plain string")
    print("=" * 60)

    parser = StrOutputParser()

    messages = [HumanMessage(content="Name three programming languages in one sentence.")]
    raw = chat.invoke(messages)
    text = parser.invoke(raw)

    print(f"Raw: {raw.content}")
    print(f"Type   : {type(text).__name__}")
    print(f"Content: {text}")


# ===========================================================================
# 3. ChatPromptTemplate — parameterised prompts
# ===========================================================================
def demo_prompt_template():
    print("\n" + "=" * 60)
    print("3a. ChatPromptTemplate — parameterised prompt")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert in {domain}. Be concise."),
            ("human", "Explain {topic} in simple terms."),
        ]
    )

    formatted = prompt.invoke({"domain": "cloud computing", "topic": "serverless functions"})
    response = chat.invoke(formatted)

    print(textwrap.fill(response.content, width=72))

    print("\n" + "=" * 60)
    print("3b. ChatPromptTemplate — prompt template")
    print("=" * 60)

    template_string = """Translate the text \
    that is delimited by triple backticks \
    into a style that is {style}. \
    text: ```{text}```
    """
    prompt_template = ChatPromptTemplate.from_template(template_string)
    customer_style = """American English \
    in a calm and respectful tone
    """

    # Call the LLM to translate to the style of the customer message
    customer_email = """
    Arrr, I be fuming that me blender lid \
    flew off and splattered me kitchen walls \
    with smoothie! And to make matters worse, \
    the warranty don't cover the cost of \
    cleaning up me kitchen. I need yer help \
    right now, matey!
    """
    customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)
    # print(type(customer_messages))
    # print(type(customer_messages[0]))
    # print(customer_messages[0])

    # Call the LLM to translate to the style of the customer message
    customer_response = chat.invoke(customer_messages)
    print(textwrap.fill(customer_response.content, width=72))

    # Call the LLM to translate to the style of the service reply
    service_reply = """Hey there customer, \
    the warranty does not cover \
    cleaning expenses for your kitchen \
    because it's your fault that \
    you misused your blender \
    by forgetting to put the lid on before \
    starting the blender. \
    Tough luck! See ya!
    """

    service_style_pirate = """\
    a polite tone \
    that speaks in English Pirate\
    """

    service_messages = prompt_template.format_messages(
    style=service_style_pirate,
    text=service_reply)

    # print(service_messages[0].content)

    service_response = chat.invoke(service_messages)
    print(textwrap.fill(service_response.content, width=72))


# ===========================================================================
# 4. LCEL chain — prompt | model | parser
# ===========================================================================
def demo_lcel_chain():
    print("\n" + "=" * 60)
    print("4. LCEL chain  (prompt | model | parser)")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "Give me a one-paragraph summary of {subject}."),
        ]
    )

    chain = prompt | chat | StrOutputParser()

    result = chain.invoke({"subject": "the Python GIL"})
    print(textwrap.fill(result, width=72))


# ===========================================================================
# 5. PydanticOutputParser — schema-driven structured extraction
#
# This is the modern equivalent of the legacy ResponseSchema +
# StructuredOutputParser pattern (from LangChain 0.x / ChatGPT tutorials).
#
# How it maps:
#   ResponseSchema(name, description)          → Pydantic Field(description=...)
#   StructuredOutputParser.from_response_schemas([...]) → PydanticOutputParser(pydantic_object=...)
#   parser.get_format_instructions()           → same method, same purpose
#
# Advantages over the legacy approach:
#   • Type-safe — output is a validated Pydantic object, not a raw dict.
#   • format_instructions are auto-generated from field descriptions.
#   • First-class support in langchain_core; no extra package needed.
# ===========================================================================

customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""


class ReviewExtraction(BaseModel):
    gift: bool = Field(
        description=(
            "Was the item purchased as a gift for someone else? "
            "True if yes, False if not or unknown."
        )
    )
    delivery_days: int = Field(
        description=(
            "How many days did it take for the product to arrive? "
            "Output -1 if this information is not found."
        )
    )
    price_value: list[str] = Field(
        description=(
            "All sentences about the value or price of the product, "
            "as a list of strings."
        )
    )


def demo_structured_output_parser():
    print("\n" + "=" * 60)
    print("5. PydanticOutputParser (structured extraction) → typed object")
    print("=" * 60)

    # ── 1. Build parser from the Pydantic schema ──────────────────────────
    parser = PydanticOutputParser(pydantic_object=ReviewExtraction)

    # ── 2. Prompt — inject format_instructions so the model knows the schema ─
    #    {format_instructions} expands to a full JSON schema description
    #    that instructs the model on the exact keys and value types to return.
    review_template = """\
For the following customer review, extract the requested information.

{format_instructions}

text: {text}
"""
    prompt = ChatPromptTemplate.from_template(review_template)

    # ── 3. Chain: prompt | model | parser ─────────────────────────────────
    chain = prompt | chat | parser

    result: ReviewExtraction = chain.invoke(
        {
            "text": customer_review,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    print(f"Type         : {type(result).__name__}")
    print(f"gift         : {result.gift}")
    print(f"delivery_days: {result.delivery_days}")
    print(f"price_value  : {result.price_value}")
    print("\nFull object:")
    print(json.dumps(result.model_dump(), indent=2))

    # ── review_template_2: field descriptions inline, {format_instructions} at end ─
    #    Difference from review_template_1:
    #      • The field extraction rules are written directly in the prompt body
    #        (more explicit for the model).
    #      • {format_instructions} appears AFTER the text, not before it.
    #        Both orderings work; putting it last keeps the output-format
    #        instructions close to where the model writes its answer.
    print("\n--- review_template_2 (inline field rules, format_instructions last) ---")

    review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

    prompt2 = ChatPromptTemplate.from_template(review_template_2)
    chain2 = prompt2 | chat | parser

    result2: ReviewExtraction = chain2.invoke(
        {
            "text": customer_review,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    print(f"Type         : {type(result2).__name__}")
    print(f"gift         : {result2.gift}")
    print(f"delivery_days: {result2.delivery_days}")
    print(f"price_value  : {result2.price_value}")
    print("\nFull object:")
    print(json.dumps(result2.model_dump(), indent=2))


# ===========================================================================
# 6. Pydantic output parser — type-safe structured output
# ===========================================================================
class ProductReview(BaseModel):
    sentiment: str = Field(description="Overall sentiment: positive, neutral, or negative")
    score: int = Field(description="Sentiment score from 1 (very negative) to 5 (very positive)")
    key_points: list[str] = Field(description="Up to three key points from the review")


def demo_pydantic_parser():
    print("\n" + "=" * 60)
    print("6. Pydantic parser → typed Python object")
    print("=" * 60)

    parser = JsonOutputParser(pydantic_object=ProductReview)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a sentiment analyser. Respond with valid JSON only — "
                "no markdown fences.\n{format_instructions}",
            ),
            ("human", "Review: {review}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | chat | parser

    review_text = (
        "The noise-cancelling headphones are fantastic — crystal-clear audio and "
        "very comfortable. Battery life could be better, but overall a great buy."
    )

    result = chain.invoke({"review": review_text})

    print(f"Type      : {type(result).__name__}")
    print(f"Result    : {result}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    demo_basic_call()
    demo_str_output_parser()
    demo_prompt_template()
    demo_lcel_chain()
    demo_structured_output_parser()
    demo_pydantic_parser()
