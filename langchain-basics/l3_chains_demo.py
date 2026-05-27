"""
LangChain Chains Demo — Amazon Bedrock Edition
===============================================
Based on the L3-Chains course notebook, adapted to use AWS Bedrock (Claude)
instead of OpenAI.

Covers:
  1. LLMChain          — Basic prompt + LLM chain
  2. SimpleSequentialChain — Two chains where Chain 1 output feeds Chain 2 input
  3. SequentialChain    — Multi-step chain with named inputs/outputs
  4. Router Chain       — Route input to specialist sub-chains

  ┌─────────────────────────────────────────────────────────────────────┐
  │  APPROACH A — Legacy (LLMChain / langchain_classic)                 │
  │  Still functional in LangChain 0.3 but officially deprecated.       │
  │  Closely mirrors the original course notebook style.                │
  ├─────────────────────────────────────────────────────────────────────┤
  │  APPROACH B — Modern LCEL (prompt | llm | parser)                   │
  │  Recommended from LangChain 0.2+.                                   │
  └─────────────────────────────────────────────────────────────────────┘

Run:
    python3 l3_chains_demo.py

Dependencies:
    langchain>=1.0
    langchain-aws
    langchain-classic
    langchain-core
    python-dotenv
"""

import os
import warnings

from dotenv import load_dotenv, find_dotenv
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Legacy chain imports (deprecated but still functional) ────────────────────
# In langchain 1.x, chains have moved to the langchain-classic package.
from langchain_classic.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain_classic.chains.router import MultiPromptChain
from langchain_classic.chains.router.llm_router import LLMRouterChain, RouterOutputParser

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")

MODEL_ID   = "us.anthropic.claude-sonnet-4-6"
AWS_REGION = os.getenv("AWS_REGION", "us-west-1")

# ── Sample product review data (replaces the CSV from the original notebook) ──
PRODUCT_REVIEWS = [
    {
        "Product": "Queen Size Sheet Set",
        "Review": "I ordered a king size set. My only criticism would be that I wish seller would be more"
                  " careful in future to double check the size of the sheet set they are sending. It looks"
                  " like my kids will be using these sheets. I appreciate the quick delivery and the"
                  " following up from the company on the order.",
    },
    {
        "Product": "Waterproof Phone Pouch",
        "Review": "I loved the waterproof sac, although the opening was a bit difficult to use as,"
                  " once you've put the phone in, it's almost impossible to get open again. I think the"
                  " pouch could benefit from a slightly larger opening.",
    },
    {
        "Product": "Luxury Air Mattress",
        "Review": "This mattress had a small hole in the top of it (barely noticeable) when I opened the"
                  " box. It didn't affect the function of the mattress, but I thought it was worth noting.",
    },
    {
        "Product": "Pillows Insert",
        "Review": "This is the best throw pillow fillers on Amazon. I've tried many others and they all"
                  " go flat. These hold their shape for a very long time. I bought 10 of them and they"
                  " are still as plump as the day I bought them.",
    },
    {
        "Product": "Milk Frother Handheld",
        "Review": "I loved this product. But they only seem to last a few months. The motor on this one"
                  " stopped working after 2 months of use. I've gone through 3 of them now.",
    },
    {
        "Product": "L'Or Espresso Capsules",
        "Review": "Je suis very disappointed de ce produit. La qualité du café est inférieure. "
                  "Il n'y a pas assez de café dans les capsules. Je ne recommande pas ce produit.",
    },
]


def make_llm(temperature: float = 0.9, max_tokens: int = 1024) -> ChatBedrock:
    return ChatBedrock(
        model_id=MODEL_ID,
        region_name=AWS_REGION,
        model_kwargs={"temperature": temperature, "max_tokens": max_tokens},
    )


def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LLMChain
# ─────────────────────────────────────────────────────────────────────────────

def demo_llm_chain():
    separator("1. LLMChain")

    llm = make_llm()

    # ── APPROACH A: Legacy LLMChain ───────────────────────────────────────────
    print("\n[Approach A — Legacy LLMChain]")

    prompt = ChatPromptTemplate.from_template(
        "What is the best name to describe a company that makes {product}?"
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    product = "Queen Size Sheet Set"
    result = chain.run(product)
    print(f"Product : {product}")
    print(f"Result  : {result}")

    # ── APPROACH B: Modern LCEL ───────────────────────────────────────────────
    print("\n[Approach B — Modern LCEL]")

    lcel_chain = prompt | llm | StrOutputParser()
    result_b = lcel_chain.invoke({"product": product})
    print(f"Product : {product}")
    print(f"Result  : {result_b}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SimpleSequentialChain
# ─────────────────────────────────────────────────────────────────────────────

def demo_simple_sequential_chain():
    separator("2. SimpleSequentialChain")

    llm = make_llm()

    # ── APPROACH A: Legacy SimpleSequentialChain ──────────────────────────────
    print("\n[Approach A — Legacy SimpleSequentialChain]")

    first_prompt = ChatPromptTemplate.from_template(
        "What is the best name to describe a company that makes {product}?"
    )
    chain_one = LLMChain(llm=llm, prompt=first_prompt)

    second_prompt = ChatPromptTemplate.from_template(
        "Write a 20 words description for the following company: {company_name}"
    )
    chain_two = LLMChain(llm=llm, prompt=second_prompt)

    overall_simple_chain = SimpleSequentialChain(
        chains=[chain_one, chain_two],
        verbose=True,
    )

    product = "Queen Size Sheet Set"
    result = overall_simple_chain.run(product)
    print(f"\nFinal output: {result}")

    # ── APPROACH B: Modern LCEL ───────────────────────────────────────────────
    print("\n[Approach B — Modern LCEL]")

    lcel_chain = (
        first_prompt
        | llm
        | StrOutputParser()
        | (lambda company_name: {"company_name": company_name})
        | second_prompt
        | llm
        | StrOutputParser()
    )
    result_b = lcel_chain.invoke({"product": product})
    print(f"Final output: {result_b}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SequentialChain  (multiple named inputs/outputs)
# ─────────────────────────────────────────────────────────────────────────────

def demo_sequential_chain():
    separator("3. SequentialChain")

    llm = make_llm()

    # ── APPROACH A: Legacy SequentialChain ────────────────────────────────────
    print("\n[Approach A — Legacy SequentialChain]")

    # Chain 1: Translate review to English
    first_prompt = ChatPromptTemplate.from_template(
        "Translate the following review to English:\n\n{Review}"
    )
    chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="English_Review")

    # Chain 2: Summarise the English review
    second_prompt = ChatPromptTemplate.from_template(
        "Can you summarize the following review in 1 sentence:\n\n{English_Review}"
    )
    chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")

    # Chain 3: Detect the original language
    third_prompt = ChatPromptTemplate.from_template(
        "What language is the following review:\n\n{Review}"
    )
    chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="language")

    # Chain 4: Write a follow-up response in the original language
    fourth_prompt = ChatPromptTemplate.from_template(
        "Write a follow up response to the following summary in the specified language:\n\n"
        "Summary: {summary}\n\nLanguage: {language}"
    )
    chain_four = LLMChain(llm=llm, prompt=fourth_prompt, output_key="followup_message")

    overall_chain = SequentialChain(
        chains=[chain_one, chain_two, chain_three, chain_four],
        input_variables=["Review"],
        output_variables=["English_Review", "summary", "followup_message"],
        verbose=True,
    )

    review = PRODUCT_REVIEWS[5]["Review"]   # French espresso review
    print(f"\nOriginal review:\n{review}\n")
    result = overall_chain(review)

    print("\n--- Outputs ---")
    print(f"English_Review   : {result['English_Review']}")
    print(f"Summary          : {result['summary']}")
    print(f"Followup message : {result['followup_message']}")

    # ── APPROACH B: Modern LCEL ───────────────────────────────────────────────
    print("\n[Approach B — Modern LCEL]")

    from operator import itemgetter
    from langchain_core.runnables import RunnableParallel, RunnableLambda

    translate_chain = first_prompt | llm | StrOutputParser()
    summarize_chain = second_prompt | llm | StrOutputParser()
    detect_lang_chain = third_prompt | llm | StrOutputParser()
    followup_chain = fourth_prompt | llm | StrOutputParser()

    lcel_overall = (
        RunnableParallel(
            English_Review=translate_chain,
            language=detect_lang_chain,
            Review=itemgetter("Review"),
        )
        | RunnableParallel(
            English_Review=itemgetter("English_Review"),
            summary=RunnableLambda(
                lambda d: summarize_chain.invoke({"English_Review": d["English_Review"]})
            ),
            language=itemgetter("language"),
        )
        | RunnableParallel(
            English_Review=itemgetter("English_Review"),
            summary=itemgetter("summary"),
            followup_message=RunnableLambda(
                lambda d: followup_chain.invoke(
                    {"summary": d["summary"], "language": d["language"]}
                )
            ),
        )
    )

    result_b = lcel_overall.invoke({"Review": review})
    print(f"\nEnglish_Review   : {result_b['English_Review']}")
    print(f"Summary          : {result_b['summary']}")
    print(f"Followup message : {result_b['followup_message']}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Router Chain
# ─────────────────────────────────────────────────────────────────────────────

PHYSICS_TEMPLATE = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise \
and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""

MATH_TEMPLATE = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down hard problems into their \
component parts, answer the component parts, and then put them together \
to answer the broader question.

Here is a question:
{input}"""

HISTORY_TEMPLATE = """You are a very good historian. \
You have an excellent knowledge of and understanding of people, \
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence \
and the ability to make use of it to support your explanations and judgements.

Here is a question:
{input}"""

COMPUTER_SCIENCE_TEMPLATE = """You are a successful computer scientist. \
You have a passion for creativity, collaboration, forward-thinking, confidence, \
strong problem-solving capabilities, understanding of theories and algorithms, \
and excellent communication skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by describing the solution \
in imperative steps that a machine can easily interpret and you know how to choose a \
solution that has a good balance between time complexity and space complexity.

Here is a question:
{input}"""

PROMPT_INFOS = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": PHYSICS_TEMPLATE,
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": MATH_TEMPLATE,
    },
    {
        "name": "History",
        "description": "Good for answering history questions",
        "prompt_template": HISTORY_TEMPLATE,
    },
    {
        "name": "computer science",
        "description": "Good for answering computer science questions",
        "prompt_template": COMPUTER_SCIENCE_TEMPLATE,
    },
]

MULTI_PROMPT_ROUTER_TEMPLATE = """\
Given a raw text input to a language model select the model prompt best suited \
for the input. You will be given the names of the available prompts and a \
description of what the prompt is best suited for. You may also revise the \
original input if you think that revising it will ultimately lead to a better \
response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \\ "DEFAULT" or name of the prompt to use in {destinations}
    "next_inputs": string \\ a potentially modified version of the original input
}}}}
```

REMEMBER: The value of "destination" MUST match one of the candidate prompts listed \
below. If "destination" does not fit any of the specified prompts, set it to "DEFAULT."
REMEMBER: "next_inputs" can just be the original input if you don't think any \
modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""


def demo_router_chain():
    separator("4. Router Chain")

    llm = make_llm(temperature=0)

    # ── APPROACH A: Legacy MultiPromptChain ───────────────────────────────────
    print("\n[Approach A — Legacy MultiPromptChain]")

    destination_chains = {}
    for p_info in PROMPT_INFOS:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain

    destinations = [f"{p['name']}: {p['description']}" for p in PROMPT_INFOS]
    destinations_str = "\n".join(destinations)

    default_prompt = ChatPromptTemplate.from_template("{input}")
    default_chain = LLMChain(llm=llm, prompt=default_prompt)

    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    chain = MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True,
    )

    questions = [
        "What is black body radiation?",
        "what is 2 + 2",
        "Why does every cell in our body contain DNA?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        result = chain.run(question)
        print(f"Answer  : {result}")

    # ── APPROACH B: Modern LCEL with RunnableParallel routing ─────────────────
    print("\n[Approach B — Modern LCEL routing with RunnableBranch]")

    import json
    from langchain_core.runnables import RunnableBranch, RunnableLambda

    # Build destination chains as LCEL runnables
    lcel_destination_chains = {}
    for p_info in PROMPT_INFOS:
        dest_prompt = ChatPromptTemplate.from_template(p_info["prompt_template"])
        lcel_destination_chains[p_info["name"]] = dest_prompt | llm | StrOutputParser()

    default_lcel_chain = ChatPromptTemplate.from_template("{input}") | llm | StrOutputParser()

    # Router: classifies the question and returns (destination, next_inputs)
    router_prompt_lcel = PromptTemplate(
        template=router_template,
        input_variables=["input"],
    )
    router_lcel = router_prompt_lcel | llm | StrOutputParser()

    def route(info: dict) -> str:
        raw = router_lcel.invoke({"input": info["input"]})
        # Extract JSON from the markdown code block
        try:
            json_start = raw.index("```json") + 7
            json_end = raw.index("```", json_start)
            parsed = json.loads(raw[json_start:json_end].strip())
        except (ValueError, json.JSONDecodeError):
            return default_lcel_chain.invoke({"input": info["input"]})

        destination = parsed.get("destination", "DEFAULT")
        next_input = parsed.get("next_inputs", info["input"])
        print(f"  → routed to: {destination}")

        if destination in lcel_destination_chains:
            return lcel_destination_chains[destination].invoke({"input": next_input})
        return default_lcel_chain.invoke({"input": next_input})

    for question in questions:
        print(f"\nQuestion: {question}")
        result = RunnableLambda(route).invoke({"input": question})
        print(f"Answer  : {result}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("LangChain Chains Demo — Amazon Bedrock Edition")
    print("Model:", MODEL_ID)

    demo_llm_chain()
    demo_simple_sequential_chain()
    demo_sequential_chain()
    demo_router_chain()
