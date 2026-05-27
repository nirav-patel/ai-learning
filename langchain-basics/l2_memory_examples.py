"""
LangChain Memory Models — Amazon Bedrock Edition
=================================================
Covers all four memory types using TWO approaches side-by-side:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  APPROACH A — Legacy (ConversationChain + langchain.memory)         │
  │  Still functional in LangChain 0.3 but officially deprecated.       │
  │  Simple, less code, but couples memory to the chain object.         │
  ├─────────────────────────────────────────────────────────────────────┤
  │  APPROACH B — Modern LCEL (RunnableWithMessageHistory + trim)       │
  │  Recommended from LangChain 0.2+. Composes prompt | llm with        │
  │  explicit history management and trim_messages for token control.   │
  └─────────────────────────────────────────────────────────────────────┘

Memory strategies demonstrated:

  1. Buffer (full history)
     Legacy : ConversationBufferMemory
     Modern : InMemoryChatMessageHistory  (keeps all messages)

  2. Window (last-k exchanges)
     Legacy : ConversationBufferWindowMemory(k=N)
     Modern : trim_messages(strategy="last", max_tokens=N, token_counter=len)

  3. Token-limited buffer
     Legacy : ConversationTokenBufferMemory(max_token_limit=N)
     Modern : trim_messages(strategy="last", max_tokens=N, token_counter=llm)

  4. Summary buffer
     Legacy : ConversationSummaryBufferMemory(max_token_limit=N)
     Modern : manual summarise-then-append pattern with RunnableWithMessageHistory

Run:
    python3 l2_memory_examples.py

Dependencies:
    langchain>=0.3
    langchain-aws
    langchain-core
    python-dotenv
"""

import os
import warnings
from typing import Dict

from dotenv import load_dotenv, find_dotenv

# ── Legacy imports (still functional, but deprecated) ─────────────────────────
from langchain_aws import ChatBedrock
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryBufferMemory,
)

# ── Modern LCEL imports ────────────────────────────────────────────────────────
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv(find_dotenv())
warnings.filterwarnings("ignore")

MODEL_ID   = "us.anthropic.claude-sonnet-4-6"
AWS_REGION = os.getenv("AWS_REGION", "us-west-1")

SYSTEM_PROMPT = (
    "The following is a friendly conversation between a human and an AI. "
    "The AI is talkative and provides lots of specific details from its context. "
    "If the AI does not know the answer to a question, it truthfully says it does not know."
)


def make_llm(temperature: float = 0.0, max_tokens: int = 512) -> ChatBedrock:
    return ChatBedrock(
        model_id=MODEL_ID,
        region_name=AWS_REGION,
        model_kwargs={"temperature": temperature, "max_tokens": max_tokens},
    )


class _ApproxTokenChatBedrock(ChatBedrock):
    """ChatBedrock with local 4-chars-per-token approximation for token counting.

    Bedrock's CountTokens API rejects conversations ending with an AI message,
    which breaks ConversationTokenBufferMemory and ConversationSummaryBufferMemory.
    This subclass overrides those two methods to count locally instead.
    """

    def get_num_tokens_from_messages(self, messages) -> int:  # type: ignore[override]
        return max(1, sum(len(m.content) for m in messages) // 4)

    def get_num_tokens(self, text: str) -> int:  # type: ignore[override]
        return max(1, len(text) // 4)


def make_llm_approx_tokens(temperature: float = 0.0, max_tokens: int = 512) -> _ApproxTokenChatBedrock:
    """Like make_llm() but uses local token counting — needed by memory classes
    that call count_tokens on histories ending with an AI message."""
    return _ApproxTokenChatBedrock(
        model_id=MODEL_ID,
        region_name=AWS_REGION,
        model_kwargs={"temperature": temperature, "max_tokens": max_tokens},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Modern LCEL helper
# Creates a chain with RunnableWithMessageHistory so callers just pass
# {"input": "..."} and a session config.
# ─────────────────────────────────────────────────────────────────────────────
def make_modern_chain(llm: ChatBedrock, store: Dict, history_modifier=None):
    """
    Build a modern LCEL chain backed by an in-memory per-session history store.

    Parameters
    ----------
    history_modifier : callable, optional
        Applied to the message list before the prompt sees it.
        Use this to implement windowing, token trimming, etc.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    def get_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    base_chain = prompt | llm

    if history_modifier is not None:
        # Wrap the chain so history is trimmed before hitting the prompt
        from langchain_core.runnables import RunnableLambda

        def add_trimmed_history(inputs):
            history_obj = get_history(inputs["session_id"])
            trimmed = history_modifier(history_obj.messages)
            return {"history": trimmed, "input": inputs["input"]}

        return RunnableLambda(add_trimmed_history) | base_chain, get_history

    return RunnableWithMessageHistory(
        base_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    ), get_history


def modern_invoke(chain, session_id: str, user_input: str) -> str:
    """Invoke a RunnableWithMessageHistory chain and return the text reply."""
    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    return response.content


# ===========================================================================
# 1. Buffer Memory — full, unbounded history
# ===========================================================================
def demo_buffer_memory():
    print("\n" + "=" * 70)
    print("1. BUFFER MEMORY — stores the entire conversation verbatim")
    print("   Pros: perfect recall | Cons: grows forever")
    print("=" * 70)

    llm = make_llm()

    # ── APPROACH A: Legacy ─────────────────────────────────────────────
    print("\n[LEGACY] ConversationBufferMemory + ConversationChain")
    memory = ConversationBufferMemory()
    chain  = ConversationChain(llm=llm, memory=memory, verbose=False)

    chain.predict(input="Hi, my name is Alice")
    chain.predict(input="What is 1+1?")
    reply = chain.predict(input="What is my name?")

    print(f"  → Reply: {reply}")
    print(f"  → memory.buffer:\n{memory.buffer}")
    print(f"  → load_memory_variables: {memory.load_memory_variables({})}")

    # ── APPROACH B: Modern LCEL ────────────────────────────────────────
    print("\n[MODERN] InMemoryChatMessageHistory + RunnableWithMessageHistory")

    store: Dict = {}
    chain_m, get_history = make_modern_chain(llm, store)
    sid = "buffer-demo"

    modern_invoke(chain_m, sid, "Hi, my name is Alice")
    modern_invoke(chain_m, sid, "What is 1+1?")
    reply_m = modern_invoke(chain_m, sid, "What is my name?")

    print(f"  → Reply: {reply_m}")
    # Inspect the history object directly
    history = get_history(sid)
    print("  → Stored messages:")
    for msg in history.messages:
        role = type(msg).__name__.replace("Message", "")
        print(f"    [{role}] {msg.content[:80]}")


# ===========================================================================
# 2. Window Memory — last-k exchanges only
# ===========================================================================
def demo_window_memory():
    print("\n" + "=" * 70)
    print("2. WINDOW MEMORY — only the last k exchanges are kept")
    print("   Pros: bounded footprint | Cons: forgets early context")
    print("=" * 70)

    llm = make_llm()
    K   = 1   # keep only the single most recent exchange

    # ── APPROACH A: Legacy ─────────────────────────────────────────────
    print(f"\n[LEGACY] ConversationBufferWindowMemory(k={K})")

    # Stand-alone inspection
    wmem = ConversationBufferWindowMemory(k=K)
    wmem.save_context({"input": "Hi"},                    {"output": "What's up"})
    wmem.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
    print(f"  → After 2 saves, only last exchange survives: {wmem.load_memory_variables({})}")

    memory = ConversationBufferWindowMemory(k=K)
    chain  = ConversationChain(llm=llm, memory=memory, verbose=False)
    chain.predict(input="Hi, my name is Alice")   # turn 1 — will be evicted
    chain.predict(input="What is 1+1?")           # turn 2 — evicts turn 1
    reply = chain.predict(input="What is my name?")
    print(f"  → Reply (name forgotten): {reply}")

    # ── APPROACH B: Modern LCEL ────────────────────────────────────────
    print(f"\n[MODERN] trim_messages(strategy='last', max_tokens={K * 2}, token_counter=len)")
    # k exchanges = k*2 messages (each exchange is 1 Human + 1 AI)

    from langchain_core.runnables import RunnableLambda

    store: Dict = {}
    trimmer = lambda msgs: trim_messages(
        msgs,
        strategy="last",
        max_tokens=K * 2,     # with token_counter=len, this = keep last K*2 messages
        token_counter=len,    # each message costs 1 'token', so max_tokens == max_messages
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    sid = "window-demo"
    store[sid] = InMemoryChatMessageHistory()

    def invoke_with_trim(user_input: str) -> str:
        history_obj = store[sid]
        trimmed = trimmer(history_obj.messages)
        response = (prompt | llm).invoke({"history": trimmed, "input": user_input})
        # Manually append to the full history (we still store everything;
        # trimming happens only at prompt-injection time)
        history_obj.add_user_message(user_input)
        history_obj.add_ai_message(response.content)
        return response.content

    invoke_with_trim("Hi, my name is Alice")
    invoke_with_trim("What is 1+1?")
    reply_m = invoke_with_trim("What is my name?")
    print(f"  → Reply (name forgotten): {reply_m}")
    print(f"  → Full history has {len(store[sid].messages)} messages; "
          f"only last {K*2} are sent to the model")


# ===========================================================================
# 3. Token-limited Buffer Memory
# ===========================================================================
def demo_token_buffer_memory():
    print("\n" + "=" * 70)
    print("3. TOKEN-LIMITED BUFFER — trims by token count, not message count")
    print("   Pros: stays within model limits precisely | Cons: needs token counter")
    print("=" * 70)

    llm        = make_llm()
    MAX_TOKENS = 50

    # Use make_llm_approx_tokens() for the legacy memory object:
    # Bedrock's CountTokens API rejects histories ending with an AI message,
    # so we subclass ChatBedrock to count locally (4 chars ≈ 1 token) instead.
    llm_for_counting = make_llm_approx_tokens()

    # ── APPROACH A: Legacy ─────────────────────────────────────────────
    print(f"\n[LEGACY] ConversationTokenBufferMemory(max_token_limit={MAX_TOKENS})")
    memory = ConversationTokenBufferMemory(llm=llm_for_counting, max_token_limit=MAX_TOKENS)
    memory.save_context({"input": "AI is what?!"},            {"output": "Amazing!"})
    memory.save_context({"input": "Backpropagation is what?"}, {"output": "Beautiful!"})
    memory.save_context({"input": "Chatbots are what?"},       {"output": "Charming!"})

    result = memory.load_memory_variables({})
    print(f"  → History (oldest turns trimmed to fit {MAX_TOKENS} tokens):")
    print(f"    {result}")

    # ── APPROACH B: Modern LCEL ────────────────────────────────────────
    print(f"\n[MODERN] trim_messages(strategy='last', max_tokens={MAX_TOKENS}, token_counter=llm)")

    sid   = "token-demo"
    store = {sid: InMemoryChatMessageHistory()}
    history_obj = store[sid]

    # Pre-populate with the same three exchanges
    for human, ai in [
        ("AI is what?!",            "Amazing!"),
        ("Backpropagation is what?", "Beautiful!"),
        ("Chatbots are what?",       "Charming!"),
    ]:
        history_obj.add_user_message(human)
        history_obj.add_ai_message(ai)

    # approx_tokens counts chars//4 per message, matching the legacy counter above
    def approx_tokens(msgs):
        return max(1, sum(len(m.content) for m in msgs) // 4)

    trimmed = trim_messages(
        history_obj.messages,
        strategy="last",
        max_tokens=MAX_TOKENS,
        token_counter=approx_tokens,  # local approximation; no Bedrock API call
        include_system=True,
        allow_partial=False,
    )
    print(f"  → Full history: {len(history_obj.messages)} messages")
    print(f"  → After trim (≤{MAX_TOKENS} tokens): {len(trimmed)} messages kept")
    for msg in trimmed:
        role = type(msg).__name__.replace("Message", "")
        print(f"    [{role}] {msg.content}")


# ===========================================================================
# 4. Summary Buffer Memory — rolling LLM summary + recent verbatim turns
# ===========================================================================
def demo_summary_buffer_memory():
    print("\n" + "=" * 70)
    print("4. SUMMARY BUFFER — LLM summarises old turns, keeps recent ones verbatim")
    print("   Pros: long-term recall in compact form | Cons: extra LLM calls")
    print("=" * 70)

    schedule = (
        "There is a meeting at 8am with your product team. "
        "You will need your powerpoint presentation prepared. "
        "9am-12pm have time to work on your LangChain project which will go "
        "quickly because Langchain is such a powerful tool. "
        "At Noon, lunch at the italian resturant with a customer who is driving "
        "from over an hour away to meet you to understand the latest in AI. "
        "Be sure to bring your laptop to show the latest LLM demo."
    )

    llm        = make_llm()
    MAX_TOKENS = 100

    # ── APPROACH A: Legacy ─────────────────────────────────────────────
    print(f"\n[LEGACY] ConversationSummaryBufferMemory(max_token_limit={MAX_TOKENS})")
    # ConversationSummaryBufferMemory needs the LLM for both summarisation AND
    # token counting; use approx variant to avoid Bedrock CountTokens API issues.
    memory = ConversationSummaryBufferMemory(llm=make_llm_approx_tokens(), max_token_limit=MAX_TOKENS)
    memory.save_context({"input": "Hello"},                          {"output": "What's up"})
    memory.save_context({"input": "Not much, just hanging"},         {"output": "Cool"})
    memory.save_context({"input": "What is on the schedule today?"}, {"output": schedule})

    result = memory.load_memory_variables({})
    print("  → Memory (old turns summarised into 'System:' prefix):")
    print(f"    {result}")

    chain = ConversationChain(llm=llm, memory=memory, verbose=False)
    reply = chain.predict(input="What would be a good demo to show?")
    print(f"  → Reply: {reply}")
    print(f"  → Updated memory: {memory.load_memory_variables({})}")

    # ── APPROACH B: Modern LCEL ────────────────────────────────────────
    print(f"\n[MODERN] Manual summarise-then-inject pattern")
    print("  (No built-in summary memory in LCEL; we implement it explicitly)")

    from langchain_core.prompts import PromptTemplate

    summarise_prompt = PromptTemplate.from_template(
        "Progressively summarise the lines of conversation provided, "
        "adding onto the previous summary and returning a new summary.\n\n"
        "Current summary:\n{summary}\n\n"
        "New lines of conversation:\n{new_lines}\n\n"
        "New summary:"
    )

    sid         = "summary-demo"
    store       = {sid: InMemoryChatMessageHistory()}
    history_obj = store[sid]
    running_summary = ""

    def summarise_and_compress(new_human: str, new_ai: str) -> str:
        """Summarise all history + new exchange into a compact string."""
        new_lines = f"Human: {new_human}\nAI: {new_ai}"
        summary_chain = summarise_prompt | llm
        result = summary_chain.invoke({"summary": running_summary, "new_lines": new_lines})
        return result.content

    def invoke_with_summary(user_input: str) -> str:
        """
        Run one turn:
          1. Build context = SystemMessage(summary) + recent messages
          2. Invoke the LLM
          3. Update the running summary if history is growing large
        """
        nonlocal running_summary

        # Build prompt with summary injected as a system message
        messages = []
        if running_summary:
            messages.append(SystemMessage(content=f"Summary so far: {running_summary}"))
        # Append recent raw turns
        messages.extend(history_obj.messages[-4:])   # keep last 2 exchanges verbatim
        messages.append(HumanMessage(content=user_input))

        response = llm.invoke(messages)
        ai_reply = response.content

        # Save full turn to history
        history_obj.add_user_message(user_input)
        history_obj.add_ai_message(ai_reply)

        # Compress into summary when history grows beyond 4 raw messages
        if len(history_obj.messages) > 4:
            running_summary = summarise_and_compress(user_input, ai_reply)

        return ai_reply

    invoke_with_summary("Hello")
    invoke_with_summary("Not much, just hanging")
    invoke_with_summary(f"What is on the schedule today? {schedule}")
    reply_m = invoke_with_summary("What would be a good demo to show?")

    print(f"  → Reply: {reply_m}")
    print(f"  → Running summary:\n    {running_summary}")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  LangChain Memory Models — Amazon Bedrock Demo")
    print("#  (Legacy ConversationChain vs Modern LCEL side-by-side)")
    print("#" * 70)

    demo_buffer_memory()
    demo_window_memory()
    demo_token_buffer_memory()
    demo_summary_buffer_memory()

    print("\n" + "#" * 70)
    print("#  All demos complete.")
    print("#" * 70)
