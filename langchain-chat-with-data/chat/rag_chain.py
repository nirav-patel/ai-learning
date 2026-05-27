"""LCEL RAG chain — prompts and chain builder.

HOW THE CHAIN WORKS
────────────────────
  Input:  question (str)  +  session_id (str, used to look up history)

  Step 1 — Condenser (only when history is non-empty):
    history + question → self-contained standalone question  (LLM call)
    e.g. "what about fathers?" → "what is the paternity leave policy?"

  Step 2 — Retriever:
    standalone_question → top-k diverse chunks  (MMR vector search, no LLM)

  Step 3 — Answerer:
    context + history + original question → streamed answer  (LLM call)

History is maintained externally (in AppState) and injected via get_history_fn
so the chain itself has no side-effects — it only reads history, and the caller
is responsible for appending the new turn after streaming completes.
"""
from __future__ import annotations

from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

_CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the conversation history and a follow-up question, rephrase the "
        "follow-up as a self-contained standalone question. "
        "If there is no history, return the question unchanged. "
        "Return ONLY the rephrased question — do not answer it.",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

_QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the question using ONLY the context "
        "provided below. If the answer is not in the context, say exactly: "
        "'I don't know based on the available documents.'\n\n"
        "Context:\n{context}",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def build_rag_chain(retriever, llm, get_history_fn: Callable[[str], list]) -> Callable:
    """Build a pure LCEL RAG chain.

    Args:
        retriever:      LangChain retriever (MMR-configured Chroma retriever).
        llm:            ChatBedrock (or any ChatModel) instance.
        get_history_fn: Callable(session_id) -> list[HumanMessage | AIMessage].
                        Called at runtime to fetch the current session's history.

    Returns:
        A callable ``chain(question, session_id) -> generator[str]`` that streams
        the answer token-by-token and appends the completed turn to the history.
    """
    condense_chain = _CONDENSE_PROMPT | llm | StrOutputParser()
    answer_chain   = _QA_PROMPT | llm | StrOutputParser()

    def _run(question: str, session_id: str):
        history = get_history_fn(session_id)

        # Step 1: condense follow-up into a standalone question (only with history)
        standalone = (
            condense_chain.invoke({"input": question, "chat_history": history})
            if history else question
        )

        # Step 2: retrieve diverse context chunks
        docs    = retriever.invoke(standalone)
        context = "\n\n---\n\n".join(d.page_content for d in docs)

        # Step 3: stream the answer
        answer_chunks: list[str] = []
        for chunk in answer_chain.stream(
            {"input": question, "chat_history": history, "context": context}
        ):
            answer_chunks.append(chunk)
            yield chunk

        # Persist the completed turn to session history
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content="".join(answer_chunks)))

    return _run
