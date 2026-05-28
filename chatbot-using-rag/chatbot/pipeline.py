"""pipeline.py — RAGPipeline: the core 3-step RAG chain.

HOW IT WORKS
────────────
  Input:  question (str)  +  session_id (str)

  Step 1 — Condenser (only when history exists):
    history + question → self-contained standalone question  (LLM call)
    e.g. "what about fathers?" → "what is the paternity leave policy?"

  Step 2 — Retriever:
    standalone_question → top-k diverse chunks  (hybrid vector + BM25, no LLM)

  Step 3 — Answerer:
    context + history + original question → streamed answer  (LLM call)

History is read from AppState and appended after each completed turn.
The pipeline itself is stateless — it only reads the history callable.
"""
from __future__ import annotations

from typing import Callable, Generator

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


class RAGPipeline:
    """Three-step RAG pipeline: condense → retrieve → stream answer.

    Args:
        retriever:      LangChain retriever (hybrid Weaviate retriever).
        llm:            Any LangChain ChatModel.
        get_history_fn: Callable(session_id) -> list[HumanMessage | AIMessage].
    """

    def __init__(self, retriever, llm, get_history_fn: Callable[[str], list]) -> None:
        self._retriever       = retriever
        self._get_history_fn  = get_history_fn
        self._condense_chain  = _CONDENSE_PROMPT | llm | StrOutputParser()
        self._answer_chain    = _QA_PROMPT | llm | StrOutputParser()

    def stream(self, question: str, session_id: str) -> Generator[str, None, None]:
        """Stream the RAG answer token-by-token, then persist the turn to history.

        Yields:
            Answer text chunks as they arrive from the LLM.
        """
        history = self._get_history_fn(session_id)

        # Step 1: condense follow-up into a standalone question (only with history)
        standalone = (
            self._condense_chain.invoke({"input": question, "chat_history": history})
            if history else question
        )

        # Step 2: retrieve diverse context chunks
        docs    = self._retriever.invoke(standalone)
        context = "\n\n---\n\n".join(d.page_content for d in docs)

        # Step 3: stream the answer
        answer_chunks: list[str] = []
        for chunk in self._answer_chain.stream(
            {"input": question, "chat_history": history, "context": context}
        ):
            answer_chunks.append(chunk)
            yield chunk

        # Persist the completed turn to session history
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content="".join(answer_chunks)))
