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

LCEL CHAIN
──────────
  `self.lcel_chain` exposes the pipeline as a pure LangChain Runnable:
    Input:  {"input": str, "chat_history": list[HumanMessage | AIMessage]}
    Output: str  (the answer)

  This enables:
  - TruChain wrapping for offline evaluation (chatbot/eval/run_evaluation.py)
  - Future live PROD tracing via TruLens
  - Composability with other LangChain components
"""
from __future__ import annotations

from typing import Callable, Generator

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

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

    Attributes:
        lcel_chain: A pure LangChain Runnable that can be wrapped by TruChain
                    for evaluation or live PROD tracing.
                    Input:  {"input": str, "chat_history": list}
                    Output: str
    """

    def __init__(self, retriever, llm, get_history_fn: Callable[[str], list]) -> None:
        self._retriever       = retriever
        self._get_history_fn  = get_history_fn
        self._condense_chain  = _CONDENSE_PROMPT | llm | StrOutputParser()
        self._answer_chain    = _QA_PROMPT | llm | StrOutputParser()
        self.lcel_chain       = self._build_lcel_chain()

    def _build_lcel_chain(self) -> Runnable:
        """Build an LCEL chain: {"input", "chat_history"} → str.

        The retriever is wired in as a direct chain component (not wrapped in a
        lambda) so TruChain.select_context() can auto-detect it for RAG Triad
        feedback and the call graph is clearly readable in the TruLens dashboard.
        """
        from operator import itemgetter

        condense = self._condense_chain
        retriever = self._retriever
        answer = self._answer_chain

        def _condense_step(inputs: dict) -> dict:
            history = inputs.get("chat_history", [])
            standalone = (
                condense.invoke({"input": inputs["input"], "chat_history": history})
                if history else inputs["input"]
            )
            return {**inputs, "standalone_question": standalone}

        def _format_docs(docs) -> str:
            return "\n\n---\n\n".join(d.page_content for d in docs)

        # retriever is a direct component here (not hidden in a closure) so
        # TruChain.select_context() can walk the chain graph and find it.
        return (
            RunnableLambda(_condense_step).with_config(run_name="condense")
            | RunnablePassthrough.assign(
                context=(
                    itemgetter("standalone_question")
                    | retriever
                    | RunnableLambda(_format_docs)
                )
            ).with_config(run_name="retrieve")
            | answer
        )

    def stream(self, question: str, session_id: str) -> Generator[str, None, None]:
        """Stream the RAG answer token-by-token, then persist the turn to history.

        When TruLens PROD tracing is active (TRULENS_PROD_ENABLED=true), the
        invocation is recorded inside a TruChain context so every turn is logged
        to the TruLens DB for quality monitoring.

        Yields:
            Answer text chunks as they arrive from the LLM.
        """
        history = self._get_history_fn(session_id)
        answer_chunks: list[str] = []

        # Use TruChain context manager if PROD tracing is active
        tru_chain = None
        try:
            from chatbot.eval.prod_tracing import get_tru_chain
            tru_chain = get_tru_chain()
        except Exception:
            pass

        def _stream_chain():
            yield from self.lcel_chain.stream({"input": question, "chat_history": history})

        if tru_chain is not None:
            with tru_chain:
                for chunk in _stream_chain():
                    answer_chunks.append(chunk)
                    yield chunk
        else:
            for chunk in _stream_chain():
                answer_chunks.append(chunk)
                yield chunk

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content="".join(answer_chunks)))

