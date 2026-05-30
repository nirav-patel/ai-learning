"""chatbot/eval/run_evaluation.py — TruLens RAG Triad offline evaluation.

Measures three RAG quality metrics using the RAG Triad:
  • Answer Relevance  — Is the answer relevant to the question?
  • Context Relevance — Is each retrieved chunk relevant to the question?
  • Groundedness      — Is the answer grounded in the retrieved context?

Uses AWS Bedrock as the LLM-as-judge feedback provider (same credentials as
the chatbot — no extra API keys required).

Usage
─────
    pip install -e ".[eval]"

    python -m chatbot.eval.run_evaluation
    python -m chatbot.eval.run_evaluation --dashboard         # + Streamlit UI on :8501
    python -m chatbot.eval.run_evaluation --reset             # clear DB before run
    python -m chatbot.eval.run_evaluation --app-version v2    # label this run
    python -m chatbot.eval.run_evaluation --dashboard-only    # just open dashboard
    python -m chatbot.eval.run_evaluation --results-only      # print past results
    python -m chatbot.eval.run_evaluation --judge-model ID    # override judge model
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import warnings
from pathlib import Path

import nest_asyncio

nest_asyncio.apply()

# ── Logging (minimal) ────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))
logger.addHandler(_h)

_DASHBOARD_URL_MSG = "Dashboard -> http://localhost:8501"

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from chatbot.infrastructure.env_loader import load_env  # noqa: E402

load_env()


# ── Evaluation question bank ─────────────────────────────────────────────────

def _load_questions() -> list[str]:
    """Load evaluation questions from eval_questions.toml."""
    import tomllib
    toml_path = Path(__file__).parent / "eval_questions.toml"
    with open(toml_path, "rb") as fh:
        data = tomllib.load(fh)
    questions: list[str] = []
    for section in ("easy", "medium", "complex"):
        questions.extend(data.get(section, {}).get("questions", []))
    if not questions:
        raise ValueError(f"No questions found in {toml_path}.")
    return questions


EVAL_QUESTIONS: list[str] = _load_questions()


# ── Infrastructure bootstrap ─────────────────────────────────────────────────

def _build_eval_chain(
    backend: str | None = None,
    sentence_window_rerank_enabled: bool | None = None,
):
    """Build the RAG LCEL chain for evaluation.

    Returns:
        (eval_chain, config): the LCEL Runnable and the AppConfig.
    """
    from chatbot.config import AppConfig
    from chatbot.providers.embeddings import make_embeddings
    from chatbot.providers.llm import make_llm
    from chatbot.state import AppState
    from chatbot.storage import make_vector_store

    config = AppConfig()
    if backend:
        config.retrieval_backend = backend
    if sentence_window_rerank_enabled is not None:
        config.sentence_window_rerank_enabled = sentence_window_rerank_enabled

    state = AppState()
    state.vector_store = make_vector_store(config)
    embeddings = make_embeddings(config)

    logger.info(
        "Initialising vector store — backend=%s, model=%s",
        config.retrieval_backend,
        config.llm_model_id,
    )
    retriever = state.vector_store.initialise(config, embeddings)
    if retriever is None:
        raise RuntimeError(
            "No documents indexed. Add PDFs to data-sources/ and restart."
        )

    llm = make_llm(config)

    from chatbot.pipeline import RAGPipeline

    pipeline = RAGPipeline(retriever, llm, get_history_fn=lambda _: [])
    return pipeline.lcel_chain, config


# ── Bedrock provider for TruLens ─────────────────────────────────────────────

def _make_bedrock_provider(config, judge_model_id: str | None = None):
    """Create a TruLens Bedrock provider that works with cross-region inference profiles.

    AWS cross-region inference requires the full profile ID (e.g. "us.anthropic.claude-...")
    but TruLens routes on the base model ID (e.g. "anthropic.claude-...").
    This minimal subclass bridges the gap and increases the connection pool to
    avoid "Connection pool is full, discarding connection" warnings.
    """
    import json as _json

    import boto3
    from botocore.config import Config as BotoConfig
    from trulens.providers.bedrock import Bedrock

    profile_id = judge_model_id or config.llm_model_id
    base_model_id = re.sub(r"^[a-z]{2}\.", "", profile_id)

    # Match pool size to max_workers in _compute_feedbacks (default: 4).
    boto_config = BotoConfig(max_pool_connections=25)
    client = boto3.client(
        "bedrock-runtime",
        region_name=config.aws_region,
        config=boto_config,
    )

    class _BedrockCrossRegion(Bedrock):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            _orig = self.endpoint.client.invoke_model

            def _patched(**kw):
                kw["modelId"] = profile_id
                if "body" in kw:
                    try:
                        body = _json.loads(kw["body"])
                        if "temperature" in body and "top_p" in body:
                            del body["top_p"]
                        kw["body"] = _json.dumps(body)
                    except (ValueError, TypeError):
                        pass
                return _orig(**kw)

            self.endpoint.client.invoke_model = _patched

    return _BedrockCrossRegion(
        model_id=base_model_id, region_name=config.aws_region, client=client,
    )


# ── Feedback functions (RAG Triad) ───────────────────────────────────────────

def _build_feedbacks(config, judge_model_id: str | None = None):
    """Build three RAG Triad feedback metrics using Bedrock as the judge."""
    import numpy as np

    from trulens.core import Metric

    provider = _make_bedrock_provider(config, judge_model_id)

    # The LCEL chain input is {"input": str, "chat_history": list}.
    # Wrappers extract the question string for provider methods.
    def _answer_relevance(prompt, response):
        q = prompt["input"] if isinstance(prompt, dict) else str(prompt)
        return provider.relevance_with_cot_reasons(prompt=q, response=response)

    def _context_relevance(question, context):
        q = question["input"] if isinstance(question, dict) else str(question)
        return provider.context_relevance_with_cot_reasons(question=q, context=context)

    def _groundedness(source, statement):
        return provider.groundedness_measure_with_cot_reasons(
            source=source, statement=statement
        )

    # Answer Relevance: question → answer
    f_answer_relevance = (
        Metric(implementation=_answer_relevance, name="Answer Relevance")
        .on_input_output()
    )

    # Context Relevance: question vs each retrieved chunk (averaged)
    f_context_relevance = (
        Metric(
            implementation=_context_relevance,
            name="Context Relevance",
            agg=np.mean,
        )
        .on_input()
        .on_context(arg="context", collect_list=False)
    )

    # Groundedness: all context chunks vs answer
    f_groundedness = (
        Metric(
            implementation=_groundedness,
            name="Groundedness",
            agg=np.mean,
        )
        .on_context(arg="source", collect_list=True)
        .on_output()
    )

    return [f_answer_relevance, f_context_relevance, f_groundedness]


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    app_version: str = "v1",
    reset: bool = False,
    dashboard: bool = True,
    dashboard_only: bool = False,
    results_only: bool = False,
    judge_model: str | None = None,
    backend: str | None = None,
    sentence_window_rerank_enabled: bool | None = None,
) -> None:
    """Run the full evaluation suite and print the leaderboard."""
    from trulens.core import TruSession

    db_path = Path(__file__).parent / "trulens.db"
    db_url = os.getenv("TRULENS_DB_URL", f"sqlite:///{db_path}")
    session = TruSession(database_url=db_url)

    if dashboard_only:
        logger.info(_DASHBOARD_URL_MSG)
        session.run_dashboard()
        return

    if results_only:
        leaderboard = session.get_leaderboard()
        print("\n" + "=" * 70)
        print("  TruLens Leaderboard  (past results)")
        print("=" * 70)
        if leaderboard is not None and not leaderboard.empty:
            print(leaderboard.to_string())
        else:
            print("  No results found. Run an evaluation first.")
        print("=" * 70)
        return

    if reset:
        logger.info("Resetting TruLens database …")
        session.reset_database()

    from trulens.apps.langchain import TruChain

    logger.info("Building RAG eval chain …")
    eval_chain, config = _build_eval_chain(
        backend=backend,
        sentence_window_rerank_enabled=sentence_window_rerank_enabled,
    )

    logger.info("Building feedback functions (judge: %s) …", judge_model or config.llm_model_id)
    feedbacks = _build_feedbacks(config, judge_model_id=judge_model)

    # Disable the background evaluator thread — we compute feedbacks
    # synchronously after all questions to avoid incomplete results.
    tru_chain = TruChain(
        eval_chain,
        app_name=f"RAGChatbot[{config.retrieval_backend}]",
        app_version=app_version,
        feedbacks=feedbacks,
        start_evaluator=False,
    )

    total = len(EVAL_QUESTIONS)
    logger.info("Running %d evaluation questions …\n", total)

    for i, question in enumerate(EVAL_QUESTIONS, 1):
        logger.info("[%02d/%d] %s", i, total, question[:80])
        with tru_chain:
            eval_chain.invoke({"input": question, "chat_history": []})

    # Flush all OTEL spans to the database, then compute all feedbacks at once.
    logger.info("Flushing spans …")
    session.force_flush()

    logger.info("Computing feedback scores (this may take a few minutes) …")
    _compute_feedbacks(tru_chain, max_workers=5)
    session.force_flush()

    # Print leaderboard
    _print_leaderboard(session, db_url)

    if dashboard:
        logger.info(_DASHBOARD_URL_MSG)
        session.run_dashboard()


def run_ab_compare(
    app_version: str = "v1",
    reset: bool = False,
    dashboard: bool = False,
    judge_model: str | None = None,
    sentence_window_rerank_enabled: bool | None = None,
) -> None:
    """Run the same question set against both retrieval backends."""
    from trulens.core import TruSession

    db_path = Path(__file__).parent / "trulens.db"
    db_url = os.getenv("TRULENS_DB_URL", f"sqlite:///{db_path}")
    session = TruSession(database_url=db_url)

    if reset:
        logger.info("Resetting TruLens database …")
        session.reset_database()

    backends = ["weaviate_langchain", "llamaindex_sentence_window"]
    for backend in backends:
        run(
            app_version=f"{app_version}-{backend}",
            reset=False,
            dashboard=False,
            dashboard_only=False,
            results_only=False,
            judge_model=judge_model,
            backend=backend,
            sentence_window_rerank_enabled=sentence_window_rerank_enabled,
        )

    _print_ab_summary(
        session=session,
        app_version_prefix=app_version,
        backends=backends,
    )

    if dashboard:
        logger.info(_DASHBOARD_URL_MSG)
        session.run_dashboard()


def _compute_feedbacks(tru_app, max_workers: int = 5) -> None:
    """Compute all pending feedback scores with bounded concurrency.

    TruLens defaults to one thread per input which can spawn 30+ threads
    and exhaust the boto3 connection pool. This wrapper caps concurrency.
    """
    from trulens.feedback.computer import compute_feedback_by_span_group

    events = tru_app.connector.get_events(
        app_name=tru_app.app_name, app_version=tru_app.app_version,
    )
    for feedback in tru_app.feedbacks:
        compute_feedback_by_span_group(
            events, feedback,
            raise_error_on_no_feedbacks_computed=False,
            max_workers=max_workers,
        )


def _print_leaderboard(session, db_url: str) -> None:
    """Print the TruLens leaderboard."""
    print("\n" + "=" * 70)
    print("  TruLens Leaderboard")
    print("=" * 70)
    leaderboard = session.get_leaderboard()
    if leaderboard is not None and not leaderboard.empty:
        print(leaderboard.to_string())
    else:
        print("  No results yet — scoring may still be in progress.")
        print(f"  DB: {db_url}")
    print("=" * 70)


def _print_ab_summary(session, app_version_prefix: str, backends: list[str]) -> None:
    """Print a compact A/B summary from the leaderboard."""
    leaderboard = session.get_leaderboard()

    print("\n" + "=" * 70)
    print("  A/B Backend Comparison")
    print("=" * 70)
    if leaderboard is None or leaderboard.empty:
        print("  No results found for A/B comparison.")
        print("=" * 70)
        return

    versions = [f"{app_version_prefix}-{backend}" for backend in backends]
    subset = leaderboard[leaderboard["app_version"].isin(versions)].copy()
    if subset.empty:
        print("  No matching app_version rows found.")
        print("=" * 70)
        return

    metric_cols = [
        col for col in ("Answer Relevance", "Context Relevance", "Groundedness")
        if col in subset.columns
    ]
    keep_cols = ["app_name", "app_version", *metric_cols]
    print(subset[keep_cols].to_string(index=False))
    print("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TruLens RAG Triad evaluation.",
    )
    parser.add_argument("--app-version", default="v1", metavar="VERSION",
                        help="Label for this evaluation run (default: v1).")
    parser.add_argument("--reset", action="store_true",
                        help="Clear the TruLens DB before running.")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch TruLens Streamlit dashboard after evaluation.")
    parser.add_argument("--dashboard-only", action="store_true", dest="dashboard_only",
                        help="Skip evaluation — just open dashboard for past results.")
    parser.add_argument("--results-only", action="store_true", dest="results_only",
                        help="Print leaderboard for past results without re-running.")
    parser.add_argument("--judge-model", default=None, metavar="MODEL_ID",
                        help="Bedrock model ID for the judge (default: same as chatbot).")
    parser.add_argument("--backend", default=None, metavar="NAME",
                        help="Override retrieval backend for this run.")
    parser.add_argument("--ab-compare", action="store_true", dest="ab_compare",
                        help="Run A/B evaluation for weaviate_langchain vs llamaindex_sentence_window.")
    parser.add_argument("--sentence-window-rerank", action="store_true", dest="sentence_window_rerank",
                        help="Enable reranker during sentence-window backend evaluation.")
    parser.add_argument("--no-sentence-window-rerank", action="store_true", dest="no_sentence_window_rerank",
                        help="Disable reranker during sentence-window backend evaluation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rerank_override = None
    if args.sentence_window_rerank:
        rerank_override = True
    if args.no_sentence_window_rerank:
        rerank_override = False

    if args.ab_compare:
        run_ab_compare(
            app_version=args.app_version,
            reset=args.reset,
            dashboard=args.dashboard,
            judge_model=args.judge_model,
            sentence_window_rerank_enabled=rerank_override,
        )
    else:
        run(
            app_version=args.app_version,
            reset=args.reset,
            dashboard=args.dashboard,
            dashboard_only=args.dashboard_only,
            results_only=args.results_only,
            judge_model=args.judge_model,
            backend=args.backend,
            sentence_window_rerank_enabled=rerank_override,
        )
