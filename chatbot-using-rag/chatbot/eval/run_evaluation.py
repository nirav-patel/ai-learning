"""chatbot/eval/run_evaluation.py — TruLens RAG Triad offline evaluation.

Measures three RAG quality metrics across 20 sample AI/ML/Transformer questions:
  • Answer Relevance  — Is the answer relevant to the question?
  • Context Relevance — Is each retrieved chunk relevant to the question?
  • Groundedness      — Is the answer grounded in the retrieved context?

Uses AWS Bedrock as the LLM-as-judge feedback provider (same credentials as
the chatbot — no extra API keys required).

Usage
─────
    # Install eval extras first:
    pip install -e ".[eval]"

    # Run evaluation (AWS credentials must be configured):
    python -m chatbot.eval.run_evaluation
    python -m chatbot.eval.run_evaluation --show-records      # + per-question breakdown in console
    python -m chatbot.eval.run_evaluation --dashboard         # + Streamlit UI on :8501
    python -m chatbot.eval.run_evaluation --reset             # clear DB before run
    python -m chatbot.eval.run_evaluation --app-version v2    # label this run

    # View previous results without re-running:
    python -m chatbot.eval.run_evaluation --results-only      # leaderboard + per-question in console
    python -m chatbot.eval.run_evaluation --dashboard-only    # Streamlit UI on :8501

Prerequisites
─────────────
  - AWS credentials configured (Bedrock used for chatbot LLM + feedback scoring)
  - At least one PDF in data-sources/ (chatbot needs a corpus to search against)
  - For best scores: AI/ML/RAG-related PDFs so context is relevant to the questions

Output
──────
  - Console: leaderboard table per app version
  - SQLite DB: chatbot/eval/trulens.db  (inspect across runs, queryable)
  - With --dashboard / --dashboard-only: TruLens Streamlit UI at http://localhost:8501
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import warnings

import nest_asyncio

nest_asyncio.apply()

# ── Logging ───────────────────────────────────────────────────────────────────
# Configure root logger with NullHandler so ALL third-party logs are swallowed
# (regardless of when those libraries initialise their own child loggers).
# Our private logger writes directly to stderr and does NOT propagate to root.
logging.root.handlers.clear()
logging.root.addHandler(logging.NullHandler())
logging.root.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))
logger.addHandler(_h)

# Tell huggingface_hub and transformers to stay silent BEFORE they are imported
# (they read these env vars on first import to set their own logger levels).
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Pre-create placeholder loggers with CRITICAL level for libraries that
# add their own StreamHandlers at init time. Child loggers (e.g.
# transformers_modules.x.y.z) inherit the effective level via hierarchy,
# so WARNING records are dropped before reaching any handler.
for _pre_silence in ("transformers_modules", "weaviate-client", "sentence_transformers"):
    logging.getLogger(_pre_silence).setLevel(logging.CRITICAL)

# Suppress all Python-level warnings (C-extension warnings require -W ignore).
warnings.filterwarnings("ignore")


def _suppress_third_party_handlers() -> None:
    """Clear any StreamHandlers that library loggers added during initialisation.

    Libraries like huggingface_hub, transformers and weaviate-client add their
    own handlers to child loggers at import time, bypassing our root NullHandler.
    Calling this after model/chain initialisation removes those handlers so they
    no longer emit to stderr.
    """
    our_prefix = __name__
    for name, obj in list(logging.root.manager.loggerDict.items()):
        if name == our_prefix or name.startswith(our_prefix + "."):
            continue
        if isinstance(obj, logging.Logger):
            obj.handlers.clear()
            obj.addHandler(logging.NullHandler())
            obj.setLevel(logging.CRITICAL)
            obj.propagate = False

# ── Project root on sys.path (so we can import chatbot.*) ─────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Load .env before AppConfig reads os.getenv() ─────────────────────────────
from chatbot.infrastructure.env_loader import load_env  # noqa: E402

load_env()

# ── Evaluation question bank ──────────────────────────────────────────────────
# Questions are stored in eval_questions.toml alongside this file.
# Edit that file to add, remove, or re-categorise questions — no code changes
# needed here.  Questions run in order: easy → medium → complex.

def _load_questions() -> list[str]:
    """Load evaluation questions from eval_questions.toml (stdlib tomllib, no extra dep)."""
    import tomllib
    toml_path = Path(__file__).parent / "eval_questions.toml"
    with open(toml_path, "rb") as fh:
        data = tomllib.load(fh)
    questions: list[str] = []
    for section in ("easy", "medium", "complex"):
        questions.extend(data.get(section, {}).get("questions", []))
    if not questions:
        raise ValueError(f"No questions found in {toml_path}. Check the file format.")
    return questions


EVAL_QUESTIONS: list[str] = _load_questions()


# ── Infrastructure bootstrap ─────────────────────────────────────────────────

def _build_eval_chain(app_version: str):
    """Initialise the RAG infrastructure and return an LCEL chain ready for TruChain.

    The chain takes {"input": str, "chat_history": list} and returns str.
    For evaluation each question is asked with an empty chat_history so every
    turn is scored independently.

    Returns:
        (eval_chain, config): the LCEL Runnable and the AppConfig used.
    """
    from chatbot.config import AppConfig
    from chatbot.providers.embeddings import make_embeddings
    from chatbot.providers.llm import make_llm
    from chatbot.state import AppState

    config = AppConfig()
    state = AppState()
    embeddings = make_embeddings(config)

    logger.info(
        "Initialising vector store — llm=%s model=%s embed=%s",
        config.llm_provider, config.llm_model_id, config.embed_model_name,
    )
    retriever = state.vector_store.initialise(config, embeddings)
    if retriever is None:
        raise RuntimeError(
            "No documents indexed. "
            "Add PDF files to data-sources/ and restart, or upload via the Gradio UI."
        )

    llm = make_llm(config)

    from chatbot.pipeline import RAGPipeline

    pipeline = RAGPipeline(retriever, llm, get_history_fn=lambda _: [])
    return pipeline.lcel_chain, config


# ── Feedback functions ────────────────────────────────────────────────────────

def _build_feedbacks(config, eval_chain):
    """Build the three RAG Triad feedback functions using Bedrock as the judge LLM.

    Uses TruLens 2.8 OTEL-native Metric + Selector API (Feedback is deprecated).

    RAG Triad
    ─────────
    1. Answer Relevance  — relevance_with_cot_reasons(prompt, response)
    2. Context Relevance — context_relevance_with_cot_reasons(question, context),
                           scored per chunk then averaged
    3. Groundedness      — groundedness_measure_with_cot_reasons(source, statement),
                           all context chunks aggregated against the answer
    """
    import functools
    import re
    from typing import Any

    from trulens.core import Metric
    from trulens.core.metric.selector import Selector
    from trulens.providers.bedrock import Bedrock

    # TruLens Bedrock provider routes on model_id prefix (e.g. "anthropic.", "amazon.").
    # Cross-region inference IDs (e.g. "us.anthropic.claude-sonnet-4-6") satisfy that
    # routing rule only after the region prefix is stripped.  However, Bedrock's
    # InvokeModel API requires the *full* inference profile ID — so we need:
    #   model_id    = "anthropic.claude-sonnet-4-6"     ← for TruLens format routing
    #   profile_id  = "us.anthropic.claude-sonnet-4-6"  ← for the actual boto3 call
    #
    # Bedrock.__init__ forwards **kwargs → BedrockEndpoint → boto3.client(), so we
    # must NOT pass profile_id as a kwarg to super().__init__().  Instead we override
    # __init__ to consume profile_id first, then patch invoke_model on the created
    # endpoint client to substitute the full inference profile ID at call time.
    profile_id = config.llm_model_id  # e.g. "us.anthropic.claude-sonnet-4-6"
    base_model_id = re.sub(r"^[a-z]{2}\.", "", profile_id)  # strip "us." / "eu." / "ap."

    class _BedrockWithProfile(Bedrock):
        """Extends Bedrock to route on the base model ID while calling AWS with
        the full cross-region inference profile ID (e.g. "us.anthropic.*")."""

        def __init__(self, *args: Any, inference_profile_id: str = "", **kwargs: Any) -> None:
            # Call parent WITHOUT inference_profile_id so it doesn't reach boto3.client()
            super().__init__(*args, **kwargs)
            # Patch boto3's invoke_model to:
            #   1. swap in the cross-region inference profile ID
            #   2. remove top_p when temperature is also present — Claude 4.x rejects both
            if inference_profile_id:
                import json as _json

                client = self.endpoint.client
                _orig = client.invoke_model

                @functools.wraps(_orig)
                def _patched(**kw: Any) -> Any:
                    kw["modelId"] = inference_profile_id
                    if "body" in kw:
                        try:
                            body = _json.loads(kw["body"])
                            if "temperature" in body and "top_p" in body:
                                del body["top_p"]
                            kw["body"] = _json.dumps(body)
                        except (_json.JSONDecodeError, TypeError):
                            pass
                    return _orig(**kw)

                client.invoke_model = _patched

    provider = _BedrockWithProfile(
        model_id=base_model_id,
        inference_profile_id=profile_id,
        region_name=config.aws_region,
    )

    # Register the class on the top-level __main__ module so TruLens serialization
    # can find it (it serialises as "__main__._BedrockWithProfile").
    import sys as _sys
    _main = _sys.modules.get("__main__") or _sys.modules.get(__name__)
    if _main is not None and not hasattr(_main, "_BedrockWithProfile"):
        setattr(_main, "_BedrockWithProfile", _BedrockWithProfile)

    # The LCEL chain receives a dict input: {'input': question, 'chat_history': [...]}.
    # Selector.select_record_input() returns that full dict, so we wrap the feedback
    # functions to extract just the question string before scoring.
    def _extract_question(prompt: object) -> str:
        if isinstance(prompt, dict):
            return str(prompt.get("input", prompt))
        return str(prompt)

    def _answer_relevance(prompt: object, response: str) -> object:
        return provider.relevance_with_cot_reasons(
            prompt=_extract_question(prompt), response=response
        )

    def _context_relevance(question: object, context: str) -> object:
        return provider.context_relevance_with_cot_reasons(
            question=_extract_question(question), context=context
        )

    # Answer Relevance: question vs answer
    f_answer_relevance = (
        Metric(implementation=_answer_relevance, name="Answer Relevance")
        .on_input_output()
    )

    # Context Relevance: question vs each retrieved chunk (scored separately → averaged)
    f_context_relevance = (
        Metric(
            implementation=_context_relevance,
            name="Context Relevance",
        )
        .on({
            "question": Selector.select_record_input(),
            "context": Selector.select_context(collect_list=False),
        })
    )

    # Groundedness: all retrieved chunks (source) vs the answer (statement)
    f_groundedness = (
        Metric(
            implementation=provider.groundedness_measure_with_cot_reasons,
            name="Groundedness",
        )
        .on({
            "source": Selector.select_context(collect_list=True),
            "statement": Selector.select_record_output(),
        })
    )

    return [f_answer_relevance, f_context_relevance, f_groundedness]


# ── Main entry point ──────────────────────────────────────────────────────────

def _print_records(session) -> None:
    """Print a per-question score breakdown to the console."""
    try:
        records, feedback_cols = session.get_records_and_feedback()
    except Exception as exc:
        print(f"  Could not retrieve per-question records: {exc}")
        return

    if records is None or records.empty:
        print("  No records found.")
        return

    metric_cols = [c for c in feedback_cols if c in records.columns]
    display_cols = ["input", "output"] + metric_cols

    # Trim input/output for readability
    for col in ("input", "output"):
        if col in records.columns:
            records[col] = records[col].astype(str).str.slice(0, 80)

    available = [c for c in display_cols if c in records.columns]
    print(records[available].to_string(index=True))


def run(
    app_version: str = "v1",
    reset: bool = False,
    dashboard: bool = False,
    dashboard_only: bool = False,
    results_only: bool = False,
    show_records: bool = False,
) -> None:
    """Run the full evaluation suite and print the leaderboard.

    Args:
        app_version:    Label for this evaluation run (visible in the TruLens dashboard).
        reset:          If True, clear the TruLens DB before running.
        dashboard:      If True, launch the TruLens Streamlit dashboard after evaluation.
        dashboard_only: If True, skip evaluation and just open the dashboard for past results.
        results_only:   If True, print leaderboard + per-question scores for past results (no re-run).
        show_records:   If True, also print per-question breakdown after evaluation.
    """
    from trulens.core import TruSession

    db_path = Path(__file__).parent / "trulens.db"
    db_url = os.getenv("TRULENS_DB_URL", f"sqlite:///{db_path}")

    session = TruSession(database_url=db_url)

    if dashboard_only:
        logger.info("Opening TruLens dashboard for existing results at %s …", db_url)
        logger.info("Dashboard → http://localhost:8501")
        try:
            session.run_dashboard()
        finally:
            try:
                session.stop_evaluator()
            except Exception:
                pass
        return

    if results_only:
        print("\n" + "=" * 70)
        print("  TruLens Leaderboard  (past results)")
        print("=" * 70)
        leaderboard = session.get_leaderboard()
        if leaderboard is not None and not leaderboard.empty:
            print(leaderboard.to_string())
        else:
            print("  No results found. Run an evaluation first.")
        print("=" * 70)
        print("\n" + "=" * 70)
        print("  Per-question Scores")
        print("=" * 70)
        _print_records(session)
        print("=" * 70)
        return

    if reset:
        logger.info("Resetting TruLens database at %s …", db_url)
        session.reset_database()

    import io
    from contextlib import redirect_stdout
    from trulens.apps.langchain import TruChain

    logger.info("Building RAG eval chain …")
    with redirect_stdout(io.StringIO()):
        eval_chain, config = _build_eval_chain(app_version)

    logger.info("Building feedback functions (Bedrock judge: %s) …", config.llm_model_id)
    with redirect_stdout(io.StringIO()):
        feedbacks = _build_feedbacks(config, eval_chain)

    # Remove any StreamHandlers that huggingface_hub / transformers / weaviate /
    # TruLens registered during their initialisation — these bypass our root
    # NullHandler and emit to stderr.
    _suppress_third_party_handlers()

    # TruLens instruments the chain via unconditional print() calls — suppress them.
    # start_evaluator=False: skip the background evaluator thread.  Python 3.13
    # sets concurrent.futures.thread._shutdown=True via atexit *before* that thread
    # can finish spawning its inner ThreadPoolExecutor, causing
    # "cannot schedule new futures after interpreter shutdown".
    # We instead call compute_now() from the main thread below, which runs before
    # the interpreter shutdown sequence starts.
    with redirect_stdout(io.StringIO()):
        tru_chain = TruChain(
            eval_chain,
            app_name="RAGChatbot",
            app_version=app_version,
            feedbacks=feedbacks,
            start_evaluator=False,
        )

    total = len(EVAL_QUESTIONS)
    logger.info("Running %d evaluation questions …\n", total)

    for i, question in enumerate(EVAL_QUESTIONS, 1):
        difficulty = "easy" if i <= 6 else "medium" if i <= 14 else "complex"
        logger.info("[%02d/%d] (%s) %s", i, total, difficulty, question[:80])
        with tru_chain:
            eval_chain.invoke({"input": question, "chat_history": []})

    # Flush all OTEL spans to the DB, then compute feedbacks synchronously in
    # the main thread (avoids the Python 3.13 atexit / ThreadPoolExecutor race).
    # compute_now already parallelises per-record LLM calls within each metric.
    logger.info("Flushing spans and scoring feedbacks …")
    session.force_flush()
    tru_chain._evaluator.compute_now(record_ids=None)
    session.force_flush()  # persist computed scores

    # ── Print leaderboard ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TruLens Leaderboard")
    print("=" * 70)
    leaderboard = session.get_leaderboard()
    if leaderboard is not None and not leaderboard.empty:
        print(leaderboard.to_string())
    else:
        print("  No results yet — feedback scoring may still be in progress.")
        print(f"  Re-run or inspect the DB at: {db_url}")
    print("=" * 70)

    if show_records:
        print("\n" + "=" * 70)
        print("  Per-question Scores")
        print("=" * 70)
        _print_records(session)
        print("=" * 70)

    if dashboard:
        logger.info("Launching TruLens dashboard → http://localhost:8501 …")
        try:
            session.run_dashboard()
        finally:
            try:
                session.stop_evaluator()
            except Exception:
                pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TruLens RAG Triad evaluation on 20 AI/ML/Transformer questions.",
    )
    parser.add_argument(
        "--app-version",
        default="v1",
        metavar="VERSION",
        help="Label for this evaluation run in the TruLens dashboard (default: v1).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the TruLens SQLite DB before running (useful for clean re-runs).",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the TruLens Streamlit dashboard (http://localhost:8501) after evaluation.",
    )
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        dest="dashboard_only",
        help="Skip evaluation — just open the Streamlit dashboard to browse past results.",
    )
    parser.add_argument(
        "--results-only",
        action="store_true",
        dest="results_only",
        help="Print leaderboard + per-question scores for past results without re-running.",
    )
    parser.add_argument(
        "--show-records",
        action="store_true",
        dest="show_records",
        help="After evaluation, also print a per-question score breakdown to the console.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        app_version=args.app_version,
        reset=args.reset,
        dashboard=args.dashboard,
        dashboard_only=args.dashboard_only,
        results_only=args.results_only,
        show_records=args.show_records,
    )
