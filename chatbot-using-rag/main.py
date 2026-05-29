"""main.py — entry point for the RAG chatbot.

All configuration is driven by AppConfig, which reads from environment variables.
Copy .env.example to .env and set your values before running.

Run:
    cd chatbot-using-rag
    python main.py

To use a different embedding model, set EMBED_MODEL_NAME in .env or override
the AppConfig field directly below.
"""
import os
import warnings

# Tell noisy libraries to stay quiet before they are imported.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("GRPC_VERBOSITY", "NONE")   # suppress gRPC info/warn to stderr
os.environ.setdefault("GLOG_minloglevel", "3")      # suppress absl/glog to FATAL only

# Suppress all Python-level warnings from third-party libraries.
# Some libraries call filterwarnings("default") after startup, overriding our filter.
# Overriding showwarning is the belt-and-suspenders approach: even if the filter
# says "show this warning", the display function simply discards it.
warnings.filterwarnings("ignore")
warnings.showwarning = lambda *_args, **_kwargs: None

from chatbot.infrastructure.env_loader import load_env

load_env()  # load .env before AppConfig reads os.getenv() defaults

from chatbot.config import AppConfig  # noqa: E402
from chatbot.app    import run_app    # noqa: E402

config = AppConfig(
    # Override individual fields here, or configure everything via .env
    # embed_model_name = "BAAI/bge-base-en-v1.5",
    # embed_provider   = "huggingface",
    # llm_provider     = "bedrock",
    # port             = 7860,
)

if __name__ == "__main__":
    run_app(config)
