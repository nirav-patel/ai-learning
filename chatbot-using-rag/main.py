"""main.py — entry point for the RAG chatbot.

All configuration is driven by AppConfig, which reads from environment variables.
Copy .env.example to .env and set your values before running.

Run:
    cd chatbot-using-rag
    python main.py

To use a different embedding model, set EMBED_MODEL_NAME in .env or override
the AppConfig field directly below.
"""
import warnings

# Suppress DeprecationWarnings from third-party libraries (Gradio / Starlette)
# that have not yet been updated for Python 3.14 API changes. These warnings
# originate inside library code and are not actionable from our side.
warnings.filterwarnings(
    "ignore",
    message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*asyncio\.iscoroutinefunction.*",
    category=DeprecationWarning,
)

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
