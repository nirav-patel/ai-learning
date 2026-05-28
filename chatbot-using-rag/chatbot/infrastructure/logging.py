"""infrastructure/logging.py — configure application-wide structured logging.

Call configure_logging() once at the start of run_app() so every module that
does logging.getLogger(__name__) inherits the same handler and format.

Log level is controlled by the LOG_LEVEL environment variable (default: INFO).
"""
from __future__ import annotations

import logging
import os
import sys


def configure_logging() -> None:
    """Set up the root logger. Idempotent — safe to call multiple times."""
    if logging.root.handlers:
        return

    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.root.setLevel(level)
    logging.root.addHandler(handler)

    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "gradio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
