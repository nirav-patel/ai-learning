"""infrastructure/env_loader.py — load .env file once at startup.

Call load_env() at the top of main.py before constructing AppConfig so every
os.getenv() default_factory picks up values from the .env file.

The .env file is optional — no-op in CI environments that inject vars directly.
"""
from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_env(env_file: str | Path | None = None) -> None:
    """Load a .env file into os.environ (does not override existing vars).

    Args:
        env_file: Path to the .env file. Defaults to <project_root>/.env.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    path = Path(env_file) if env_file else _PROJECT_ROOT / ".env"
    if path.exists():
        load_dotenv(dotenv_path=path, override=False)
