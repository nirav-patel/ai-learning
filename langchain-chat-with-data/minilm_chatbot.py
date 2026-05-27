"""Compatibility wrapper for the renamed BGE chatbot entrypoint.

Use bge_chatbot.py going forward. This file is kept to avoid breaking older
commands or scripts that still run minilm_chatbot.py.
"""

from bge_chatbot import config
from chat.ui import run_app


if __name__ == "__main__":
    run_app(config)
