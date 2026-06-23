"""
CLI demo for the email assistant (LangChain + LangGraph + LangMem + Bedrock).

Runs three sample emails through the assistant and prints triage decisions and
agent actions. Memory persists to ./memory_db between runs — re-running shows
prior facts surfaced via search_memory.

Usage:
    python main.py
    python main.py --log-level DEBUG
"""
from __future__ import annotations

import argparse
import logging

from email_assistant import EmailAssistant
from memory_store import load_store, save_store

# ── Sample emails (from the DeepLearning.AI notebook) ─────────────────────────

MARKETING_EMAIL = {
    "author": "Marketing Team <marketing@amazingdeals.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "🔥 EXCLUSIVE OFFER: Limited Time Discount on Developer Tools! 🔥",
    "email_thread": """Dear Valued Developer,

Don't miss out on this INCREDIBLE opportunity!

🚀 For a LIMITED TIME ONLY, get 80% OFF on our Premium Developer Suite!

💰 Regular Price: $999/month
🎉 YOUR SPECIAL PRICE: Just $199/month!

Click here to claim your discount: https://amazingdeals.com/special-offer

Best regards,
Marketing Team
---
To unsubscribe, click here
""",
}

NOTIFY_EMAIL = {
    "author": "Build System <ci@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Nightly build #482 completed successfully",
    "email_thread": """Hi John,

The nightly build #482 for the authentication service completed successfully.
All 1,243 tests passed and the artifact has been published to the staging registry.

No action is required.

— CI
""",
}

RESPOND_EMAIL = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
}

SAMPLES = [
    ("Marketing blast", MARKETING_EMAIL),
    ("CI notification", NOTIFY_EMAIL),
    ("Direct question", RESPOND_EMAIL),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Email assistant demo")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    store = load_store()
    assistant = EmailAssistant(store)

    respond_result = None
    for label, email in SAMPLES:
        print(f"\n{'=' * 70}\n{label}: {email['subject']}\n{'=' * 70}")
        result = assistant.run(email)
        print(f"\nClassification: {result.classification.upper()}")
        if result.reasoning:
            print(f"Reasoning: {result.reasoning}")
        if result.answer:
            print(f"\nAgent response:\n{result.answer}")
        if result.tool_calls:
            print("\nTool calls:")
            for tc in result.tool_calls:
                print(f"  • {tc['name']}({tc['args']})")
        if result.classification == "respond":
            respond_result = result

    # ── Procedural memory demo (lesson_5) ─────────────────────────────────────
    # Simulate human feedback on the RESPOND email and update the procedural prompts.
    if respond_result:
        feedback = "Always sign your emails `John Doe`"
        print(f"\n{'=' * 70}")
        print(f"Procedural memory update — feedback: \"{feedback}\"")
        print("=" * 70)
        changed = assistant.update_prompts(
            messages=respond_result.messages,
            feedback=feedback,
        )
        if changed:
            for name, new_prompt in changed.items():
                print(f"  Updated [{name}]: {new_prompt[:120]}{'...' if len(new_prompt) > 120 else ''}")
        else:
            print("  No prompts changed.")

    save_store(store)
    print("\n✅ Memory saved.")


if __name__ == "__main__":
    main()
