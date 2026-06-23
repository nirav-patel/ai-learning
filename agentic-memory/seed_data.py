"""
Default profile, instructions, and episodic examples seeded on first run.

Once seeded, all entries live in the InMemoryStore (persisted to JSON) and
can be updated at runtime — profile/instructions via agent tools, examples
via add_example() after human-in-the-loop corrections.
"""

# Stored in the `profile` namespace (singleton id `user_profile`).
DEFAULT_PROFILE = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

# Stored in the `instructions` namespace (singleton id `agent_instructions`).
DEFAULT_INSTRUCTIONS = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently.",
}

# Seeded into the `examples` namespace on first run (episodic few-shots for triage).
# Format mirrors lesson_4: {"email": {author, to, subject, email_thread}, "label": str}
DEFAULT_EXAMPLES = [
    {
        "email": {
            "author": "Alice Smith <alice.smith@company.com>",
            "to": "John Doe <john.doe@company.com>",
            "subject": "Quick question about API documentation",
            "email_thread": (
                "Hi John,\n\nI was reviewing the API documentation for the new "
                "authentication service and noticed a few endpoints seem to be missing "
                "from the specs. Could you help clarify if this was intentional or if "
                "we should update the docs?\n\nSpecifically, I'm looking at:\n"
                "- /auth/refresh\n- /auth/validate\n\nThanks!\nAlice"
            ),
        },
        "label": "respond",
    },
    {
        "email": {
            "author": "Sarah Chen <sarah.chen@company.com>",
            "to": "John Doe <john.doe@company.com>",
            "subject": "Update: Backend API Changes Deployed to Staging",
            "email_thread": (
                "Hi John,\n\nJust wanted to let you know that I've deployed the new "
                "authentication endpoints to the staging environment. All tests are "
                "passing and the changes are ready for review.\n\n"
                "No immediate action needed from your side — just keeping you in the loop.\n\n"
                "Best regards,\nSarah"
            ),
        },
        "label": "ignore",
    },
]
