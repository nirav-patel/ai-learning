"""
Unit tests for the research agent workflow.

Each function accepts the output string produced by the corresponding
workflow step and validates it — no API calls, no file I/O needed.
Called directly from main.py after the workflow run.

Can also be run standalone if outputs are available:
    python tests.py
"""

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Minimal test_case / print_feedback helpers (mirrors dlai_grader style)
# ---------------------------------------------------------------------------

@dataclass
class test_case:
    failed: bool = False
    msg:    str  = ""
    want:   Any  = None
    got:    Any  = None


def print_feedback(cases: list[test_case]) -> None:
    for t in cases:
        if t.failed:
            print(f"  ❌ FAILED — {t.msg}")
            if t.want is not None:
                print(f"     Expected : {t.want}")
            if t.got is not None:
                print(f"     Got      : {t.got}")
        else:
            print("  ✅ PASSED")
    if not any(t.failed for t in cases):
        print("  All tests passed.\n")


# ---------------------------------------------------------------------------
# Test 1: generate_draft
# ---------------------------------------------------------------------------

def test_generate_draft(draft: str) -> None:
    def g():
        cases = []

        # Must be str
        t = test_case()
        if not isinstance(draft, str):
            t.failed = True
            t.msg  = "generate_draft must return a str"
            t.want = str
            t.got  = type(draft)
            return [t]

        # Length > 100
        t = test_case()
        if len(draft) <= 100:
            t.failed = True
            t.msg  = f"generate_draft must return text with length > 100 (got {len(draft)})"
            t.want = "> 100 chars"
            t.got  = len(draft)
        cases.append(t)
        return cases

    print_feedback(g())


# ---------------------------------------------------------------------------
# Test 2: reflect_on_draft
# ---------------------------------------------------------------------------

def test_reflect_on_draft(feedback: str) -> None:
    def g():
        cases = []

        # Must be str
        t = test_case()
        if not isinstance(feedback, str):
            t.failed = True
            t.msg  = "reflect_on_draft must return a str"
            t.want = str
            t.got  = type(feedback)
            return [t]
        cases.append(t)

        # Must be non-empty
        t = test_case()
        if len(feedback) == 0:
            t.failed = True
            t.msg  = "reflect_on_draft must return a non-empty string"
            t.want = "len > 0"
            t.got  = 0
        cases.append(t)
        return cases

    print_feedback(g())


# ---------------------------------------------------------------------------
# Test 3: revise_draft
# ---------------------------------------------------------------------------

def test_revise_draft(revised: str) -> None:
    def g():
        cases = []

        # Must be str
        t = test_case()
        if not isinstance(revised, str):
            t.failed = True
            t.msg  = "revise_draft must return a str"
            t.want = str
            t.got  = type(revised)
            return [t]

        # Length > 100
        t = test_case()
        if len(revised) <= 100:
            t.failed = True
            t.msg  = f"revise_draft must return text with length > 100 (got {len(revised)})"
            t.want = "> 100 chars"
            t.got  = len(revised)
        cases.append(t)
        return cases

    print_feedback(g())
