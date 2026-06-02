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


# ---------------------------------------------------------------------------
# Test 4: generate_research_report_with_tools
# ---------------------------------------------------------------------------

def test_generate_research_report_with_tools(report: str) -> None:
    def g():
        cases = []

        # Must be str
        t = test_case()
        if not isinstance(report, str):
            t.failed = True
            t.msg  = "generate_research_report_with_tools must return a str"
            t.want = str
            t.got  = type(report)
            return [t]
        cases.append(t)

        # Non-trivial length
        t = test_case()
        if len((report or "").strip()) <= 50:
            t.failed = True
            t.msg  = f"report text should be non-trivial (length > 50); got {len((report or '').strip())}"
            t.want = "> 50 chars"
            t.got  = len((report or "").strip())
        cases.append(t)

        return cases

    print_feedback(g())


# ---------------------------------------------------------------------------
# Test 5: reflection_and_rewrite
# ---------------------------------------------------------------------------

def test_reflection_and_rewrite(out: dict) -> None:
    def g():
        cases = []

        # Must be dict
        t = test_case()
        if not isinstance(out, dict):
            t.failed = True
            t.msg  = "reflection_and_rewrite must return a dict"
            t.want = dict
            t.got  = type(out)
            return [t]
        cases.append(t)

        # Required keys
        t = test_case()
        keys = set(out.keys())
        if not {"reflection", "revised_report"} <= keys:
            t.failed = True
            t.msg  = "dict must include keys 'reflection' and 'revised_report'"
            t.want = {"reflection", "revised_report"}
            t.got  = keys
            return [t]
        cases.append(t)

        # Values are strings
        t = test_case()
        if not isinstance(out["reflection"], str) or not isinstance(out["revised_report"], str):
            t.failed = True
            t.msg  = "'reflection' and 'revised_report' must be strings"
            t.want = "str, str"
            t.got  = (type(out["reflection"]), type(out["revised_report"]))
            return [t]
        cases.append(t)

        # Reflection must mention the four headings
        t = test_case()
        low = out["reflection"].lower()
        expected = ["strengths", "limitations", "suggestions", "opportunities"]
        has_all = all(h in low for h in expected)
        if not has_all:
            t.failed = True
            t.msg  = "reflection should mention Strengths, Limitations, Suggestions, Opportunities"
            t.want = expected
            t.got  = [h for h in expected if h in low]
        cases.append(t)

        # revised_report non-trivial
        t = test_case()
        if len(out["revised_report"].strip()) <= 50:
            t.failed = True
            t.msg  = f"revised_report should be non-trivial (length > 50); got {len(out['revised_report'].strip())}"
            t.want = "> 50 chars"
            t.got  = len(out["revised_report"].strip())
        cases.append(t)

        return cases

    print_feedback(g())


# ---------------------------------------------------------------------------
# Test 6: convert_report_to_html
# ---------------------------------------------------------------------------

def test_convert_report_to_html(html: str) -> None:
    def g():
        cases = []

        # Must be str
        t = test_case()
        if not isinstance(html, str):
            t.failed = True
            t.msg  = "convert_report_to_html must return a str"
            t.want = str
            t.got  = type(html)
            return [t]
        cases.append(t)

        # Must look like HTML
        t = test_case()
        low = (html or "").lower()
        looks_like_html = (
            "<html" in low or "</" in low or "<h1" in low or "<p" in low
        )
        if not looks_like_html:
            t.failed = True
            t.msg  = "output should look like HTML (contain <html>, <h1>, <p>, or closing tags)"
            t.want = "HTML-like string"
            t.got  = html[:120]
        cases.append(t)

        return cases

    print_feedback(g())


# ---------------------------------------------------------------------------
# Test 7: evaluate_report_sources (component-level eval)
# ---------------------------------------------------------------------------

def test_evaluate_report_sources(flag: bool, eval_report: str) -> None:
    def g():
        cases = []

        # flag must be bool
        t = test_case()
        if not isinstance(flag, bool):
            t.failed = True
            t.msg  = "evaluate_report_sources must return a bool as first element"
            t.want = bool
            t.got  = type(flag)
            return [t]
        cases.append(t)

        # eval_report must be str
        t = test_case()
        if not isinstance(eval_report, str):
            t.failed = True
            t.msg  = "evaluate_report_sources must return a str as second element"
            t.want = str
            t.got  = type(eval_report)
            return [t]
        cases.append(t)

        # eval_report must contain expected headings
        t = test_case()
        low = eval_report.lower()
        if "evaluation" not in low:
            t.failed = True
            t.msg  = "eval_report should contain 'Evaluation' heading"
            t.want = "'evaluation' in report"
            t.got  = eval_report[:120]
        cases.append(t)

        # eval_report must contain ratio/total statistics
        t = test_case()
        if "ratio" not in low and "total" not in low:
            t.failed = True
            t.msg  = "eval_report should contain ratio/total statistics"
            t.want = "'ratio' or 'total' in report"
            t.got  = eval_report[:120]
        cases.append(t)

        return cases

    print_feedback(g())
