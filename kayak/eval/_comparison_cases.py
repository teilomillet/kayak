"""Pure paired classification evidence and a bounded, readable view of its cases."""

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

from ._schema import Example


class CaseComparison(TypedDict):
    id: str
    text: str
    expected: str
    baseline_choice: str
    candidate_choice: str
    outcome: Literal["both_correct", "fixed", "regressed", "both_wrong"]


def compare_cases(
    examples: Sequence[Example], baseline: Mapping[str, str], candidate: Mapping[str, str]
) -> tuple[dict[str, int], list[CaseComparison]]:
    """Use every example in suite order, after callers verify complete comparable inputs."""
    counts = dict.fromkeys(("both_correct", "fixed", "regressed", "both_wrong"), 0)
    cases: list[CaseComparison] = []
    for example in examples:
        before, after = baseline[example.id], candidate[example.id]
        outcome: Literal["both_correct", "fixed", "regressed", "both_wrong"]
        if before == example.label and after == example.label:
            outcome = "both_correct"
        elif after == example.label:
            outcome = "fixed"
        elif before == example.label:
            outcome = "regressed"
        else:
            outcome = "both_wrong"
        counts[outcome] += 1
        cases.append(
            CaseComparison(
                id=example.id,
                text=example.text,
                expected=example.label,
                baseline_choice=before,
                candidate_choice=after,
                outcome=outcome,
            )
        )
    return counts, cases


def markdown_cell(value: object) -> str:
    """Render input as text, never as Markdown links, images, or raw HTML."""
    text = " ".join(str(value).split())
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for character in ("\\", "`", "*", "_", "[", "]", "|", "~"):
        text = text.replace(character, "\\" + character)
    return text


def render_case_changes(cases: Sequence[CaseComparison]) -> str:
    """Show at most ten cases per outcome; complete text and every case stay in JSON."""
    lines = [
        "Cases in suite order; native runs use first measured attempts. Up to 10 per group;",
        "cells are shortened to 240 characters. JSON retains every case and full text.",
        "",
    ]
    for outcome, title in (
        ("regressed", "Regressions"),
        ("fixed", "Fixes"),
        ("both_wrong", "Still wrong"),
    ):
        selected = [case for case in cases if case["outcome"] == outcome]
        lines.extend([f"#### {title} ({len(selected)})", ""])
        if not selected:
            lines.extend(["None.", ""])
            continue
        lines.extend(
            [
                f"Showing {min(10, len(selected))} of {len(selected)}.",
                "",
                "| ID | Text | Expected | Baseline | Candidate |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for case in selected[:10]:
            cells = [
                case[key]
                for key in ("id", "text", "expected", "baseline_choice", "candidate_choice")
            ]
            lines.append(
                "| "
                + " | ".join(
                    markdown_cell(value if len(value) <= 240 else value[:239] + "…")
                    for value in cells
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)
