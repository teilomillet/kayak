from __future__ import annotations

import unittest
from pathlib import Path

from kayak_bridge.cache_paths import REPO_ROOT


BANNED_LEGACY_STAGE_TOKENS = (
    "Stage2Operator",
    "stage2_operator",
    "stage2_operator_kind",
    "stage2_operator_for_components",
    "SearchPlanCompatibilitySemantics",
    "search_plan_compatibility_semantics",
    "same_search_plan_compatibility_semantics",
    "exact_stage_kind",
    "reranker_kind",
)
DELETED_COMPATIBILITY_MODULES = (
    Path("kayak/planning/stage2_operator.mojo"),
    Path("python/kayak_bridge/stage2_operator.py"),
)

SCAN_ROOTS = (
    Path("kayak"),
    Path("python/kayak"),
    Path("python/kayak_bridge"),
)
SOURCE_SUFFIXES = {".mojo", ".py"}


def _iter_production_source_files() -> tuple[Path, ...]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        for path in (REPO_ROOT / root).rglob("*"):
            if path.suffix not in SOURCE_SUFFIXES:
                continue
            relative_path = path.relative_to(REPO_ROOT)
            if any(part == "__pycache__" for part in relative_path.parts):
                continue
            files.append(relative_path)
    return tuple(sorted(files))


def _matching_legacy_tokens(relative_path: Path) -> tuple[str, ...]:
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    return tuple(token for token in BANNED_LEGACY_STAGE_TOKENS if token in text)


class StageSemanticGuardrailTests(unittest.TestCase):
    def test_legacy_stage_compatibility_is_absent_from_production_source(self) -> None:
        violations: list[str] = []

        for relative_path in _iter_production_source_files():
            matching_tokens = _matching_legacy_tokens(relative_path)
            if not matching_tokens:
                continue
            violations.append(f"{relative_path}: {', '.join(matching_tokens)}")

        self.assertEqual(
            violations,
            [],
            "legacy combined-stage naming reappeared in production source",
        )

    def test_deleted_compatibility_modules_stay_deleted(self) -> None:
        stale_modules = [
            str(relative_path)
            for relative_path in DELETED_COMPATIBILITY_MODULES
            if (REPO_ROOT / relative_path).exists()
        ]
        self.assertEqual(
            stale_modules,
            [],
            "legacy compatibility modules should stay deleted",
        )


if __name__ == "__main__":
    unittest.main()
