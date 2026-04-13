from __future__ import annotations

from pathlib import Path
import unittest

from kayak_bridge.cache_paths import REPO_ROOT


LEGACY_STAGE_COMPATIBILITY_TOKENS = (
    "Stage2Operator",
    "stage2_operator_kind",
    "search_plan_with_stage2_operator",
    "exact_stage_kind",
    "reranker_kind",
)

# These files currently own the compatibility boundary. Any new production file
# that starts using the legacy combined stage naming should fail this test until
# we explicitly decide that it belongs at the compatibility edge too.
LEGACY_STAGE_COMPATIBILITY_ALLOWLIST = {
    Path("kayak/__init__.mojo"),
    Path("kayak/benchmarks/search_plan_semantics_json.mojo"),
    Path("kayak/planning/__init__.mojo"),
    Path("kayak/planning/json.mojo"),
    Path("kayak/planning/search_plan.mojo"),
    Path("kayak/planning/stage2_operator.mojo"),
    Path("kayak/service/json.mojo"),
    Path("kayak/service/runtime.mojo"),
    Path("kayak/service/search_contracts.mojo"),
    Path("python/kayak/__init__.py"),
    Path("python/kayak_bridge/__init__.py"),
    Path("python/kayak_bridge/late_ops.py"),
    Path("python/kayak_bridge/search_plan.py"),
    Path("python/kayak_bridge/stage2_operator.py"),
}

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
    return tuple(
        token
        for token in LEGACY_STAGE_COMPATIBILITY_TOKENS
        if token in text
    )


class StageSemanticGuardrailTests(unittest.TestCase):
    def test_legacy_stage_compatibility_is_quarantined_to_allowlist(self) -> None:
        violations: list[str] = []

        for relative_path in _iter_production_source_files():
            matching_tokens = _matching_legacy_tokens(relative_path)
            if not matching_tokens:
                continue
            if relative_path in LEGACY_STAGE_COMPATIBILITY_ALLOWLIST:
                continue
            violations.append(
                f"{relative_path}: {', '.join(matching_tokens)}"
            )

        self.assertEqual(
            violations,
            [],
            "legacy combined stage compatibility leaked outside the allowlisted boundary",
        )

    def test_compatibility_allowlist_stays_minimal_and_real(self) -> None:
        missing_files = [
            str(relative_path)
            for relative_path in sorted(LEGACY_STAGE_COMPATIBILITY_ALLOWLIST)
            if not (REPO_ROOT / relative_path).exists()
        ]
        self.assertEqual(missing_files, [])

        stale_allowlist_entries = [
            str(relative_path)
            for relative_path in sorted(LEGACY_STAGE_COMPATIBILITY_ALLOWLIST)
            if (REPO_ROOT / relative_path).exists()
            and not _matching_legacy_tokens(relative_path)
        ]
        self.assertEqual(
            stale_allowlist_entries,
            [],
            "remove allowlist entries that no longer own compatibility-stage naming",
        )


if __name__ == "__main__":
    unittest.main()
