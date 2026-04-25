from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

from kayak_bridge.cache_paths import REPO_ROOT


SOURCES_PATH = REPO_ROOT / "python" / "kayak.egg-info" / "SOURCES.txt"
RUNTIME_BRIDGE_MODULE_STEMS = (
    "__init__",
    "api_types",
    "array_conversions",
    "backend_dispatch",
    "backend_info",
    "batch_dispatch",
    "bundled_mojopkg_metadata",
    "cache_paths",
    "candidate_generator",
    "candidate_stage",
    "clause_text",
    "dtypes",
    "late_documents",
    "late_index",
    "late_ops",
    "late_query",
    "late_query_batch",
    "late_scores",
    "layouts",
    "mojo_bridge_info",
    "mojo_exact_cpu",
    "mojo_payload_cache",
    "mojo_payloads",
    "plaid_approx",
    "planned_search",
    "prepared_index_cache",
    "prepared_index_storage_artifact",
    "reference_maxsim",
    "reference_scoring_semantics",
    "search_plan",
    "search_stage_profile",
    "stage2_reference_operator",
    "stage3_verifier_operator",
    "stage_artifact_materialization",
)


def _tracked_paths(prefix: str, suffixes: tuple[str, ...]) -> tuple[str, ...]:
    output = subprocess.check_output(
        ["git", "ls-files", prefix],
        cwd=REPO_ROOT,
        text=True,
    )
    return tuple(
        line
        for line in output.splitlines()
        if line and Path(line).suffix in suffixes
    )


def _runtime_bridge_python_paths() -> tuple[str, ...]:
    return tuple(
        f"python/kayak_bridge/{module_stem}.py"
        for module_stem in RUNTIME_BRIDGE_MODULE_STEMS
    )


class SdistManifestTests(unittest.TestCase):
    def test_egg_info_sources_cover_required_python_and_mojo_inputs(self) -> None:
        subprocess.run(
            [sys.executable, "setup.py", "egg_info"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        sources = set(SOURCES_PATH.read_text(encoding="utf-8").splitlines())
        required_paths = (
            _tracked_paths("python/kayak", (".py", ".md", ".typed"))
            + _runtime_bridge_python_paths()
            + _tracked_paths("python/kayak_bridge", (".mojo",))
            + _tracked_paths("kayak", (".mojo",))
        )

        missing = tuple(path for path in required_paths if path not in sources)
        self.assertEqual(
            missing,
            (),
            f"SOURCES.txt is missing build inputs: {missing}",
        )

    def test_egg_info_sources_exclude_internal_engine_package(self) -> None:
        subprocess.run(
            [sys.executable, "setup.py", "egg_info"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        sources = set(SOURCES_PATH.read_text(encoding="utf-8").splitlines())
        leaked = tuple(
            path for path in sources if path.startswith("python/kayak_engine/")
        )
        self.assertEqual(
            leaked,
            (),
            f"SOURCES.txt must not ship internal engine files: {leaked}",
        )

    def test_egg_info_sources_exclude_internal_bridge_helpers(self) -> None:
        subprocess.run(
            [sys.executable, "setup.py", "egg_info"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        sources = set(SOURCES_PATH.read_text(encoding="utf-8").splitlines())
        bridge_python_paths = {
            f"python/kayak_bridge/{path.name}"
            for path in (REPO_ROOT / "python" / "kayak_bridge").glob("*.py")
        }
        runtime_paths = set(_runtime_bridge_python_paths())
        leaked = tuple(sorted((bridge_python_paths - runtime_paths) & sources))
        self.assertEqual(
            leaked,
            (),
            f"SOURCES.txt must not ship internal bridge helpers: {leaked}",
        )


if __name__ == "__main__":
    unittest.main()
