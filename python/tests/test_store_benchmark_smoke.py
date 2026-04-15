from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from kayak_bridge.cache_paths import REPO_ROOT


class StoreBenchmarkSmokeTests(unittest.TestCase):
    def test_store_load_benchmark_script_emits_json_summary(self) -> None:
        script_path = REPO_ROOT / "python" / "scripts" / "bench_store_load.py"
        output_path = Path(tempfile.mkdtemp()) / "store_benchmark.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "python")

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--store",
                "memory",
                "--document-count",
                "8",
                "--vectors-per-document",
                "4",
                "--vector-dim",
                "16",
                "--subset-size",
                "3",
                "--repeats",
                "1",
                "--warmup",
                "0",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=REPO_ROOT,
        )

        self.assertIn('"store_kind": "memory"', result.stdout)
        payload = json.loads(result.stdout[result.stdout.index("{") :])
        self.assertEqual(payload["store_kind"], "memory")
        self.assertEqual(payload["document_count"], 8)
        self.assertEqual(payload["subset_size"], 3)
        self.assertIn("mean_full_load_seconds", payload)
        self.assertIn("mean_subset_load_seconds", payload)
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)

    def test_store_search_benchmark_script_emits_json_summary(self) -> None:
        script_path = REPO_ROOT / "python" / "scripts" / "bench_store_search.py"
        output_path = Path(tempfile.mkdtemp()) / "store_search_benchmark.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "python")

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--store",
                "memory",
                "--document-count",
                "8",
                "--vectors-per-document",
                "4",
                "--vector-dim",
                "16",
                "--candidate-window-size",
                "3",
                "--repeats",
                "1",
                "--warmup",
                "0",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=REPO_ROOT,
        )

        self.assertIn('"store_kind": "memory"', result.stdout)
        payload = json.loads(result.stdout[result.stdout.index("{") :])
        self.assertEqual(payload["store_kind"], "memory")
        self.assertEqual(payload["candidate_window_size"], 3)
        self.assertIn("mean_full_exact_search_seconds", payload)
        self.assertIn("mean_candidate_exact_search_seconds", payload)
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)

    def test_store_compare_benchmark_script_emits_json_summary(self) -> None:
        script_path = REPO_ROOT / "python" / "scripts" / "bench_store_compare.py"
        output_path = Path(tempfile.mkdtemp()) / "store_compare_benchmark.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "python")

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--stores",
                "directory",
                "memory",
                "--baseline-store",
                "directory",
                "--document-count",
                "8",
                "--vectors-per-document",
                "4",
                "--vector-dim",
                "16",
                "--subset-size",
                "3",
                "--repeats",
                "1",
                "--warmup",
                "0",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=REPO_ROOT,
        )

        payload = json.loads(result.stdout[result.stdout.index("{") :])
        self.assertEqual(payload["baseline_store"], "directory")
        self.assertEqual(payload["stores"], ["directory", "memory"])
        self.assertIn("summaries", payload)
        self.assertIn("ratios", payload)
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)

    def test_directory_subset_load_benchmark_script_emits_json_summary(self) -> None:
        script_path = REPO_ROOT / "python" / "scripts" / "bench_directory_subset_load.py"
        output_path = Path(tempfile.mkdtemp()) / "directory_subset_benchmark.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "python")

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--document-count",
                "32",
                "--vectors-per-document",
                "4",
                "--vector-dim",
                "16",
                "--subset-size",
                "8",
                "--repeats",
                "1",
                "--warmup",
                "0",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=REPO_ROOT,
        )

        payload = json.loads(result.stdout[result.stdout.index("{") :])
        self.assertEqual(payload["document_count"], 32)
        self.assertEqual(payload["subset_size"], 8)
        self.assertIn("mean_subset_load_seconds", payload)
        self.assertIn("mean_legacy_reference_seconds", payload)
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
