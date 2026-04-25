from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from kayak_bridge.cache_paths import REPO_ROOT


SCRIPT_PATH = REPO_ROOT / "python" / "scripts" / "bench_fastplaid_cpu_pareto.py"
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.append(str(SCRIPT_PATH.parent))

from bench_fastplaid_cpu_pareto import (  # noqa: E402
    _fastplaid_dominated_by_kayak,
    kayak_plaid_config_presets,
    pareto_front,
    shape_presets,
)


class FastPlaidCpuParetoTests(unittest.TestCase):
    def test_shape_presets_keep_vector_counts_explicit(self) -> None:
        presets = shape_presets("smoke")

        self.assertGreaterEqual(len(presets), 2)
        for preset in presets:
            payload = preset.shape.to_json_ready()
            self.assertIn("document_vector_count_total", payload)
            self.assertIn("query_vector_count_total", payload)
            self.assertEqual(payload["vector_dim"], 128)

    def test_config_presets_are_valid_for_top10(self) -> None:
        for preset in kayak_plaid_config_presets("smoke"):
            preset.config.validate(final_k=10)

    def test_long_token_presets_isolate_candidate_window(self) -> None:
        presets = kayak_plaid_config_presets("long_token")

        self.assertEqual(
            [preset.config.candidate_k for preset in presets],
            [96, 128, 160, 192],
        )
        self.assertEqual(
            {preset.config.centroid_count for preset in presets},
            {128},
        )
        self.assertEqual(
            {preset.config.centroids_per_query_vector for preset in presets},
            {32},
        )
        shape = shape_presets("long_token")[0].shape
        self.assertEqual(shape.document_vector_count, 300)
        self.assertEqual(shape.query_vector_count, 50)

    def test_cpu_matrix_v2_covers_explicit_vector_axes(self) -> None:
        shapes = shape_presets("cpu_matrix_v2")

        self.assertEqual(
            {preset.shape.document_count for preset in shapes},
            {128, 512, 2048},
        )
        self.assertEqual(
            {preset.shape.document_vector_count for preset in shapes},
            {32, 128, 300},
        )
        self.assertEqual(
            {preset.shape.query_vector_count for preset in shapes},
            {16, 50, 96},
        )

    def test_cpu_matrix_v2_configs_include_fixed_ratio_and_full_window(self) -> None:
        shape = shape_presets("cpu_matrix_v2_smoke")[0].shape
        presets = kayak_plaid_config_presets("cpu_matrix_v2")
        rows = {
            preset.name: preset.metadata_for_shape(shape)
            for preset in presets
        }

        self.assertEqual(rows["fixed_64"]["candidate_k_policy"], "fixed")
        self.assertEqual(rows["fixed_64"]["candidate_k_effective"], 64)
        self.assertEqual(rows["ratio_10pct"]["candidate_k_policy"], "ratio")
        self.assertEqual(rows["ratio_10pct"]["candidate_k_effective"], 13)
        self.assertEqual(rows["ratio_25pct"]["candidate_k_effective"], 32)
        self.assertEqual(rows["i8_ratio_25pct"]["payload"], "i8")
        self.assertEqual(rows["i8_ratio_50pct"]["candidate_k_effective"], 64)
        self.assertEqual(rows["i8_ratio_625pct"]["candidate_k_effective"], 80)
        self.assertEqual(rows["i8_ratio_725pct"]["candidate_k_effective"], 93)
        self.assertEqual(rows["i8_ratio_75pct"]["candidate_k_effective"], 96)
        self.assertEqual(rows["full_window"]["candidate_k_policy"], "full_window")
        self.assertEqual(rows["full_window"]["candidate_k_effective"], 128)
        self.assertEqual(rows["i8_full_window"]["payload"], "i8")

    def test_pareto_front_removes_dominated_rows(self) -> None:
        rows = [
            _row("dominating", recall=0.8, qps=100.0, index_bytes=100),
            _row("dominated", recall=0.8, qps=90.0, index_bytes=100),
            _row("tradeoff", recall=0.75, qps=120.0, index_bytes=80),
        ]

        front = pareto_front(
            rows,
            maximize=("recall_at_k_vs_kayak_exact", "query_qps"),
            minimize=("index_bytes",),
        )

        self.assertEqual(
            {row["system_name"] for row in front},
            {"dominating", "tradeoff"},
        )

    def test_fastplaid_dominated_by_kayak_requires_all_pareto_axes(self) -> None:
        rows = [
            _row("fastplaid", engine="fastplaid", recall=0.6, qps=100.0, index_bytes=300),
            _row("kayak_win", recall=0.7, qps=120.0, index_bytes=200),
            _row("kayak_recall_only", recall=0.8, qps=80.0, index_bytes=200),
        ]

        self.assertTrue(_fastplaid_dominated_by_kayak(rows))

    def test_script_emits_kayak_only_pareto_smoke_report(self) -> None:
        temp_root = Path(tempfile.mkdtemp())
        output_path = temp_root / "summary.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "python")

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--engines",
                "kayak_exact,kayak_plaid",
                "--shape-set",
                "smoke",
                "--kayak-plaid-config-set",
                "smoke",
                "--warmup-iterations",
                "0",
                "--measurement-iterations",
                "1",
                "--index-root",
                str(temp_root / "indexes"),
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=REPO_ROOT,
        )

        self.assertIn('"benchmark": "fastplaid_cpu_pareto"', result.stdout)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["benchmark"], "fastplaid_cpu_pareto")
        self.assertEqual(payload["controls"]["shape_set"], "smoke")
        self.assertGreaterEqual(len(payload["shape_results"]), 2)
        self.assertGreater(len(payload["all_shape_front_rows"]), 0)


def _row(
    name: str,
    *,
    recall: float,
    qps: float,
    index_bytes: int,
    engine: str = "kayak",
) -> dict[str, object]:
    return {
        "system_name": name,
        "engine": engine,
        "status": "ok",
        "index_kind": "sampled_centroid_postings_exact_rerank",
        "recall_at_k_vs_kayak_exact": recall,
        "query_qps": qps,
        "query_batch_mean_seconds": 1.0 / qps,
        "index_bytes": index_bytes,
    }


if __name__ == "__main__":
    unittest.main()
