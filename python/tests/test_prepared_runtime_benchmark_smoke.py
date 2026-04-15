from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


class PreparedRuntimeBenchmarkSmokeTests(unittest.TestCase):
    def test_hosted_prepared_runtime_backpressure_benchmark_emits_json_summary(
        self,
    ) -> None:
        script_path = (
            REPO_ROOT
            / "python"
            / "scripts"
            / "bench_hosted_prepared_exact_runtime_backpressure.py"
        )
        temp_root = Path(tempfile.mkdtemp(prefix="kayak-bench-smoke-"))
        task_path = temp_root / "tiny_task.json"
        output_path = temp_root / "hosted_prepared_runtime_backpressure.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "python")

        task = {
            "dataset_id": "dataset://tiny",
            "model_name": "unit-test-model",
            "family": "tiny_family",
            "slice_name": "tiny_slice",
            "primary_metric": "ndcg",
            "k": 2,
            "nominal_query_vector_count": 1,
            "nominal_document_vector_count": 1,
            "vector_dim": 2,
            "documents": [
                {
                    "doc_id": "doc-a",
                    "text": "alpha",
                    "vector_count": 1,
                    "vectors": [[1.0, 0.0]],
                },
                {
                    "doc_id": "doc-b",
                    "text": "beta",
                    "vector_count": 1,
                    "vectors": [[0.0, 1.0]],
                },
            ],
            "queries": [
                {
                    "query_id": "q-1",
                    "text": "alpha question",
                    "relevant_doc_ids": ["doc-a"],
                    "vector_count": 1,
                    "vectors": [[1.0, 0.0]],
                },
                {
                    "query_id": "q-2",
                    "text": "beta question",
                    "relevant_doc_ids": ["doc-b"],
                    "vector_count": 1,
                    "vectors": [[0.0, 1.0]],
                },
            ],
        }
        task_path.write_text(json.dumps(task), encoding="utf-8")

        result = subprocess.run(
            [
                "pixi",
                "run",
                "python",
                str(script_path),
                "--task",
                str(task_path),
                "--request-pool-count",
                "2",
                "--burst-sizes",
                "2",
                "--lane-counts",
                "1",
                "--worker-counts",
                "1",
                "--outstanding-counts",
                "1",
                "--max-batch-size",
                "4",
                "--max-batch-wait-ms",
                "25",
                "--repeats",
                "1",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=REPO_ROOT,
        )

        self.assertIn("Mean:", result.stdout)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["dataset_id"], "dataset://tiny")
        self.assertEqual(payload["burst_sizes"], [2])
        self.assertEqual(payload["lane_counts"], [1])
        self.assertEqual(payload["worker_counts"], [1])
        self.assertEqual(payload["outstanding_counts"], [1])
        self.assertEqual(len(payload["results"]), 1)
        row = payload["results"][0]
        self.assertEqual(row["repeat_index"], 0)
        self.assertEqual(row["burst_size"], 2)
        self.assertEqual(row["concurrency_lane_count"], 1)
        self.assertEqual(row["worker_count"], 1)
        self.assertEqual(row["requested_max_outstanding_request_count"], 1)
        self.assertEqual(
            row["accepted_request_count"] + row["rejected_request_count"],
            2,
        )
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
