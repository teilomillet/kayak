from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from kayak_bridge.cache_paths import REPO_ROOT


SCRIPT_PATH = REPO_ROOT / "python" / "scripts" / "bench_fastplaid_speed_track.py"
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.append(str(SCRIPT_PATH.parent))

from bench_fastplaid_speed_track import (
    build_pairwise_rows,
    fastplaid_result_positions,
    mean_recall_at_k,
    parse_engine_list,
)
from kayak_bridge.plaid_approx import KayakPlaidApproxConfig, KayakPlaidApproxIndex


class FastPlaidSpeedTrackTests(unittest.TestCase):
    def test_parse_engine_list_requires_exact_reference_for_fastplaid(self) -> None:
        self.assertEqual(
            parse_engine_list("kayak_exact,fastplaid"),
            ("kayak_exact", "fastplaid"),
        )
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "exact reference"):
            parse_engine_list("fastplaid")

    def test_mean_recall_at_k_compares_against_exact_topk(self) -> None:
        recall = mean_recall_at_k(
            candidate_positions_by_query=((3, 2, 1), (9, 8, 7)),
            reference_positions_by_query=((1, 2, 4), (9, 0, 7)),
            k=3,
        )

        self.assertAlmostEqual(recall, (2.0 / 3.0 + 2.0 / 3.0) / 2.0)

    def test_fastplaid_result_positions_accepts_tuple_and_dict_hits(self) -> None:
        positions = fastplaid_result_positions(
            [
                [(3, 12.0), {"document_index": 4, "score": 11.0}],
                [{"doc_index": 1}, (0, 9.0)],
            ]
        )

        self.assertEqual(positions, ((3, 4), (1, 0)))

    def test_pairwise_rows_report_speed_and_recall_against_kayak(self) -> None:
        rows = build_pairwise_rows(
            [
                {
                    "system_name": "kayak_mojo_exact_cpu",
                    "engine": "kayak",
                    "status": "ok",
                    "query_batch_mean_seconds": 0.02,
                    "query_qps": 100.0,
                    "index_bytes": 400,
                },
                {
                    "system_name": "fastplaid",
                    "engine": "fastplaid",
                    "status": "ok",
                    "query_batch_mean_seconds": 0.01,
                    "query_qps": 200.0,
                    "index_bytes": 100,
                    "recall_at_k_vs_kayak_exact": 0.75,
                },
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["candidate"], "fastplaid")
        self.assertAlmostEqual(
            rows[0]["query_batch_seconds_ratio_vs_baseline"],
            0.5,
        )
        self.assertAlmostEqual(rows[0]["query_qps_ratio_vs_baseline"], 2.0)
        self.assertAlmostEqual(rows[0]["recall_at_k_vs_baseline"], 0.75)
        self.assertAlmostEqual(rows[0]["index_bytes_ratio_vs_baseline"], 0.25)

    def test_kayak_plaid_approx_returns_exact_reranked_candidates(self) -> None:
        documents = np.zeros((3, 2, 128), dtype=np.float32)
        documents[0, 0, 0] = 1.0
        documents[0, 1, 0] = 0.9
        documents[0, 1, 1] = 0.1
        documents[1, 0, 1] = 1.0
        documents[1, 1, 0] = 0.1
        documents[1, 1, 1] = 0.9
        documents[2, 0, 0] = 1.0
        documents[2, 0, 1] = 1.0
        documents[2, 1, 0] = 0.8
        documents[2, 1, 1] = 0.8
        queries = np.zeros((1, 2, 128), dtype=np.float32)
        queries[0, 0, 0] = 1.0
        queries[0, 1, 0] = 1.0
        queries[0, 1, 1] = 1.0
        index = KayakPlaidApproxIndex.build(
            doc_ids=("doc-a", "doc-b", "doc-c"),
            documents=documents,
            config=KayakPlaidApproxConfig(
                centroid_count=3,
                centroids_per_query_vector=2,
                candidate_k=2,
            ),
            final_k=2,
        )

        positions = index.search_batch_positions(queries, final_k=2)

        self.assertEqual(positions[0][0], 2)
        self.assertEqual(len(positions[0]), 2)

    def test_kayak_plaid_i8_scores_explicit_candidate_positions(self) -> None:
        documents = np.zeros((3, 2, 128), dtype=np.float32)
        documents[0, 0, 0] = 1.0
        documents[0, 1, 0] = 0.9
        documents[1, 0, 1] = 1.0
        documents[1, 1, 1] = 0.9
        documents[2, 0, 0] = 1.0
        documents[2, 0, 1] = 1.0
        documents[2, 1, 0] = 0.8
        documents[2, 1, 1] = 0.8
        queries = np.zeros((1, 2, 128), dtype=np.float32)
        queries[0, 0, 0] = 1.0
        queries[0, 1, 0] = 1.0
        queries[0, 1, 1] = 1.0
        index = KayakPlaidApproxIndex.build(
            doc_ids=("doc-a", "doc-b", "doc-c"),
            documents=documents,
            config=KayakPlaidApproxConfig(
                centroid_count=3,
                centroids_per_query_vector=2,
                candidate_k=2,
                payload="i8",
            ),
            final_k=2,
        )

        candidate_positions = index.i8_candidate_positions_batch(queries)
        scores = index.i8_score_candidate_positions_batch(
            queries,
            candidate_positions,
        )
        ranked_offsets = sorted(
            range(len(candidate_positions[0])),
            key=lambda offset: scores[0][offset],
            reverse=True,
        )
        ranked_positions = tuple(
            candidate_positions[0][offset] for offset in ranked_offsets
        )

        self.assertEqual(len(candidate_positions[0]), 2)
        self.assertEqual(len(scores[0]), 2)
        self.assertEqual(
            ranked_positions,
            index.search_batch_positions(queries, final_k=2)[0],
        )

    def test_kayak_plaid_i8_unordered_candidate_positions_keep_set(self) -> None:
        rng = np.random.default_rng(11)
        documents = rng.normal(size=(8, 4, 128)).astype(np.float32)
        queries = rng.normal(size=(2, 3, 128)).astype(np.float32)
        index = KayakPlaidApproxIndex.build(
            doc_ids=tuple(f"doc-{index}" for index in range(8)),
            documents=documents,
            config=KayakPlaidApproxConfig(
                centroid_count=8,
                centroids_per_query_vector=3,
                candidate_k=4,
                payload="i8",
            ),
            final_k=2,
        )

        ordered = index.i8_candidate_positions_batch(queries)
        unordered = index.i8_candidate_positions_batch_unordered(queries)

        self.assertEqual(len(unordered), len(ordered))
        for ordered_row, unordered_row in zip(ordered, unordered, strict=True):
            self.assertEqual(set(unordered_row), set(ordered_row))

    def test_kayak_plaid_i8_full_window_candidate_positions_skip_proxy_sort(
        self,
    ) -> None:
        documents = np.zeros((3, 2, 128), dtype=np.float32)
        documents[0, 0, 0] = 1.0
        documents[1, 0, 1] = 1.0
        documents[2, 0, 0] = 1.0
        documents[2, 0, 1] = 1.0
        queries = np.zeros((1, 2, 128), dtype=np.float32)
        queries[0, 0, 0] = 1.0
        queries[0, 1, 1] = 1.0
        index = KayakPlaidApproxIndex.build(
            doc_ids=("doc-a", "doc-b", "doc-c"),
            documents=documents,
            config=KayakPlaidApproxConfig(
                centroid_count=3,
                centroids_per_query_vector=2,
                candidate_k=3,
                payload="i8",
            ),
            final_k=2,
        )

        candidate_positions = index.i8_candidate_positions_batch(queries)
        scores = index.i8_score_candidate_positions_batch(
            queries,
            candidate_positions,
        )
        ranked_offsets = sorted(
            range(len(candidate_positions[0])),
            key=lambda offset: scores[0][offset],
            reverse=True,
        )

        self.assertEqual(candidate_positions, ((0, 1, 2),))
        self.assertEqual(
            tuple(candidate_positions[0][offset] for offset in ranked_offsets[:2]),
            index.search_batch_positions(queries, final_k=2)[0],
        )

    def test_kayak_plaid_i8_full_window_candidate_positions_batch_rows(
        self,
    ) -> None:
        documents = np.zeros((3, 2, 128), dtype=np.float32)
        queries = np.zeros((2, 1, 128), dtype=np.float32)
        index = KayakPlaidApproxIndex.build(
            doc_ids=("doc-a", "doc-b", "doc-c"),
            documents=documents,
            config=KayakPlaidApproxConfig(
                centroid_count=3,
                centroids_per_query_vector=2,
                candidate_k=3,
                payload="i8",
            ),
            final_k=2,
        )

        candidate_positions = index.i8_candidate_positions_batch(queries)

        self.assertEqual(candidate_positions, ((0, 1, 2), (0, 1, 2)))

    def test_kayak_plaid_i8_full_window_ties_keep_lower_positions(
        self,
    ) -> None:
        documents = np.zeros((3, 1, 128), dtype=np.float32)
        queries = np.zeros((1, 1, 128), dtype=np.float32)
        index = KayakPlaidApproxIndex.build(
            doc_ids=("doc-a", "doc-b", "doc-c"),
            documents=documents,
            config=KayakPlaidApproxConfig(
                centroid_count=3,
                centroids_per_query_vector=2,
                candidate_k=3,
                payload="i8",
            ),
            final_k=3,
        )

        self.assertEqual(
            index.search_batch_positions(queries, final_k=3)[0],
            (0, 1, 2),
        )

    def test_kayak_plaid_i8_payload_snapshot_exposes_flat_buffers(self) -> None:
        documents = np.zeros((2, 2, 128), dtype=np.float32)
        documents[0, 0, 0] = 1.0
        documents[0, 1, 1] = -0.5
        documents[1, 0, 2] = 0.25
        documents[1, 1, 3] = -1.0
        index = KayakPlaidApproxIndex.build(
            doc_ids=("doc-a", "doc-b"),
            documents=documents,
            config=KayakPlaidApproxConfig(
                centroid_count=2,
                centroids_per_query_vector=1,
                candidate_k=2,
                payload="i8",
            ),
            final_k=2,
        )

        snapshot = index.i8_payload_snapshot()

        self.assertEqual(snapshot.doc_offsets.tolist(), [0, 2, 4])
        self.assertEqual(snapshot.token_codes.shape, (4 * 128,))
        self.assertEqual(snapshot.token_scales.shape, (4,))
        self.assertEqual(snapshot.token_codes.dtype, np.int8)
        self.assertEqual(snapshot.token_scales.dtype, np.float32)
        self.assertEqual(snapshot.centroid_token_indices.shape, (2,))
        self.assertEqual(snapshot.centroid_doc_offsets.shape, (3,))
        self.assertEqual(snapshot.centroid_doc_indices.dtype, np.int64)
        self.assertEqual(snapshot.centroid_count, 2)
        self.assertGreaterEqual(snapshot.posting_count, 2)
        self.assertGreater(
            snapshot.candidate_generation_byte_counts()["centroid_doc_indices"],
            0,
        )
        self.assertGreaterEqual(int(snapshot.token_codes.max()), 0)
        self.assertLessEqual(int(snapshot.token_codes.min()), 0)

    def test_script_emits_kayak_only_smoke_report(self) -> None:
        temp_root = Path(tempfile.mkdtemp())
        output_path = temp_root / "summary.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "python")

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--engines",
                "kayak_exact",
                "--document-count",
                "6",
                "--document-vector-count",
                "2",
                "--query-count",
                "2",
                "--query-vector-count",
                "2",
                "--vector-dim",
                "8",
                "--top-k",
                "2",
                "--warmup-iterations",
                "0",
                "--measurement-iterations",
                "1",
                "--kayak-backend",
                "numpy_reference",
                "--index-root",
                str(temp_root / "indexes"),
                "--overwrite-index-root",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=REPO_ROOT,
        )

        self.assertIn('"benchmark": "fastplaid_speed_track"', result.stdout)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["shape"]["document_count"], 6)
        self.assertEqual(payload["shape"]["document_vector_count"], 2)
        self.assertEqual(payload["systems"][0]["system_name"], "kayak_numpy_reference")
        self.assertEqual(payload["systems"][0]["status"], "ok")
        self.assertEqual(payload["systems"][0]["recall_at_k_vs_kayak_exact"], 1.0)


if __name__ == "__main__":
    unittest.main()
