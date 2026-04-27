from __future__ import annotations

import sys
import unittest

import numpy as np

from kayak_bridge.cache_paths import REPO_ROOT


SCRIPT_ROOT = REPO_ROOT / "python" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

import profile_gpu_i8_candidate_posting_traversal as traversal_script  # noqa: E402
from kayak_bridge.gpu_device_capability import (  # noqa: E402
    CommandResult,
    GPU_STATUS_AVAILABLE,
    GPU_STATUS_UNAVAILABLE,
    MojoGpuCapability,
)
from kayak_bridge.gpu_i8_address_serve_sweep import (  # noqa: E402
    STATUS_OK,
    STATUS_PARTIAL_GPU_UNAVAILABLE,
)
from kayak_bridge.gpu_i8_candidate_posting_traversal import (  # noqa: E402
    STATUS_BLOCKED_GPU_POSTING_TRAVERSAL_FAILED,
    comparison_payload,
    report_status,
    selected_posting_summary,
    summary_payload,
)
from kayak_bridge.mojo_gpu_i8_rerank import (  # noqa: E402
    MojoGpuI8SelectedPostingTraversalResult,
    _selected_posting_traversal_reference,
)
from kayak_bridge.plaid_approx import (  # noqa: E402
    KayakPlaidI8PayloadSnapshot,
    KayakPlaidI8SelectedCentroids,
)


class GpuI8CandidatePostingTraversalTests(unittest.TestCase):
    def test_traversal_result_reports_agreement(self) -> None:
        result = MojoGpuI8SelectedPostingTraversalResult(
            host_marshalling_seconds=0.1,
            extension_call_seconds=0.2,
            mojo_host_ingest_mean_seconds=0.03,
            payload_host_to_device_mean_seconds=0.04,
            selected_host_to_device_mean_seconds=0.05,
            kernel_mean_seconds=0.06,
            device_to_host_mean_seconds=0.07,
            selected_position_out_of_range_count=0,
            selected_offset_violation_count=0,
            doc_mismatch_count=0,
            doc_index_out_of_range_count=0,
            score_delta_max_abs=0.0,
            selected_centroid_count=8,
            expanded_posting_count=32,
            document_count=16,
        )

        payload = result.to_json_ready()

        self.assertTrue(payload["traversal_agreement_ok"])
        self.assertEqual(payload["selected_centroid_count"], 8)
        self.assertEqual(payload["expanded_posting_count"], 32)

    def test_selected_posting_reference_expands_docs_and_scores(self) -> None:
        payload = _payload()
        selected = _selected()

        reference = _selected_posting_traversal_reference(
            payload=payload,
            selected=selected,
        )

        self.assertEqual(reference.selected_posting_offsets.tolist(), [0, 2, 5])
        self.assertEqual(reference.expected_doc_indices.tolist(), [1, 2, 0, 2, 3])
        self.assertEqual(
            [round(float(value), 2) for value in reference.expected_scores],
            [0.5, 0.5, 0.25, 0.25, 0.25],
        )

    def test_selected_posting_summary_keeps_visit_count_explicit(self) -> None:
        summary = selected_posting_summary(payload=_payload(), selected=_selected())

        self.assertEqual(summary["selected_centroid_count"], 2)
        self.assertEqual(summary["expanded_posting_count"], 5)
        self.assertEqual(summary["expanded_postings_per_selected_centroid"], 2.5)

    def test_comparison_payload_reports_candidate_and_posting_denominators(
        self,
    ) -> None:
        comparison = comparison_payload(
            cpu_candidate_generation_mean_seconds=0.02,
            cpu_posting_accumulation_seconds=0.01,
            gpu_parsed={
                "extension_call_seconds": 0.006,
                "mojo_host_ingest_mean_seconds": 0.001,
                "payload_host_to_device_mean_seconds": 0.002,
                "selected_host_to_device_mean_seconds": 0.003,
                "kernel_mean_seconds": 0.004,
                "device_to_host_mean_seconds": 0.005,
            },
        )

        self.assertEqual(
            comparison[
                "gpu_posting_kernel_seconds_per_cpu_candidate_generation_second"
            ],
            0.2,
        )
        self.assertEqual(
            comparison[
                "gpu_posting_selected_h2d_kernel_d2h_seconds_per_cpu_posting_accumulation_second"
            ],
            1.2,
        )
        self.assertAlmostEqual(
            comparison["gpu_posting_all_measured_mean_seconds"],
            0.014,
        )

    def test_summary_reports_traversal_ratios_and_visit_count(self) -> None:
        rows = [
            _row(
                name="a",
                kind="non_full",
                kernel_ratio=0.1,
                selected_path_ratio=0.2,
                all_ratio=0.3,
                selected_path_posting_ratio=0.4,
                visits=64,
            ),
            _row(
                name="b",
                kind="full",
                kernel_ratio=0.05,
                selected_path_ratio=0.25,
                all_ratio=0.35,
                selected_path_posting_ratio=0.45,
                visits=128,
            ),
        ]

        summary = summary_payload(rows)

        self.assertEqual(summary["ok_case_count"], 2)
        self.assertEqual(summary["ok_non_full_case_count"], 1)
        self.assertEqual(summary["full_window_case_count"], 1)
        self.assertEqual(
            summary["best_kernel_vs_candidate_generation_ratio"],
            0.05,
        )
        self.assertEqual(
            summary["worst_all_measured_vs_candidate_generation_ratio"],
            0.35,
        )
        self.assertEqual(
            summary["best_selected_path_vs_posting_accumulation_ratio"],
            0.4,
        )
        self.assertEqual(summary["max_expanded_posting_count"], 128)

    def test_report_status_distinguishes_missing_gpu_from_failed_probe(self) -> None:
        unavailable = MojoGpuCapability(
            status=GPU_STATUS_UNAVAILABLE,
            probe=CommandResult(("gpu-query",), 1, "", "no gpu"),
        )
        available = MojoGpuCapability(
            status=GPU_STATUS_AVAILABLE,
            probe=CommandResult(("gpu-query",), 0, "name: test", ""),
            target_accelerator="nvidia:sm_89",
        )

        self.assertEqual(
            report_status(capability=unavailable, rows=[]),
            STATUS_PARTIAL_GPU_UNAVAILABLE,
        )
        self.assertEqual(
            report_status(capability=available, rows=[{"status": STATUS_OK}]),
            STATUS_OK,
        )
        self.assertEqual(
            report_status(capability=available, rows=[{"status": "error"}]),
            STATUS_BLOCKED_GPU_POSTING_TRAVERSAL_FAILED,
        )

    def test_traversal_script_defaults_to_shape_policy(self) -> None:
        args = traversal_script.parse_args([])

        self.assertEqual(
            traversal_script.centroid_budget_policy_from_args(args),
            "shape_rule_v0",
        )
        self.assertEqual(
            traversal_script.case_selection_from_args(args),
            {"source": "wide_topk", "case_count": 5},
        )

    def test_traversal_script_allows_static_centroid_budget(self) -> None:
        args = traversal_script.parse_args(["--centroid-budget-policy", "none"])

        self.assertIsNone(traversal_script.centroid_budget_policy_from_args(args))


def _payload() -> KayakPlaidI8PayloadSnapshot:
    return KayakPlaidI8PayloadSnapshot(
        doc_offsets=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        token_codes=np.zeros(4 * 128, dtype=np.int8),
        token_scales=np.ones(4, dtype=np.float32),
        centroid_token_indices=np.asarray([0, 1], dtype=np.int64),
        centroid_doc_offsets=np.asarray([0, 2, 5], dtype=np.int64),
        centroid_doc_indices=np.asarray([1, 2, 0, 2, 3], dtype=np.int64),
        document_count=4,
        total_vector_count=4,
        vector_dim=128,
    )


def _selected() -> KayakPlaidI8SelectedCentroids:
    return KayakPlaidI8SelectedCentroids(
        positions_by_query=((0, 1),),
        scores_by_query=((0.5, 0.25),),
        query_count=1,
        query_vector_count=1,
        centroids_per_query_vector=2,
    )


def _row(
    *,
    name: str,
    kind: str,
    kernel_ratio: float,
    selected_path_ratio: float,
    all_ratio: float,
    selected_path_posting_ratio: float,
    visits: int,
) -> dict[str, object]:
    return {
        "name": name,
        "status": STATUS_OK,
        "candidate_window_kind": kind,
        "selected_postings": {"expanded_posting_count": visits},
        "comparison": {
            "gpu_posting_kernel_seconds_per_cpu_candidate_generation_second": (
                kernel_ratio
            ),
            "gpu_posting_selected_h2d_kernel_d2h_seconds_per_cpu_candidate_generation_second": (
                selected_path_ratio
            ),
            "gpu_posting_all_measured_seconds_per_cpu_candidate_generation_second": (
                all_ratio
            ),
            "gpu_posting_selected_h2d_kernel_d2h_seconds_per_cpu_posting_accumulation_second": (
                selected_path_posting_ratio
            ),
        },
    }


if __name__ == "__main__":
    unittest.main()
