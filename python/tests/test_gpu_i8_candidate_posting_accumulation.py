from __future__ import annotations

import sys
import unittest

import numpy as np

from kayak_bridge.cache_paths import REPO_ROOT


SCRIPT_ROOT = REPO_ROOT / "python" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

import profile_gpu_i8_candidate_posting_accumulation as accumulation_script  # noqa: E402
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
from kayak_bridge.gpu_i8_candidate_posting_accumulation import (  # noqa: E402
    STATUS_BLOCKED_GPU_POSTING_ACCUMULATION_FAILED,
    comparison_payload,
    report_status,
    summary_payload,
)
from kayak_bridge.mojo_gpu_i8_rerank import (  # noqa: E402
    MojoGpuI8SelectedPostingAccumulationResult,
    _selected_posting_accumulation_reference,
)
from kayak_bridge.plaid_approx import (  # noqa: E402
    KayakPlaidI8PayloadSnapshot,
    KayakPlaidI8SelectedCentroids,
)


class GpuI8CandidatePostingAccumulationTests(unittest.TestCase):
    def test_accumulation_result_reports_agreement(self) -> None:
        result = MojoGpuI8SelectedPostingAccumulationResult(
            host_marshalling_seconds=0.1,
            extension_call_seconds=0.2,
            mojo_host_ingest_mean_seconds=0.03,
            payload_host_to_device_mean_seconds=0.04,
            selected_host_to_device_mean_seconds=0.05,
            kernel_mean_seconds=0.06,
            device_to_host_mean_seconds=0.07,
            host_topk_mean_seconds=0.08,
            device_topk_kernel_mean_seconds=0.09,
            device_topk_device_to_host_mean_seconds=0.01,
            selected_position_out_of_range_count=0,
            doc_index_out_of_range_count=0,
            score_mismatch_count=0,
            score_delta_max_abs=0.0,
            topk_position_mismatch_count=0,
            device_topk_position_mismatch_count=0,
            top_k=2,
            topk_position_count=2,
            selected_centroid_count=4,
            document_score_count=4,
            document_count=4,
        )

        payload = result.to_json_ready()

        self.assertTrue(payload["accumulation_agreement_ok"])
        self.assertEqual(payload["selected_centroid_count"], 4)
        self.assertEqual(payload["document_score_count"], 4)
        self.assertEqual(payload["topk_position_count"], 2)

    def test_accumulation_reference_sums_query_vector_maxima(self) -> None:
        reference = _selected_posting_accumulation_reference(
            payload=_payload(),
            selected=_selected_two_vectors(),
        )

        self.assertEqual(reference.selected_centroid_count, 4)
        self.assertEqual(reference.document_score_count, 4)
        self.assertEqual(
            [round(float(value), 2) for value in reference.expected_document_scores],
            [1.0, 0.6, 1.25, 1.0],
        )

    def test_comparison_payload_reports_posting_denominators(self) -> None:
        comparison = comparison_payload(
            cpu_candidate_generation_mean_seconds=0.02,
            cpu_centroid_scoring_plus_selection_seconds=0.001,
            cpu_posting_accumulation_seconds=0.01,
            gpu_parsed={
                "extension_call_seconds": 0.006,
                "mojo_host_ingest_mean_seconds": 0.001,
                "payload_host_to_device_mean_seconds": 0.002,
                "selected_host_to_device_mean_seconds": 0.003,
                "kernel_mean_seconds": 0.004,
                "device_to_host_mean_seconds": 0.005,
                "host_topk_mean_seconds": 0.006,
                "device_topk_kernel_mean_seconds": 0.007,
                "device_topk_device_to_host_mean_seconds": 0.008,
            },
        )

        self.assertEqual(
            comparison[
                "gpu_posting_accumulation_kernel_seconds_per_cpu_posting_accumulation_second"
            ],
            0.4,
        )
        self.assertAlmostEqual(
            comparison[
                "gpu_posting_accumulation_all_measured_seconds_per_cpu_candidate_generation_second"
            ],
            0.7,
        )
        self.assertAlmostEqual(
            comparison["gpu_posting_accumulation_all_measured_mean_seconds"],
            0.014,
        )
        self.assertAlmostEqual(
            comparison[
                "gpu_posting_accumulation_all_measured_plus_host_topk_mean_seconds"
            ],
            0.02,
        )
        self.assertAlmostEqual(
            comparison[
                "projected_resident_payload_candidate_seconds_per_cpu_candidate_generation_second"
            ],
            0.95,
        )
        self.assertAlmostEqual(
            comparison[
                "gpu_posting_accumulation_all_measured_device_topk_mean_seconds"
            ],
            0.024,
        )
        self.assertAlmostEqual(
            comparison[
                "projected_device_topk_resident_payload_candidate_seconds_per_cpu_candidate_generation_second"
            ],
            1.15,
        )

    def test_summary_reports_accumulation_ratios(self) -> None:
        rows = [
            _row(
                kind="non_full",
                all_candidate_ratio=0.2,
                all_plus_topk_candidate_ratio=0.25,
                all_device_topk_candidate_ratio=0.22,
                projected_resident_ratio=0.3,
                projected_cold_ratio=0.35,
                projected_device_topk_resident_ratio=0.28,
                projected_device_topk_cold_ratio=0.33,
                all_posting_ratio=0.5,
                kernel_posting_ratio=0.3,
                visits=64,
            ),
            _row(
                kind="full",
                all_candidate_ratio=0.4,
                all_plus_topk_candidate_ratio=0.45,
                all_device_topk_candidate_ratio=0.42,
                projected_resident_ratio=0.5,
                projected_cold_ratio=0.55,
                projected_device_topk_resident_ratio=0.48,
                projected_device_topk_cold_ratio=0.53,
                all_posting_ratio=0.7,
                kernel_posting_ratio=0.6,
                visits=128,
            ),
        ]

        summary = summary_payload(rows)

        self.assertEqual(summary["ok_case_count"], 2)
        self.assertEqual(summary["ok_non_full_case_count"], 1)
        self.assertEqual(
            summary["best_all_measured_vs_candidate_generation_ratio"],
            0.2,
        )
        self.assertEqual(
            summary["worst_all_measured_vs_posting_accumulation_ratio"],
            0.7,
        )
        self.assertEqual(
            summary[
                "best_non_full_all_measured_plus_host_topk_vs_candidate_generation_ratio"
            ],
            0.25,
        )
        self.assertEqual(
            summary[
                "best_non_full_projected_resident_payload_candidate_vs_cpu_candidate_generation_ratio"
            ],
            0.3,
        )
        self.assertEqual(
            summary[
                "best_non_full_all_measured_device_topk_vs_candidate_generation_ratio"
            ],
            0.22,
        )
        self.assertEqual(
            summary[
                "best_non_full_projected_device_topk_resident_payload_candidate_vs_cpu_candidate_generation_ratio"
            ],
            0.28,
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
            STATUS_BLOCKED_GPU_POSTING_ACCUMULATION_FAILED,
        )

    def test_accumulation_script_defaults_to_shape_policy(self) -> None:
        args = accumulation_script.parse_args([])

        self.assertEqual(
            accumulation_script.centroid_budget_policy_from_args(args),
            "shape_rule_v0",
        )
        self.assertEqual(
            accumulation_script.case_selection_from_args(args),
            {"source": "wide_topk", "case_count": 5},
        )


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


def _selected_two_vectors() -> KayakPlaidI8SelectedCentroids:
    return KayakPlaidI8SelectedCentroids(
        positions_by_query=((0, 1, 1, 0),),
        scores_by_query=((0.5, 0.25, 0.75, 0.1),),
        query_count=1,
        query_vector_count=2,
        centroids_per_query_vector=2,
    )


def _row(
    *,
    kind: str,
    all_candidate_ratio: float,
    all_plus_topk_candidate_ratio: float,
    all_device_topk_candidate_ratio: float,
    projected_resident_ratio: float,
    projected_cold_ratio: float,
    projected_device_topk_resident_ratio: float,
    projected_device_topk_cold_ratio: float,
    all_posting_ratio: float,
    kernel_posting_ratio: float,
    visits: int,
) -> dict[str, object]:
    return {
        "status": STATUS_OK,
        "candidate_window_kind": kind,
        "selected_postings": {"expanded_posting_count": visits},
        "comparison": {
            "gpu_posting_accumulation_all_measured_seconds_per_cpu_candidate_generation_second": (
                all_candidate_ratio
            ),
            "gpu_posting_accumulation_all_measured_plus_host_topk_seconds_per_cpu_candidate_generation_second": (
                all_plus_topk_candidate_ratio
            ),
            "gpu_posting_accumulation_all_measured_device_topk_seconds_per_cpu_candidate_generation_second": (
                all_device_topk_candidate_ratio
            ),
            "projected_resident_payload_candidate_seconds_per_cpu_candidate_generation_second": (
                projected_resident_ratio
            ),
            "projected_cold_payload_candidate_seconds_per_cpu_candidate_generation_second": (
                projected_cold_ratio
            ),
            "projected_device_topk_resident_payload_candidate_seconds_per_cpu_candidate_generation_second": (
                projected_device_topk_resident_ratio
            ),
            "projected_device_topk_cold_payload_candidate_seconds_per_cpu_candidate_generation_second": (
                projected_device_topk_cold_ratio
            ),
            "gpu_posting_accumulation_all_measured_seconds_per_cpu_posting_accumulation_second": (
                all_posting_ratio
            ),
            "gpu_posting_accumulation_kernel_seconds_per_cpu_posting_accumulation_second": (
                kernel_posting_ratio
            ),
        },
    }


if __name__ == "__main__":
    unittest.main()
