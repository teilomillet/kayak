from __future__ import annotations

import sys
import unittest

from kayak_bridge.cache_paths import REPO_ROOT


SCRIPT_ROOT = REPO_ROOT / "python" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

import profile_gpu_i8_fused_centroid_posting_accumulation as fused_script  # noqa: E402
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
from kayak_bridge.gpu_i8_fused_centroid_posting_accumulation import (  # noqa: E402
    STATUS_BLOCKED_GPU_FUSED_CENTROID_POSTING_ACCUMULATION_FAILED,
    comparison_payload,
    report_status,
    summary_payload,
)
from kayak_bridge.mojo_gpu_i8_rerank import (  # noqa: E402
    MojoGpuI8FusedCentroidPostingAccumulationResult,
)


class GpuI8FusedCentroidPostingAccumulationTests(unittest.TestCase):
    def test_fused_result_reports_agreement(self) -> None:
        result = MojoGpuI8FusedCentroidPostingAccumulationResult(
            host_marshalling_seconds=0.1,
            extension_call_seconds=0.2,
            mojo_host_ingest_mean_seconds=0.03,
            payload_host_to_device_mean_seconds=0.04,
            query_host_to_device_mean_seconds=0.05,
            centroid_score_kernel_mean_seconds=0.06,
            centroid_selection_kernel_mean_seconds=0.07,
            accumulation_kernel_mean_seconds=0.08,
            device_to_host_mean_seconds=0.09,
            host_topk_mean_seconds=0.01,
            selected_validation_device_to_host_mean_seconds=0.02,
            centroid_token_out_of_range_count=0,
            selected_position_mismatch_count=0,
            selected_score_mismatch_count=0,
            selected_score_delta_max_abs=0.0,
            score_mismatch_count=0,
            score_delta_max_abs=0.0,
            score_delta_tolerance=0.0008,
            topk_position_mismatch_count=0,
            doc_index_out_of_range_count=0,
            top_k=2,
            topk_position_count=4,
            selected_centroid_count=16,
            centroid_score_count=64,
            document_score_count=8,
            document_count=4,
            query_count=2,
            query_vector_count=4,
            centroids_per_query_vector=2,
        )

        payload = result.to_json_ready()

        self.assertTrue(payload["fused_agreement_ok"])
        self.assertEqual(payload["selected_centroid_count"], 16)
        self.assertEqual(payload["score_delta_tolerance"], 0.0008)

    def test_comparison_payload_reports_fused_ratios(self) -> None:
        comparison = comparison_payload(
            cpu_candidate_generation_mean_seconds=0.2,
            cpu_centroid_scoring_plus_selection_seconds=0.04,
            cpu_posting_accumulation_seconds=0.05,
            cpu_final_topk_seconds=0.01,
            gpu_parsed={
                "extension_call_seconds": 0.09,
                "mojo_host_ingest_mean_seconds": 0.001,
                "payload_host_to_device_mean_seconds": 0.002,
                "query_host_to_device_mean_seconds": 0.003,
                "centroid_score_kernel_mean_seconds": 0.004,
                "centroid_selection_kernel_mean_seconds": 0.005,
                "accumulation_kernel_mean_seconds": 0.006,
                "device_to_host_mean_seconds": 0.007,
                "host_topk_mean_seconds": 0.008,
                "selected_validation_device_to_host_mean_seconds": 0.009,
            },
        )

        self.assertAlmostEqual(
            comparison["cpu_i8_centroid_selection_posting_topk_seconds"],
            0.10,
        )
        self.assertAlmostEqual(
            comparison["gpu_fused_kernel_chain_mean_seconds"],
            0.015,
        )
        self.assertAlmostEqual(
            comparison["gpu_fused_resident_payload_mean_seconds"],
            0.033,
        )
        self.assertAlmostEqual(
            comparison[
                "gpu_fused_resident_payload_seconds_per_cpu_candidate_generation_second"
            ],
            0.165,
        )
        self.assertAlmostEqual(
            comparison[
                "gpu_fused_resident_payload_seconds_per_cpu_centroid_selection_posting_topk_second"
            ],
            0.33,
        )

    def test_summary_reports_non_full_ratios(self) -> None:
        rows = [
            _row(
                kind="non_full",
                resident_candidate_ratio=0.4,
                cold_candidate_ratio=0.45,
                resident_cpu_slice_ratio=0.6,
            ),
            _row(
                kind="full",
                resident_candidate_ratio=2.0,
                cold_candidate_ratio=2.1,
                resident_cpu_slice_ratio=0.9,
            ),
        ]

        summary = summary_payload(rows)

        self.assertEqual(summary["ok_case_count"], 2)
        self.assertEqual(summary["ok_non_full_case_count"], 1)
        self.assertEqual(
            summary[
                "best_non_full_resident_payload_vs_cpu_candidate_generation_ratio"
            ],
            0.4,
        )
        self.assertEqual(
            summary[
                "worst_non_full_resident_payload_vs_cpu_centroid_selection_posting_topk_ratio"
            ],
            0.6,
        )

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
            STATUS_BLOCKED_GPU_FUSED_CENTROID_POSTING_ACCUMULATION_FAILED,
        )

    def test_script_defaults_to_shape_policy(self) -> None:
        args = fused_script.parse_args([])

        self.assertEqual(
            fused_script.centroid_budget_policy_from_args(args),
            "shape_rule_v0",
        )
        self.assertEqual(
            fused_script.case_selection_from_args(args),
            {"source": "wide_topk", "case_count": 5},
        )


def _row(
    *,
    kind: str,
    resident_candidate_ratio: float,
    cold_candidate_ratio: float,
    resident_cpu_slice_ratio: float,
) -> dict[str, object]:
    return {
        "status": STATUS_OK,
        "candidate_window_kind": kind,
        "comparison": {
            "gpu_fused_resident_payload_seconds_per_cpu_candidate_generation_second": (
                resident_candidate_ratio
            ),
            "gpu_fused_cold_payload_seconds_per_cpu_candidate_generation_second": (
                cold_candidate_ratio
            ),
            "gpu_fused_resident_payload_seconds_per_cpu_centroid_selection_posting_topk_second": (
                resident_cpu_slice_ratio
            ),
        },
    }


if __name__ == "__main__":
    unittest.main()
