from __future__ import annotations

import sys
import unittest

from kayak_bridge.cache_paths import REPO_ROOT


SCRIPT_ROOT = REPO_ROOT / "python" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

import profile_gpu_i8_fused_centroid_posting_handle as handle_script  # noqa: E402
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
from kayak_bridge.gpu_i8_fused_centroid_posting_handle import (  # noqa: E402
    STATUS_BLOCKED_GPU_FUSED_HANDLE_FAILED,
    comparison_payload,
    report_status,
    summary_payload,
)
from kayak_bridge.mojo_gpu_i8_rerank import (  # noqa: E402
    MojoGpuI8FusedCentroidPostingProfileResult,
    MojoGpuI8FusedCentroidPostingTopKResult,
)


class GpuI8FusedCentroidPostingHandleTests(unittest.TestCase):
    def test_topk_result_reports_agreement(self) -> None:
        result = MojoGpuI8FusedCentroidPostingTopKResult(
            host_marshalling_seconds=0.01,
            extension_call_seconds=0.02,
            document_score_count=16,
            top_k=2,
            topk_position_match_count=4,
            topk_score_delta_max_abs=0.0,
            positions=(1, 2, 3, 4),
            scores=(5.0, 4.0, 3.0, 2.0),
        )

        payload = result.to_json_ready()

        self.assertTrue(payload["topk_agreement_ok"])
        self.assertEqual(payload["topk_position_count"], 4)
        self.assertEqual(payload["topk_position_agreement"], 1.0)

    def test_profile_result_reports_breakdown_and_agreement(self) -> None:
        result = MojoGpuI8FusedCentroidPostingProfileResult(
            host_marshalling_seconds=0.001,
            extension_call_seconds=0.2,
            query_host_ingest_mean_seconds=0.003,
            query_host_to_device_mean_seconds=0.004,
            centroid_score_kernel_mean_seconds=0.005,
            centroid_selection_kernel_mean_seconds=0.006,
            accumulation_kernel_mean_seconds=0.007,
            reduction_kernel_mean_seconds=0.008,
            device_to_host_mean_seconds=0.009,
            host_topk_restore_mean_seconds=0.01,
            host_topk_restore_plus_destructive_mean_seconds=0.03,
            host_topk_destructive_estimated_mean_seconds=0.02,
            host_topk_non_destructive_mean_seconds=0.04,
            document_score_count=16,
            top_k=2,
            topk_position_match_count=4,
            topk_score_delta_max_abs=0.0,
            positions=(1, 2, 3, 4),
            scores=(5.0, 4.0, 3.0, 2.0),
        )

        payload = result.to_json_ready()

        self.assertTrue(payload["topk_agreement_ok"])
        self.assertEqual(
            payload["host_topk_destructive_estimated_mean_seconds"],
            0.02,
        )

    def test_comparison_payload_reports_handle_ratios(self) -> None:
        comparison = comparison_payload(
            cpu_candidate_generation_mean_seconds=0.2,
            cpu_centroid_selection_posting_topk_seconds=0.1,
            gpu_parsed={
                "prepare_extension_call_seconds": 0.03,
                "score_host_marshalling_mean_seconds": 0.004,
                "score_extension_call_mean_seconds": 0.05,
                "profile": {
                    "centroid_score_kernel_mean_seconds": 0.01,
                    "centroid_selection_kernel_mean_seconds": 0.02,
                    "accumulation_kernel_mean_seconds": 0.03,
                    "reduction_kernel_mean_seconds": 0.04,
                    "query_host_to_device_mean_seconds": 0.005,
                    "device_to_host_mean_seconds": 0.006,
                    "host_topk_destructive_estimated_mean_seconds": 0.007,
                },
            },
        )

        self.assertAlmostEqual(
            comparison[
                "gpu_fused_handle_score_extension_seconds_per_cpu_candidate_generation_second"
            ],
            0.25,
        )
        self.assertAlmostEqual(
            comparison[
                "gpu_fused_handle_score_host_plus_extension_seconds_per_cpu_candidate_generation_second"
            ],
            0.27,
        )
        self.assertAlmostEqual(
            comparison[
                "gpu_fused_handle_score_extension_seconds_per_cpu_centroid_selection_posting_topk_second"
            ],
            0.5,
        )
        self.assertAlmostEqual(
            comparison["gpu_fused_handle_profile_kernel_chain_mean_seconds"],
            0.1,
        )
        self.assertAlmostEqual(
            comparison[
                "gpu_fused_handle_profile_query_h2d_kernel_d2h_topk_mean_seconds"
            ],
            0.118,
        )

    def test_summary_reports_non_full_ratios(self) -> None:
        rows = [
            _row(kind="non_full", candidate_ratio=0.3, slice_ratio=0.4),
            _row(kind="full", candidate_ratio=9.0, slice_ratio=0.8),
        ]

        summary = summary_payload(rows)

        self.assertEqual(summary["ok_case_count"], 2)
        self.assertEqual(summary["ok_non_full_case_count"], 1)
        self.assertEqual(
            summary[
                "best_non_full_score_extension_vs_cpu_candidate_generation_ratio"
            ],
            0.3,
        )
        self.assertEqual(
            summary[
                "worst_non_full_score_extension_vs_cpu_centroid_selection_posting_topk_ratio"
            ],
            0.4,
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
            STATUS_BLOCKED_GPU_FUSED_HANDLE_FAILED,
        )

    def test_script_defaults_to_shape_policy(self) -> None:
        args = handle_script.parse_args([])

        self.assertEqual(
            handle_script.centroid_budget_policy_from_args(args),
            "shape_rule_v0",
        )
        self.assertEqual(
            handle_script.case_selection_from_args(args),
            {"source": "wide_topk", "case_count": 5},
        )


def _row(
    *,
    kind: str,
    candidate_ratio: float,
    slice_ratio: float,
) -> dict[str, object]:
    return {
        "status": STATUS_OK,
        "candidate_window_kind": kind,
        "comparison": {
            "gpu_fused_handle_score_extension_seconds_per_cpu_candidate_generation_second": (
                candidate_ratio
            ),
            "gpu_fused_handle_score_extension_seconds_per_cpu_centroid_selection_posting_topk_second": (
                slice_ratio
            ),
        },
    }


if __name__ == "__main__":
    unittest.main()
