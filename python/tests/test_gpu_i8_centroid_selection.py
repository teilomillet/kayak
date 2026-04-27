from __future__ import annotations

import sys
import unittest

from kayak_bridge.cache_paths import REPO_ROOT


SCRIPT_ROOT = REPO_ROOT / "python" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

import profile_gpu_i8_centroid_selection as selection_script  # noqa: E402
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
from kayak_bridge.gpu_i8_centroid_selection import (  # noqa: E402
    STATUS_BLOCKED_GPU_CENTROID_SELECTION_FAILED,
    comparison_payload,
    report_status,
    summary_payload,
)
from kayak_bridge.mojo_gpu_i8_rerank import (  # noqa: E402
    MojoGpuI8CentroidSelectionResult,
)


class GpuI8CentroidSelectionTests(unittest.TestCase):
    def test_selection_result_reports_agreement(self) -> None:
        result = MojoGpuI8CentroidSelectionResult(
            host_marshalling_seconds=0.1,
            extension_call_seconds=0.2,
            mojo_host_ingest_mean_seconds=0.03,
            payload_host_to_device_mean_seconds=0.04,
            query_host_to_device_mean_seconds=0.05,
            kernel_mean_seconds=0.06,
            device_to_host_mean_seconds=0.07,
            host_selection_mean_seconds=0.08,
            centroid_token_out_of_range_count=0,
            selected_position_mismatch_count=0,
            selected_score_mismatch_count=0,
            selected_score_delta_max_abs=0.0,
            selected_centroid_count=16,
            centroid_score_count=64,
            centroid_count=32,
            query_count=1,
            query_vector_count=2,
            centroids_per_query_vector=8,
        )

        payload = result.to_json_ready()

        self.assertTrue(payload["selection_agreement_ok"])
        self.assertEqual(payload["selected_centroid_count"], 16)
        self.assertEqual(payload["centroid_score_count"], 64)
        self.assertEqual(payload["selected_score_delta_tolerance"], 1.0e-4)

    def test_comparison_payload_reports_selection_denominators(self) -> None:
        comparison = comparison_payload(
            cpu_candidate_generation_mean_seconds=0.04,
            cpu_centroid_scoring_seconds=0.01,
            cpu_centroid_selection_seconds=0.02,
            gpu_parsed={
                "extension_call_seconds": 0.009,
                "mojo_host_ingest_mean_seconds": 0.001,
                "payload_host_to_device_mean_seconds": 0.002,
                "query_host_to_device_mean_seconds": 0.003,
                "kernel_mean_seconds": 0.004,
                "device_to_host_mean_seconds": 0.005,
                "host_selection_mean_seconds": 0.006,
            },
        )

        self.assertAlmostEqual(
            comparison["cpu_i8_centroid_scoring_plus_selection_seconds"],
            0.03,
        )
        self.assertAlmostEqual(
            comparison[
                "gpu_i8_centroid_selection_resident_payload_mean_seconds"
            ],
            0.018,
        )
        self.assertAlmostEqual(
            comparison[
                "gpu_i8_centroid_selection_resident_payload_seconds_per_cpu_centroid_scoring_plus_selection_second"
            ],
            0.6,
        )
        self.assertAlmostEqual(
            comparison[
                "gpu_i8_centroid_selection_cold_payload_seconds_per_cpu_candidate_generation_second"
            ],
            0.5,
        )

    def test_summary_reports_non_full_ratios(self) -> None:
        rows = [
            _row(kind="non_full", selection_ratio=0.7, candidate_ratio=0.2),
            _row(kind="full", selection_ratio=1.4, candidate_ratio=0.9),
        ]

        summary = summary_payload(rows)

        self.assertEqual(summary["ok_case_count"], 2)
        self.assertEqual(summary["ok_non_full_case_count"], 1)
        self.assertEqual(
            summary[
                "best_non_full_resident_payload_vs_cpu_centroid_scoring_plus_selection_ratio"
            ],
            0.7,
        )
        self.assertEqual(
            summary[
                "worst_resident_payload_vs_cpu_candidate_generation_ratio"
            ],
            0.9,
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
            STATUS_BLOCKED_GPU_CENTROID_SELECTION_FAILED,
        )

    def test_selection_script_defaults_to_shape_policy(self) -> None:
        args = selection_script.parse_args([])

        self.assertEqual(
            selection_script.centroid_budget_policy_from_args(args),
            "shape_rule_v0",
        )
        self.assertEqual(
            selection_script.case_selection_from_args(args),
            {"source": "wide_topk", "case_count": 5},
        )


def _row(
    *,
    kind: str,
    selection_ratio: float,
    candidate_ratio: float,
) -> dict[str, object]:
    return {
        "status": STATUS_OK,
        "candidate_window_kind": kind,
        "comparison": {
            "gpu_i8_centroid_selection_resident_payload_seconds_per_cpu_centroid_scoring_plus_selection_second": (
                selection_ratio
            ),
            "gpu_i8_centroid_selection_resident_payload_seconds_per_cpu_candidate_generation_second": (
                candidate_ratio
            ),
        },
    }


if __name__ == "__main__":
    unittest.main()
