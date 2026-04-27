from __future__ import annotations

import sys
import unittest

from kayak_bridge.cache_paths import REPO_ROOT


SCRIPT_ROOT = REPO_ROOT / "python" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

import profile_gpu_i8_candidate_generation_payload as payload_script  # noqa: E402
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
from kayak_bridge.gpu_i8_candidate_generation_payload import (  # noqa: E402
    STATUS_BLOCKED_GPU_CANDIDATE_PAYLOAD_FAILED,
    candidate_window_kind,
    comparison_payload,
    report_status,
    summary_payload,
)
from kayak_bridge.mojo_gpu_i8_rerank import (  # noqa: E402
    MojoGpuI8CandidateGenerationPayloadResult,
)


class GpuI8CandidateGenerationPayloadTests(unittest.TestCase):
    def test_payload_result_reports_copy_and_invariant_agreement(self) -> None:
        result = MojoGpuI8CandidateGenerationPayloadResult(
            host_marshalling_seconds=0.1,
            extension_call_seconds=0.2,
            mojo_host_ingest_mean_seconds=0.03,
            host_to_device_mean_seconds=0.04,
            device_to_host_mean_seconds=0.05,
            copy_mismatch_count=0,
            offset_violation_count=0,
            doc_index_out_of_range_count=0,
            centroid_count=128,
            posting_count=512,
            document_count=64,
            byte_counts={
                "centroid_token_indices": 1024,
                "centroid_doc_offsets": 1032,
                "centroid_doc_indices": 4096,
            },
        )

        payload = result.to_json_ready()

        self.assertTrue(payload["payload_agreement_ok"])
        self.assertTrue(payload["posting_invariants_ok"])
        self.assertEqual(payload["total_payload_bytes"], 6152)
        self.assertEqual(payload["centroid_count"], 128)
        self.assertEqual(payload["posting_count"], 512)

    def test_comparison_payload_keeps_cpu_candidate_generation_denominator(
        self,
    ) -> None:
        comparison = comparison_payload(
            cpu_candidate_generation_mean_seconds=0.01,
            gpu_parsed={
                "extension_call_seconds": 0.004,
                "mojo_host_ingest_mean_seconds": 0.001,
                "host_to_device_mean_seconds": 0.002,
                "device_to_host_mean_seconds": 0.003,
            },
        )

        self.assertEqual(
            comparison[
                "gpu_payload_extension_call_seconds_per_cpu_candidate_generation_second"
            ],
            0.4,
        )
        self.assertEqual(
            comparison[
                "gpu_payload_h2d_seconds_per_cpu_candidate_generation_second"
            ],
            0.2,
        )
        self.assertEqual(
            comparison[
                "gpu_payload_h2d_plus_d2h_seconds_per_cpu_candidate_generation_second"
            ],
            0.5,
        )

    def test_summary_reports_payload_ratios_and_bytes(self) -> None:
        rows = [
            _row(
                name="a",
                h2d_ratio=0.2,
                h2d_d2h_ratio=0.3,
                ext_ratio=0.4,
                bytes_=64,
                candidate_window_kind="non_full",
            ),
            _row(
                name="b",
                h2d_ratio=0.1,
                h2d_d2h_ratio=0.5,
                ext_ratio=0.6,
                bytes_=128,
                candidate_window_kind="full",
            ),
        ]

        summary = summary_payload(rows)

        self.assertEqual(summary["ok_case_count"], 2)
        self.assertEqual(summary["ok_non_full_case_count"], 1)
        self.assertEqual(summary["full_window_case_count"], 1)
        self.assertEqual(summary["best_payload_h2d_ratio"], 0.1)
        self.assertEqual(summary["worst_payload_h2d_plus_d2h_ratio"], 0.5)
        self.assertEqual(summary["worst_payload_extension_call_ratio"], 0.6)
        self.assertEqual(summary["best_non_full_payload_h2d_ratio"], 0.2)
        self.assertEqual(
            summary["worst_non_full_payload_h2d_plus_d2h_ratio"],
            0.3,
        )
        self.assertEqual(summary["max_payload_bytes"], 128)

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
            STATUS_BLOCKED_GPU_CANDIDATE_PAYLOAD_FAILED,
        )

    def test_payload_script_defaults_to_shape_policy(self) -> None:
        args = payload_script.parse_args([])

        self.assertEqual(
            payload_script.centroid_budget_policy_from_args(args),
            "shape_rule_v0",
        )
        self.assertEqual(
            payload_script.case_selection_from_args(args),
            {"source": "wide_topk", "case_count": 5},
        )

    def test_payload_script_allows_static_centroid_budget(self) -> None:
        args = payload_script.parse_args(["--centroid-budget-policy", "none"])

        self.assertIsNone(payload_script.centroid_budget_policy_from_args(args))

    def test_candidate_window_kind_marks_full_window_rows(self) -> None:
        full = payload_script.parse_sweep_case(
            "full:documents=64,document_vectors=8,queries=1,"
            "query_vectors=4,candidate_k=64"
        )
        non_full = payload_script.parse_sweep_case(
            "partial:documents=64,document_vectors=8,queries=1,"
            "query_vectors=4,candidate_k=16"
        )

        self.assertEqual(candidate_window_kind(full), "full")
        self.assertEqual(candidate_window_kind(non_full), "non_full")


def _row(
    *,
    name: str,
    h2d_ratio: float,
    h2d_d2h_ratio: float,
    ext_ratio: float,
    bytes_: int,
    candidate_window_kind: str,
) -> dict[str, object]:
    return {
        "name": name,
        "status": STATUS_OK,
        "candidate_window_kind": candidate_window_kind,
        "candidate_generation_payload": {"total_payload_bytes": bytes_},
        "comparison": {
            "gpu_payload_h2d_seconds_per_cpu_candidate_generation_second": (
                h2d_ratio
            ),
            "gpu_payload_h2d_plus_d2h_seconds_per_cpu_candidate_generation_second": (
                h2d_d2h_ratio
            ),
            "gpu_payload_extension_call_seconds_per_cpu_candidate_generation_second": (
                ext_ratio
            ),
        },
    }


if __name__ == "__main__":
    unittest.main()
