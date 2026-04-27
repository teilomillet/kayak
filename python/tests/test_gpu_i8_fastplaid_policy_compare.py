from __future__ import annotations

import sys
import unittest
from pathlib import Path

from kayak_bridge.cache_paths import REPO_ROOT


SCRIPT_ROOT = REPO_ROOT / "python" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

from kayak_bridge.gpu_i8_address_serve_sweep import AddressServeSweepCase
from kayak_bridge import gpu_i8_fastplaid_policy_compare as policy_compare


class GpuI8FastPlaidPolicyCompareTests(unittest.TestCase):
    def test_parse_fastplaid_devices_preserves_order_and_uniqueness(self) -> None:
        self.assertEqual(
            policy_compare.parse_fastplaid_devices("cpu,cuda,cpu"),
            ("cpu", "cuda"),
        )

        with self.assertRaisesRegex(ValueError, "device"):
            policy_compare.parse_fastplaid_devices(" , ")

    def test_compare_argv_uses_shape_policy_budget_and_case_seed(self) -> None:
        case = AddressServeSweepCase("query_batch4", 512, 16, 4, 8, 256)
        controls = policy_compare.FastPlaidPolicyCompareControls(
            policy_name="shape_rule_v0",
            seed=7,
            kayak_i8_positive_centroids_only=True,
            gpu_hybrid_shortlist_k=128,
        )

        argv = policy_compare.compare_argv_for_case(
            case=case,
            case_index=2,
            controls=controls,
            fastplaid_device="cuda",
            index_root=Path("indexes"),
            output=Path("report.json"),
            allow_missing_gpu=True,
            require_fastplaid=True,
            overwrite_index_root=True,
        )

        self.assertEqual(_arg_value(argv, "--seed"), "9")
        self.assertEqual(
            _arg_value(argv, "--kayak-plaid-centroids-per-query-vector"),
            "8",
        )
        self.assertEqual(_arg_value(argv, "--kayak-i8-candidate-order"), "unordered")
        self.assertEqual(_arg_value(argv, "--fastplaid-device"), "cuda")
        self.assertEqual(_arg_value(argv, "--gpu-hybrid-shortlist-k"), "128")
        self.assertIn("--allow-missing-gpu", argv)
        self.assertIn("--require-fastplaid", argv)
        self.assertIn("--overwrite-index-root", argv)
        self.assertIn("--kayak-i8-positive-centroids-only", argv)

    def test_summary_extracts_scope_metrics(self) -> None:
        case = AddressServeSweepCase("query_vectors32", 512, 16, 2, 32, 256)
        controls = policy_compare.FastPlaidPolicyCompareControls()
        row = policy_compare.summarize_case_device_report(
            case=case,
            case_index=0,
            controls=controls,
            fastplaid_device="cpu",
            report={
                "status": "ok",
                "shape": {"candidate_k": 256},
                "systems": [
                    {
                        "system_name": "kayak_plaid_mojo_probe",
                        "recall_at_k_vs_kayak_exact": 0.7,
                    },
                    {
                        "system_name": "fastplaid",
                        "recall_at_k_vs_kayak_exact": 0.4,
                        "query_batch_mean_seconds": 0.01,
                    },
                ],
                "gpu_prepared_handle_topk_no_reference_vs_fastplaid_scope_comparison": {
                    "cpu_candidate_generation_seconds_per_window": 0.0006,
                    "gpu_prepared_handle_topk_seconds_per_window": 0.0004,
                    "cpu_candidate_generation_plus_gpu_topk_seconds_per_window": 0.001,
                    "cpu_candidate_generation_plus_gpu_topk_seconds_per_fastplaid_batch_second": 0.1,
                    "topk_position_agreement": 1.0,
                },
                "gpu_fused_centroid_posting_vs_fastplaid_scope_comparison": {
                    "gpu_fused_device_topk_seconds_per_window": 0.0003,
                    "gpu_fused_device_topk_seconds_per_fastplaid_batch_second": 0.03,
                    "gpu_fused_device_topk_seconds_per_host_topk_second": 0.75,
                    "recall_at_k_vs_kayak_exact": 0.6,
                    "device_topk_position_agreement": 1.0,
                },
                "gpu_hybrid_shortlist_exact_rerank_vs_fastplaid_scope_comparison": {
                    "gpu_hybrid_seconds_per_window": 0.0005,
                    "gpu_hybrid_seconds_per_fastplaid_batch_second": 0.05,
                    "gpu_hybrid_exact_rerank_share": 0.4,
                    "recall_at_k_vs_kayak_exact": 0.65,
                    "final_topk_position_agreement": 1.0,
                    "shortlist_k": 256,
                },
            },
            report_path=Path("row.json"),
        )

        self.assertEqual(row["policy"]["centroids_per_query_vector"], 4)
        self.assertEqual(row["kayak_i8_candidate_order"], "unordered")
        self.assertIs(row["kayak_i8_positive_centroids_only"], False)
        self.assertEqual(row["kayak_i8_recall_at_k_vs_kayak_exact"], 0.7)
        self.assertEqual(row["fastplaid_recall_at_k_vs_kayak_exact"], 0.4)
        self.assertEqual(
            row[
                "cpu_candidate_generation_plus_gpu_topk_no_reference_seconds_per_fastplaid_batch_second"
            ],
            0.1,
        )
        self.assertEqual(
            row["gpu_fused_device_topk_seconds_per_fastplaid_batch_second"],
            0.03,
        )
        self.assertAlmostEqual(row["gpu_fused_recall_delta_vs_fastplaid"], 0.2)
        self.assertEqual(row["gpu_hybrid_shortlist_k"], 256)
        self.assertEqual(
            row["gpu_hybrid_seconds_per_fastplaid_batch_second"],
            0.05,
        )
        self.assertAlmostEqual(row["gpu_hybrid_recall_delta_vs_fastplaid"], 0.25)
        self.assertEqual(row["cpu_candidate_generation_share_of_envelope"], 0.6)
        self.assertEqual(row["gpu_topk_no_reference_share_of_envelope"], 0.4)
        self.assertEqual(
            policy_compare.summary_payload([row])["min_recall_delta_vs_fastplaid"],
            0.29999999999999993,
        )
        self.assertEqual(
            policy_compare.summary_payload([row])[
                "mean_cpu_candidate_generation_share_of_envelope"
            ],
            0.6,
        )
        self.assertEqual(
            policy_compare.summary_payload([row])[
                "max_gpu_fused_device_topk_seconds_per_fastplaid_batch_second"
            ],
            0.03,
        )
        self.assertEqual(
            policy_compare.summary_payload([row])[
                "max_gpu_hybrid_seconds_per_fastplaid_batch_second"
            ],
            0.05,
        )
        self.assertEqual(
            policy_compare.summary_payload([row])[
                "gpu_hybrid_final_topk_position_agreement_min"
            ],
            1.0,
        )


def _arg_value(argv: list[str], name: str) -> str:
    index = argv.index(name)
    return argv[index + 1]


if __name__ == "__main__":
    unittest.main()
