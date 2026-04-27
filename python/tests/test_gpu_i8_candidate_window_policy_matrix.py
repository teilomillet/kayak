from __future__ import annotations

import sys
import unittest

from kayak_bridge.cache_paths import REPO_ROOT


SCRIPT_ROOT = REPO_ROOT / "python" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

import compare_gpu_i8_fastplaid_candidate_window_policies as matrix_script  # noqa: E402
from kayak_bridge.gpu_i8_candidate_window_policy import (  # noqa: E402
    COVERAGE_SAFETY_V0_POLICY,
    COVERAGE_SAFETY_V1_POLICY,
    DOC_VECTORS64_125PCT_POLICY,
    INPUT_CANDIDATE_K_POLICY,
)
from kayak_bridge.gpu_i8_candidate_window_policy_matrix import (  # noqa: E402
    summarize_policy_matrix,
)


class GpuI8CandidateWindowPolicyMatrixTests(unittest.TestCase):
    def test_summarize_policy_matrix_compares_policy_to_input(self) -> None:
        baseline = _report(
            policy_name=INPUT_CANDIDATE_K_POLICY,
            effective_candidate_k=256,
            resident_recall=0.65,
            resident_delta_vs_fastplaid=-0.05,
            resident_ratio=0.10,
        )
        widened = _report(
            policy_name=DOC_VECTORS64_125PCT_POLICY,
            effective_candidate_k=320,
            resident_recall=0.75,
            resident_delta_vs_fastplaid=0.05,
            resident_ratio=0.12,
        )

        summary = summarize_policy_matrix([baseline, widened])

        self.assertEqual(summary["policy_count"], 2)
        self.assertEqual(
            summary["resident_selected_negative_recall_delta_rows"][0][
                "gpu_resident_selected_recall_delta_vs_fastplaid"
            ],
            -0.05,
        )
        policy_summaries = {
            row["candidate_window_policy"]: row
            for row in summary["policy_summaries"]
        }
        self.assertEqual(
            policy_summaries[DOC_VECTORS64_125PCT_POLICY]["widened_row_count"],
            1,
        )
        self.assertEqual(
            policy_summaries[INPUT_CANDIDATE_K_POLICY][
                "negative_recall_delta_row_count"
            ],
            1,
        )
        self.assertEqual(
            policy_summaries[DOC_VECTORS64_125PCT_POLICY][
                "negative_recall_delta_row_count"
            ],
            0,
        )
        self.assertEqual(
            policy_summaries[DOC_VECTORS64_125PCT_POLICY][
                "mean_gpu_resident_selected_cpu_selection_share"
            ],
            0.2,
        )
        self.assertEqual(
            policy_summaries[DOC_VECTORS64_125PCT_POLICY][
                "mean_gpu_resident_selected_candidate_share"
            ],
            0.3,
        )
        self.assertEqual(
            policy_summaries[DOC_VECTORS64_125PCT_POLICY][
                "mean_gpu_resident_selected_exact_share"
            ],
            0.5,
        )
        self.assertEqual(
            policy_summaries[DOC_VECTORS64_125PCT_POLICY][
                "max_effective_to_input_candidate_k"
            ],
            1.25,
        )
        comparison = summary["baseline_comparisons"][0]
        self.assertAlmostEqual(
            comparison["gpu_resident_selected_recall_delta_vs_baseline"],
            0.10,
        )
        self.assertAlmostEqual(
            comparison["gpu_resident_selected_fastplaid_delta_delta_vs_baseline"],
            0.10,
        )
        self.assertAlmostEqual(
            comparison["gpu_resident_selected_fastplaid_ratio_vs_baseline"],
            1.2,
        )
        self.assertEqual(
            summary["baseline_comparison_summary"][
                "max_gpu_resident_selected_fastplaid_ratio_vs_baseline"
            ],
            1.2,
        )

    def test_matrix_script_defaults_to_generalization_shape_family(self) -> None:
        args = matrix_script.parse_args([])

        sources = matrix_script.case_sources(args)
        policies = matrix_script.candidate_window_policies(args)

        self.assertEqual(sources[0][0], "candidate_window_generalization")
        self.assertEqual(
            policies,
            (
                INPUT_CANDIDATE_K_POLICY,
                DOC_VECTORS64_125PCT_POLICY,
                COVERAGE_SAFETY_V0_POLICY,
                COVERAGE_SAFETY_V1_POLICY,
            ),
        )

    def test_single_policy_argv_serializes_explicit_vector_counts(self) -> None:
        args = matrix_script.parse_args(
            [
                "--case",
                "doc_vectors64:documents=512,document_vectors=64,queries=2,query_vectors=8,candidate_k=256",
                "--candidate-window-policy",
                DOC_VECTORS64_125PCT_POLICY,
                "--fastplaid-devices",
                "cpu",
                "--fixed-case-seed",
                "--no-require-fastplaid",
            ]
        )
        source_name, cases = matrix_script.case_sources(args)[0]

        argv = matrix_script.single_policy_argv(
            args,
            case_source=source_name,
            cases=cases,
            candidate_window_policy=DOC_VECTORS64_125PCT_POLICY,
            output=REPO_ROOT / ".cache/test-summary.json",
        )

        self.assertIn("--case", argv)
        case_text = argv[argv.index("--case") + 1]
        self.assertIn("documents=512", case_text)
        self.assertIn("document_vectors=64", case_text)
        self.assertIn("query_vectors=8", case_text)
        self.assertEqual(
            argv[argv.index("--candidate-window-policy") + 1],
            DOC_VECTORS64_125PCT_POLICY,
        )
        self.assertIn("--fixed-case-seed", argv)
        self.assertIn("--no-require-fastplaid", argv)


def _report(
    *,
    policy_name: str,
    effective_candidate_k: int,
    resident_recall: float,
    resident_delta_vs_fastplaid: float,
    resident_ratio: float,
) -> dict[str, object]:
    return {
        "status": "ok",
        "case_selection": {"source": "candidate_window_generalization"},
        "controls": {"candidate_window_policy": policy_name},
        "rows": [
            {
                "name": "doc_vectors64",
                "status": "ok",
                "fastplaid_device": "cuda",
                "seed": 7,
                "shape": {
                    "document_count": 512,
                    "document_vector_count": 64,
                    "query_count": 2,
                    "query_vector_count": 8,
                },
                "input_candidate_k": 256,
                "effective_candidate_k": effective_candidate_k,
                "gpu_resident_selected_recall_at_k_vs_kayak_exact": (
                    resident_recall
                ),
                "fastplaid_recall_at_k_vs_kayak_exact": 0.70,
                "gpu_resident_selected_recall_delta_vs_fastplaid": (
                    resident_delta_vs_fastplaid
                ),
                "gpu_resident_selected_exact_rerank_seconds_per_fastplaid_batch_second": (
                    resident_ratio
                ),
                "gpu_resident_selected_cpu_selection_share": 0.2,
                "gpu_resident_selected_candidate_share": 0.3,
                "gpu_resident_selected_exact_rerank_exact_share": 0.5,
                "gpu_resident_selected_final_topk_position_agreement": 1.0,
                "gpu_resident_selected_candidate_position_agreement_min": 1.0,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
