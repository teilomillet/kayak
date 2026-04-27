from __future__ import annotations

import unittest

from kayak_bridge.cache_paths import REPO_ROOT

import sys


SCRIPT_ROOT = REPO_ROOT / "python" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

from kayak_bridge.gpu_i8_address_serve_sweep import AddressServeSweepCase
from kayak_bridge import gpu_i8_centroid_budget_policy as policy
from kayak_bridge import gpu_i8_centroid_budget_policy_runner as runner
from kayak_bridge import gpu_i8_centroid_budget_policy_replay as replay


class GpuI8CentroidBudgetPolicyTests(unittest.TestCase):
    def test_parse_policy_names_preserves_order_and_uniqueness(self) -> None:
        self.assertEqual(
            policy.parse_policy_names("static4, shape_rule_v0, static4"),
            ("static4", "shape_rule_v0"),
        )

        with self.assertRaisesRegex(ValueError, "unknown"):
            policy.parse_policy_names("fast")

    def test_shape_rule_v0_keeps_vector_count_axes_explicit(self) -> None:
        cases = [
            (
                AddressServeSweepCase("query_vectors32", 512, 16, 2, 32, 256),
                4,
            ),
            (
                AddressServeSweepCase("doc_vectors64", 512, 64, 2, 8, 256),
                16,
            ),
            (
                AddressServeSweepCase("query_batch4", 512, 16, 4, 8, 256),
                8,
            ),
            (
                AddressServeSweepCase("documents512", 512, 16, 2, 8, 128),
                24,
            ),
            (
                AddressServeSweepCase("query_vectors16", 256, 16, 2, 16, 128),
                16,
            ),
            (
                AddressServeSweepCase("baseline", 256, 16, 2, 8, 128),
                4,
            ),
        ]

        for case, expected_budget in cases:
            with self.subTest(case=case.name):
                choice = policy.choose_policy_budget(policy.SHAPE_RULE_V0_POLICY, case)
                self.assertEqual(choice.centroids_per_query_vector, expected_budget)
                self.assertEqual(choice.policy_kind, "shape_rule_v0")

    def test_replay_policy_row_compares_against_static32_baseline(self) -> None:
        case = AddressServeSweepCase("baseline", 256, 16, 2, 8, 128)
        row = replay.replay_policy_row(
            case=case,
            case_row=_case_row(
                _budget_row(4, candidate=0.25, plus_score=0.5),
                _budget_row(32, candidate=1.0, plus_score=1.0),
            ),
            policy_name="static4",
        )

        self.assertEqual(row["status"], policy.STATUS_OK)
        self.assertEqual(row["centroids_per_query_vector"], 4)
        self.assertEqual(
            row["comparison"]["candidate_generation_seconds_vs_baseline_budget"],
            0.25,
        )
        self.assertEqual(
            row["comparison"][
                "candidate_plus_score_seconds_vs_baseline_budget"
            ],
            0.5,
        )
        self.assertIs(
            row["comparison"]["no_final_recall_loss_vs_baseline_budget"],
            True,
        )

    def test_replay_policy_row_reports_missing_unswept_budget(self) -> None:
        case = AddressServeSweepCase("baseline", 256, 16, 2, 8, 128)
        row = replay.replay_policy_row(
            case=case,
            case_row=_case_row(_budget_row(32, candidate=1.0, plus_score=1.0)),
            policy_name="static4",
        )

        self.assertEqual(row["status"], policy.STATUS_MISSING_BUDGET)
        self.assertEqual(row["available_centroid_budgets"], [32])

        self.assertEqual(
            runner.report_status(
                [{"status": policy.STATUS_OK, "policy_rows": [row]}]
            ),
            "error",
        )

    def test_oracle_policy_is_labelled_as_calibration_ceiling(self) -> None:
        case = AddressServeSweepCase("baseline", 256, 16, 2, 8, 128)
        row = replay.replay_policy_row(
            case=case,
            case_row=_case_row(
                _budget_row(4, candidate=0.4, plus_score=0.4, final_recall=0.9),
                _budget_row(8, candidate=0.2, plus_score=0.2, final_recall=0.8),
                _budget_row(32, candidate=1.0, plus_score=1.0, final_recall=0.9),
            ),
            policy_name=policy.ORACLE_FASTEST_NO_FINAL_RECALL_LOSS,
        )

        self.assertEqual(row["policy_kind"], "oracle_calibration")
        self.assertEqual(row["centroids_per_query_vector"], 4)
        self.assertEqual(
            row["comparison"][
                "candidate_plus_score_seconds_vs_baseline_budget"
            ],
            0.4,
        )


def _case_row(*budget_rows: dict[str, object]) -> dict[str, object]:
    return {
        "baseline_centroids_per_query_vector": 32,
        "budget_rows": list(budget_rows),
    }


def _budget_row(
    budget: int,
    *,
    candidate: float,
    plus_score: float,
    window_recall: float = 1.0,
    final_recall: float = 0.9,
) -> dict[str, object]:
    return {
        "status": policy.STATUS_OK,
        "centroids_per_query_vector": budget,
        "cpu_i8_candidate_generation": {"mean_seconds": candidate},
        "cpu_i8_same_candidate_reference": {"mean_seconds": plus_score - candidate},
        "candidate_window_recall_at_k_vs_kayak_exact": window_recall,
        "recall_at_k_vs_kayak_exact": final_recall,
        "comparison": {
            "cpu_candidate_generation_plus_score_seconds": plus_score,
        },
    }


if __name__ == "__main__":
    unittest.main()
