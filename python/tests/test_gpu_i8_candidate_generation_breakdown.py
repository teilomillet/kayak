from __future__ import annotations

import unittest

from scripts.profile_gpu_i8_candidate_generation_breakdown import (
    aggregate_profiles,
    summary_payload,
)


class CandidateGenerationBreakdownTests(unittest.TestCase):
    def test_aggregate_reports_workspace_ratio_and_vector_counts(self) -> None:
        aggregate = aggregate_profiles(
            [
                _profile(full=2.0, workspace=1.0, agreement=1.0),
                _profile(full=4.0, workspace=3.0, agreement=1.0),
            ]
        )

        self.assertEqual(aggregate["query_count"], 2)
        self.assertEqual(aggregate["query_vector_count"], 8)
        self.assertEqual(aggregate["document_count"], 128)
        self.assertEqual(aggregate["document_vector_count"], 16)
        self.assertEqual(aggregate["candidate_k"], 32)
        self.assertEqual(
            aggregate[
                "workspace_full_candidate_mean_seconds_per_full_candidate_second"
            ],
            4.0 / 6.0,
        )
        self.assertEqual(
            aggregate["workspace_candidate_position_agreement_batch_sum"],
            2.0,
        )
        self.assertEqual(
            aggregate["unordered_candidate_mean_seconds_per_full_candidate_second"],
            0.5,
        )
        self.assertEqual(
            aggregate["unordered_candidate_set_agreement_batch_sum"],
            2.0,
        )
        self.assertEqual(
            aggregate[
                "unordered_final_topk_mean_seconds_per_final_topk_second"
            ],
            0.5,
        )
        self.assertEqual(
            aggregate[
                "unordered_centroid_selection_mean_seconds_per_centroid_selection_second"
            ],
            0.5,
        )

    def test_summary_reports_workspace_envelope(self) -> None:
        rows = [
            {"status": "ok", "aggregate": _aggregate(full=2.0, workspace=1.0)},
            {"status": "ok", "aggregate": _aggregate(full=4.0, workspace=3.0)},
        ]

        summary = summary_payload(rows)

        self.assertEqual(summary["ok_case_count"], 2)
        self.assertEqual(
            summary[
                "best_workspace_full_candidate_seconds_per_full_candidate_second"
            ],
            0.5,
        )
        self.assertEqual(
            summary[
                "worst_workspace_full_candidate_seconds_per_full_candidate_second"
            ],
            0.75,
        )
        self.assertEqual(
            summary["min_workspace_candidate_position_agreement"],
            1.0,
        )
        self.assertEqual(
            summary[
                "best_unordered_candidate_seconds_per_full_candidate_second"
            ],
            0.5,
        )
        self.assertEqual(
            summary["min_unordered_candidate_set_agreement"],
            1.0,
        )
        self.assertEqual(
            summary[
                "best_unordered_centroid_selection_seconds_per_centroid_selection_second"
            ],
            0.5,
        )
        self.assertEqual(
            summary["min_centroid_selection_set_agreement"],
            1.0,
        )
        self.assertEqual(
            summary[
                "best_unordered_final_topk_seconds_per_final_topk_second"
            ],
            0.4,
        )


def _profile(
    *, full: float, workspace: float, agreement: float
) -> dict[str, float | int]:
    return {
        "full_candidate_mean_seconds": full,
        "workspace_full_candidate_mean_seconds": workspace,
        "workspace_candidate_position_agreement": agreement,
        "unordered_candidate_mean_seconds": full / 2.0,
        "unordered_candidate_set_agreement": agreement,
        "centroid_scoring_mean_seconds": 0.2,
        "centroid_selection_mean_seconds": 0.3,
        "unordered_centroid_selection_mean_seconds": 0.15,
        "centroid_selection_set_agreement": agreement,
        "posting_accumulation_mean_seconds": 0.4,
        "final_topk_mean_seconds": 0.5,
        "unordered_final_topk_mean_seconds": 0.25,
        "selected_centroid_count": 64,
        "posting_visit_count": 256,
        "touched_document_count": 128,
        "output_candidate_count": 32,
        "query_vector_count": 8,
        "document_count": 128,
        "document_vector_count": 16,
        "total_document_vector_count": 2048,
        "centroid_count": 128,
        "centroids_per_query_vector": 8,
        "candidate_k": 32,
    }


def _aggregate(*, full: float, workspace: float) -> dict[str, float | int]:
    return {
        "full_candidate_mean_seconds_batch_sum": full,
        "posting_accumulation_mean_seconds_batch_sum": 0.1,
        "workspace_full_candidate_mean_seconds_per_full_candidate_second": (
            workspace / full
        ),
        "unordered_final_topk_mean_seconds_per_final_topk_second": 0.4,
        "unordered_candidate_mean_seconds_per_full_candidate_second": 0.5,
        "unordered_centroid_selection_mean_seconds_per_centroid_selection_second": 0.5,
        "workspace_candidate_position_agreement_batch_sum": 2.0,
        "unordered_candidate_set_agreement_batch_sum": 2.0,
        "centroid_selection_set_agreement_batch_sum": 2.0,
        "query_count": 2,
    }


if __name__ == "__main__":
    unittest.main()
