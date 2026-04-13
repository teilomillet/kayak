from __future__ import annotations

import unittest

import numpy as np

import kayak


def _dim128_vector(*entries: tuple[int, float]) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    for index, value in entries:
        vector[index] = np.float32(value)
    return vector


class SearchPlanApiTests(unittest.TestCase):
    def _build_fixture(self) -> tuple[kayak.LateQuery, kayak.LateIndex]:
        query = kayak.query(
            np.stack(
                [
                    _dim128_vector((0, 1.0)),
                    _dim128_vector((1, 1.0)),
                ]
            )
        )
        index = kayak.documents(
            ["doc-b", "doc-a", "doc-c"],
            [
                np.stack(
                    [
                        _dim128_vector((0, 1.0)),
                        _dim128_vector((0, 1.0)),
                    ]
                ),
                np.stack(
                    [
                        _dim128_vector((0, 1.0)),
                        _dim128_vector((1, 1.0)),
                    ]
                ),
                np.stack(
                    [
                        _dim128_vector((1, 1.0)),
                        _dim128_vector((1, 1.0)),
                    ]
                ),
            ],
        ).pack()
        return query, index

    def _build_clause_text_fixture(self) -> tuple[kayak.LateQuery, kayak.LateIndex]:
        query = kayak.query(
            np.stack(
                [
                    _dim128_vector((0, 1.0)),
                    _dim128_vector((0, 1.0)),
                ]
            ),
            text=(
                "Gugulethu township logo. founded in 1984 in a church "
                "longest serving employee artistic director"
            ),
        )
        index = kayak.documents(
            ["doc-context", "doc-answer"],
            [
                np.stack(
                    [
                        _dim128_vector((0, 1.0)),
                        _dim128_vector((0, 1.0)),
                    ]
                ),
                np.stack(
                    [
                        _dim128_vector((0, 1.0)),
                        _dim128_vector((0, 0.8), (1, 0.2)),
                    ]
                ),
            ],
            texts=[
                "Gugulethu township logo emblem heritage schools history",
                (
                    "Zama Dance School was founded in 1984 in a church and "
                    "the longest serving employee is the artistic director."
                ),
            ],
        ).pack()
        return query, index

    def test_exact_full_scan_candidate_generation_matches_exact_scores(self) -> None:
        query, index = self._build_fixture()

        result = kayak.generate_candidates(
            query,
            index,
            kayak.exact_full_scan_candidate_generator(),
            k=2,
        )

        np.testing.assert_allclose(
            result.scores.numpy(),
            np.array([1.0, 2.0, 1.0], dtype=np.float32),
        )
        self.assertEqual(result.candidate_doc_ids, ("doc-a", "doc-b"))
        self.assertEqual(result.profile.stage_name, "exact_full_scan")
        self.assertEqual(result.profile.query_vector_count, 2)
        self.assertEqual(result.profile.document_count, 3)
        self.assertEqual(result.profile.document_vector_count, 6)

    def test_document_proxy_candidate_generation_is_explicitly_compressed(self) -> None:
        query, index = self._build_fixture()
        generator = kayak.document_proxy_candidate_generator(
            query_vector_budget=1,
            document_vector_budget=1,
        )

        result = kayak.generate_candidates(query, index, generator, k=2)

        np.testing.assert_allclose(
            result.scores.numpy(),
            np.array([1.0, 1.0, 0.0], dtype=np.float32),
        )
        self.assertEqual(result.candidate_doc_ids, ("doc-b", "doc-a"))
        self.assertEqual(result.profile.stage_name, "document_proxy")
        self.assertEqual(result.profile.query_vector_count, 1)
        self.assertEqual(result.profile.document_count, 3)
        self.assertEqual(result.profile.document_vector_count, 3)
        self.assertEqual(generator.query_vector_budget, 1)
        self.assertEqual(generator.document_vector_budget, 1)

    def test_document_proxy_plan_exactly_reranks_candidate_window(self) -> None:
        query, index = self._build_fixture()
        plan = kayak.document_proxy_search_plan(
            final_k=1,
            candidate_k=2,
        )

        result = kayak.search_with_plan(query, index, plan)

        self.assertEqual(result.candidate_stage.candidate_doc_ids, ("doc-b", "doc-a"))
        self.assertEqual([hit.doc_id for hit in result.hits], ["doc-a"])
        self.assertIsNotNone(result.candidate_index)
        assert result.candidate_index is not None
        self.assertEqual(result.candidate_index.doc_ids, ("doc-b", "doc-a"))
        self.assertEqual(result.exact_stage.stage_name, "exact_late_interaction")
        self.assertEqual(result.exact_stage.input_hit_count, 2)
        self.assertEqual(result.exact_stage.output_hit_count, 1)
        self.assertEqual(result.exact_stage.query_vector_count, 2)
        self.assertEqual(result.exact_stage.document_count, 2)
        self.assertEqual(result.exact_stage.document_vector_count, 4)
        self.assertEqual(result.stage2.stage_name, "exact_late_interaction")
        self.assertIs(result.exact_stage, result.stage2)
        self.assertIs(result.exact_scores, result.stage2_scores)

    def test_exact_full_scan_plan_defaults_to_noop_stage2(self) -> None:
        query, index = self._build_fixture()
        plan = kayak.exact_full_scan_search_plan(final_k=2, candidate_k=3)

        result = kayak.search_with_plan(query, index, plan)

        self.assertEqual(result.plan.stage2_operator.kind, "noop_topk")
        self.assertEqual(result.hits, result.candidate_stage.hits[:2])
        self.assertIsNone(result.candidate_index)
        self.assertEqual(result.stage2.stage_name, "noop_topk")
        self.assertEqual(result.stage2.query_vector_count, 0)
        self.assertEqual(result.stage2.document_vector_count, 0)

    def test_clause_text_stage2_can_refine_exact_candidate_window(self) -> None:
        query, index = self._build_clause_text_fixture()
        plan = kayak.exact_full_scan_clause_text_search_plan(final_k=1, candidate_k=2)

        result = kayak.search_with_plan(query, index, plan)

        self.assertEqual(result.candidate_stage.candidate_doc_ids, ("doc-context", "doc-answer"))
        self.assertEqual([hit.doc_id for hit in result.hits], ["doc-answer"])
        self.assertIsNotNone(result.candidate_index)
        assert result.candidate_index is not None
        self.assertEqual(result.candidate_index.doc_texts, index.doc_texts)
        self.assertEqual(result.stage2.stage_name, "clause_text")
        self.assertEqual(result.stage2.query_vector_count, 0)
        self.assertEqual(result.stage2.document_vector_count, 0)
        self.assertEqual(result.stage2.document_text_count, 2)

    def test_clause_text_stage2_requires_query_text_and_document_texts(self) -> None:
        query, index = self._build_clause_text_fixture()
        plan = kayak.exact_full_scan_clause_text_search_plan(final_k=1, candidate_k=2)

        with self.assertRaisesRegex(ValueError, "requires query.text"):
            kayak.search_with_plan(query.with_text(None), index, plan)

        with self.assertRaisesRegex(ValueError, "requires document texts"):
            kayak.search_with_plan(query, index.with_texts(None), plan)

    def test_index_search_with_plan_matches_top_level_helper(self) -> None:
        query, index = self._build_fixture()
        plan = kayak.exact_full_scan_search_plan(final_k=2, candidate_k=3)

        from_function = kayak.search_with_plan(query, index, plan)
        from_method = index.search_with_plan(query, plan=plan)

        self.assertEqual(from_function.hits, from_method.hits)
        self.assertEqual(from_function.candidate_stage.hits, from_method.candidate_stage.hits)
        self.assertEqual(from_function.exact_stage, from_method.exact_stage)

    def test_zero_width_plan_returns_empty_final_hits(self) -> None:
        query, index = self._build_fixture()
        plan = kayak.exact_full_scan_search_plan(final_k=0, candidate_k=0)

        result = kayak.search_with_plan(query, index, plan)

        self.assertEqual(result.candidate_stage.hits, ())
        self.assertIsNone(result.candidate_index)
        self.assertIsNone(result.exact_scores)
        self.assertEqual(result.hits, ())
        self.assertEqual(result.exact_stage.output_hit_count, 0)

    def test_candidate_generator_validation_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported candidate generator"):
            kayak.CandidateGenerator("centroid_postings")

        with self.assertRaisesRegex(ValueError, "non-negative"):
            kayak.document_proxy_candidate_generator(query_vector_budget=-1)

        with self.assertRaisesRegex(ValueError, "does not accept"):
            kayak.CandidateGenerator("exact_full_scan", query_vector_budget=1)

        with self.assertRaisesRegex(ValueError, "unsupported stage-2 operator"):
            kayak.Stage2Operator("dense_mlp")


if __name__ == "__main__":
    unittest.main()
