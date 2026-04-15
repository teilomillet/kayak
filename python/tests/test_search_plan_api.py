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

    def _build_hybrid_stage2_fixture(self) -> tuple[kayak.LateQuery, kayak.LateIndex]:
        query = kayak.query(
            np.stack(
                [
                    _dim128_vector((0, 1.0)),
                    _dim128_vector((1, 1.0)),
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
                        _dim128_vector((1, 1.0)),
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

    def test_document_proxy_candidate_generation_matches_reference_when_full_budget(self) -> None:
        query, index = self._build_fixture()
        generator = kayak.document_proxy_candidate_generator()

        result = kayak.generate_candidates(query, index, generator, k=3)
        reference = self._reference_document_proxy_scores(
            query,
            index,
            query_vector_budget=generator.query_vector_budget,
            document_vector_budget=generator.document_vector_budget,
        )

        np.testing.assert_allclose(result.scores.numpy(), reference)
        self.assertEqual(result.profile.query_vector_count, 1)
        self.assertEqual(result.profile.document_vector_count, 3)

    def test_document_proxy_candidate_generation_matches_reference_when_budget_exceeds_lengths(self) -> None:
        query, index = self._build_fixture()
        generator = kayak.document_proxy_candidate_generator(
            query_vector_budget=16,
            document_vector_budget=16,
        )

        result = kayak.generate_candidates(query, index, generator, k=3)
        reference = self._reference_document_proxy_scores(
            query,
            index,
            query_vector_budget=generator.query_vector_budget,
            document_vector_budget=generator.document_vector_budget,
        )

        np.testing.assert_allclose(result.scores.numpy(), reference)
        self.assertEqual(result.profile.query_vector_count, 1)
        self.assertEqual(result.profile.document_vector_count, 3)

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
        self.assertEqual(result.exact_stage.document_count, 1)
        self.assertEqual(result.exact_stage.document_vector_count, 2)
        self.assertEqual(result.stage3_verifier.stage_name, "none")
        self.assertEqual(result.stage2.stage_name, "exact_late_interaction")
        self.assertEqual(len(result.stage2.materialized_artifacts), 1)
        self.assertEqual(
            result.stage2.materialized_artifacts[0].family,
            "late_interaction",
        )
        self.assertEqual(
            result.stage2.materialized_artifacts[0].document_vector_count,
            2,
        )
        self.assertIs(result.exact_stage, result.stage2)
        self.assertIs(result.exact_scores, result.stage2_scores)

    def test_exact_full_scan_plan_defaults_to_noop_stage2(self) -> None:
        query, index = self._build_fixture()
        plan = kayak.exact_full_scan_search_plan(final_k=2, candidate_k=3)

        result = kayak.search_with_plan(query, index, plan)

        self.assertEqual(result.plan.stage2_reference_operator.kind, "noop_topk")
        self.assertEqual(result.hits, result.candidate_stage.hits[:2])
        self.assertIsNone(result.candidate_index)
        self.assertEqual(result.stage2.stage_name, "noop_topk")
        self.assertEqual(result.stage3_verifier.stage_name, "none")
        self.assertEqual(result.stage2.query_vector_count, 0)
        self.assertEqual(result.stage2.document_vector_count, 0)
        self.assertEqual(result.stage2.materialized_artifacts, ())

    def _reference_document_proxy_scores(
        self,
        query: kayak.LateQuery,
        index: kayak.LateIndex,
        *,
        query_vector_budget: int,
        document_vector_budget: int,
    ) -> np.ndarray:
        query_matrix = query.as_vector_matrix()
        effective_query_budget = self._effective_budget(
            query_vector_budget,
            query.vector_count,
        )
        query_proxy = np.mean(
            query_matrix[:effective_query_budget],
            axis=0,
            dtype=np.float32,
        ).astype(np.float32, copy=False)

        token_matrix = index.as_packed_token_matrix()
        proxy_vectors = np.empty((index.document_count, index.vector_dim), dtype=np.float32)
        for document_index in range(index.document_count):
            start = int(index.doc_offsets[document_index])
            stop = int(index.doc_offsets[document_index + 1])
            effective_document_budget = self._effective_budget(
                document_vector_budget,
                stop - start,
            )
            proxy_vectors[document_index] = np.mean(
                token_matrix[start : start + effective_document_budget],
                axis=0,
                dtype=np.float32,
            )
        return np.matmul(proxy_vectors, query_proxy).astype(np.float32, copy=False)

    def _effective_budget(self, requested_budget: int, available_count: int) -> int:
        if requested_budget == 0 or requested_budget > available_count:
            return available_count
        return requested_budget

    def test_clause_text_stage2_can_refine_exact_candidate_window(self) -> None:
        query, index = self._build_clause_text_fixture()
        plan = kayak.exact_full_scan_search_plan(
            final_k=1,
            candidate_k=2,
            stage3_verifier=kayak.clause_text_stage3_verifier_operator(),
        )

        result = kayak.search_with_plan(query, index, plan)

        self.assertEqual(result.candidate_stage.candidate_doc_ids, ("doc-context", "doc-answer"))
        self.assertEqual([hit.doc_id for hit in result.hits], ["doc-answer"])
        self.assertIsNotNone(result.candidate_index)
        assert result.candidate_index is not None
        self.assertEqual(result.candidate_index.doc_texts, index.doc_texts)
        self.assertEqual(result.plan.stage2_reference_operator.kind, "noop_topk")
        self.assertEqual(result.plan.stage3_verifier.kind, "clause_text")
        self.assertEqual(result.stage2.stage_name, "noop_topk")
        self.assertEqual(result.stage3_verifier.stage_name, "clause_text")
        self.assertEqual(result.stage2.query_vector_count, 0)
        self.assertEqual(result.stage2.document_vector_count, 0)
        self.assertEqual(result.stage2.document_text_count, 0)
        self.assertEqual(len(result.stage2.materialized_artifacts), 0)
        self.assertEqual(result.stage3_verifier.document_text_count, 2)
        self.assertEqual(len(result.stage3_verifier.materialized_artifacts), 1)
        self.assertEqual(
            result.stage3_verifier.materialized_artifacts[0].family,
            "document_text",
        )
        self.assertEqual(
            result.stage3_verifier.materialized_artifacts[0].document_text_count,
            2,
        )

    def test_clause_text_stage2_requires_query_text_and_document_texts(self) -> None:
        query, index = self._build_clause_text_fixture()
        plan = kayak.exact_full_scan_search_plan(
            final_k=1,
            candidate_k=2,
            stage3_verifier=kayak.clause_text_stage3_verifier_operator(),
        )

        with self.assertRaisesRegex(ValueError, "requires query.text"):
            kayak.search_with_plan(query.with_text(None), index, plan)

        with self.assertRaisesRegex(ValueError, "requires document texts"):
            kayak.search_with_plan(query, index.with_texts(None), plan)

    def test_explicit_stage3_verifier_materializes_vectors_and_texts(self) -> None:
        query, index = self._build_hybrid_stage2_fixture()
        plan = kayak.document_proxy_search_plan(
            final_k=1,
            candidate_k=2,
            query_vector_budget=1,
            document_vector_budget=1,
            stage3_verifier=kayak.clause_text_stage3_verifier_operator(),
        )

        result = kayak.search_with_plan(query, index, plan)

        self.assertEqual(result.plan.reference_scoring_semantics.kind, "exact_late_interaction")
        self.assertEqual(result.plan.stage2_reference_operator.kind, "exact_late_interaction")
        self.assertEqual(result.plan.stage2_reference_operator.family, "late_interaction")
        self.assertEqual(
            result.plan.stage2_reference_operator.required_artifact_families,
            ("late_interaction",),
        )
        self.assertTrue(result.plan.stage2_reference_operator.executes_reference_scoring)
        self.assertEqual(result.plan.stage3_verifier.kind, "clause_text")
        self.assertEqual(result.plan.stage3_verifier.family, "text")
        self.assertEqual(
            result.plan.stage3_verifier.required_artifact_families,
            ("document_text",),
        )
        self.assertTrue(result.plan.stage3_verifier.requires_query_text)
        self.assertEqual(result.candidate_stage.candidate_doc_ids, ("doc-context", "doc-answer"))
        self.assertEqual([hit.doc_id for hit in result.hits], ["doc-answer"])
        self.assertEqual(result.stage2.stage_name, "exact_late_interaction")
        self.assertEqual(result.stage3_verifier.stage_name, "clause_text")
        self.assertEqual(result.stage2.query_vector_count, 2)
        self.assertEqual(result.stage2.document_vector_count, 4)
        self.assertEqual(result.stage2.document_text_count, 0)
        self.assertEqual(len(result.stage2.materialized_artifacts), 1)
        self.assertEqual(
            tuple(
                artifact.family
                for artifact in result.stage2.materialized_artifacts
            ),
            ("late_interaction",),
        )
        self.assertEqual(result.stage3_verifier.document_text_count, 2)
        self.assertEqual(
            tuple(
                artifact.family
                for artifact in result.stage3_verifier.materialized_artifacts
            ),
            ("document_text",),
        )

    def test_explicit_stage3_verifier_requires_query_text_and_document_texts(self) -> None:
        query, index = self._build_hybrid_stage2_fixture()
        plan = kayak.document_proxy_search_plan(
            final_k=1,
            candidate_k=2,
            query_vector_budget=1,
            document_vector_budget=1,
            stage3_verifier=kayak.clause_text_stage3_verifier_operator(),
        )

        with self.assertRaisesRegex(ValueError, "requires query.text"):
            kayak.search_with_plan(query.with_text(None), index, plan)

        with self.assertRaisesRegex(ValueError, "requires document texts"):
            kayak.search_with_plan(query, index.with_texts(None), plan)

    def test_plan_builders_reject_legacy_stage2_operator_keyword(self) -> None:
        with self.assertRaisesRegex(TypeError, "stage2_operator"):
            kayak.exact_full_scan_search_plan(
                final_k=1,
                candidate_k=2,
                stage2_operator="clause_text",
                stage3_verifier=kayak.clause_text_stage3_verifier_operator(),
            )

        with self.assertRaisesRegex(TypeError, "stage2_operator"):
            kayak.document_proxy_search_plan(
                final_k=1,
                candidate_k=2,
                stage2_operator="exact_late_interaction_clause_text",
            )

    def test_clause_text_shorthand_builder_matches_explicit_stage3_override(self) -> None:
        shorthand = kayak.exact_full_scan_clause_text_search_plan(
            final_k=1,
            candidate_k=2,
        )
        explicit = kayak.exact_full_scan_search_plan(
            final_k=1,
            candidate_k=2,
            stage3_verifier=kayak.clause_text_stage3_verifier_operator(),
        )

        self.assertEqual(shorthand, explicit)

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

        with self.assertRaisesRegex(
            ValueError, "unsupported stage-2 reference operator"
        ):
            kayak.Stage2ReferenceOperator("dense_mlp")


if __name__ == "__main__":
    unittest.main()
