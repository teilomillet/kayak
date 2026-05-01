from __future__ import annotations

import unittest

import numpy as np

from kayak_bridge import documents, query
from kayak_bridge.cache_paths import REPO_ROOT
from kayak_bridge.tachiom_probe import (
    TachiomHnswConfig,
    TachiomResidualPqConfig,
    TachiomResidualPqIndex,
    TachiomResidualPqMojoIndex,
    TachiomTacConfig,
    TachiomTacHnswIndex,
    TachiomTacHnswMojoIndex,
    TachiomTacHnswResidualPqIndex,
    TachiomTacI8MojoIndex,
    TachiomTacIndex,
    TachiomTacMojoIndex,
    allocate_tac_centroid_counts,
    exact_search_positions,
    mean_candidate_set_recall_at_k,
    mean_recall_at_k,
)


def _unit_rows(values: list[list[float]]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, np.float32(1e-12))


class TachiomProbeTests(unittest.TestCase):
    @staticmethod
    def _mojo_backend_available() -> bool:
        import shutil

        if shutil.which("mojo") is not None:
            return True
        return (REPO_ROOT / ".pixi" / "envs" / "default" / "bin" / "mojo").exists()

    def test_token_ids_must_align_with_document_vectors(self) -> None:
        documents = np.zeros((2, 3, 4), dtype=np.float32)
        token_ids = np.zeros((2, 2), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "token_ids must align"):
            TachiomTacIndex.build(
                doc_ids=("a", "b"),
                documents=documents,
                token_ids=token_ids,
                config=TachiomTacConfig(centroid_count=4, candidate_k=2),
                final_k=2,
            )

    def test_allocation_preserves_tail_tokens_when_budget_allows(self) -> None:
        token_values = _unit_rows(
            [
                [1, 0, 0, 0],
                [1, 0.1, 0, 0],
                [1, -0.1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0.1, 0],
                [0, 0, 1, 0],
            ]
        )
        token_ids = np.asarray([10, 10, 10, 20, 20, 99], dtype=np.int64)
        allocation = allocate_tac_centroid_counts(
            token_values=token_values,
            token_ids=token_ids,
            config=TachiomTacConfig(
                centroid_count=5,
                micro_token_threshold=2,
                small_token_threshold=4,
                min_vectors_per_centroid=2,
                candidate_k=2,
            ),
        )

        self.assertEqual(sum(allocation.values()), 5)
        self.assertEqual(allocation[99], 1)
        self.assertGreaterEqual(allocation[10], 1)
        self.assertGreaterEqual(allocation[20], 1)

    def test_full_candidate_window_matches_exact_reference(self) -> None:
        documents = np.asarray(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [[1, 0, 0, 0], [0, 0, 1, 0]],
                [[0, 1, 0, 0], [0, 0, 1, 0]],
            ],
            dtype=np.float32,
        )
        token_ids = np.asarray(
            [
                [1, 2],
                [1, 3],
                [2, 3],
            ],
            dtype=np.int64,
        )
        queries = np.asarray(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [[0, 0, 1, 0], [1, 0, 0, 0]],
            ],
            dtype=np.float32,
        )
        index = TachiomTacIndex.build(
            doc_ids=("doc-a", "doc-b", "doc-c"),
            documents=documents,
            token_ids=token_ids,
            config=TachiomTacConfig(centroid_count=6, candidate_k=3),
            final_k=2,
        )

        self.assertEqual(index.exact_rerank_vector_bytes, documents.nbytes)
        self.assertEqual(
            index.index_bytes,
            index.sidecar_index_bytes + index.exact_rerank_vector_bytes,
        )
        self.assertEqual(
            index.search_batch_positions(queries, final_k=2),
            exact_search_positions(documents=documents, queries=queries, final_k=2),
        )

    def test_token_aware_candidates_have_nonzero_recall_on_clustered_fixture(self) -> None:
        documents = np.asarray(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0]],
                [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]],
                [[0, 0, -1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]],
            ],
            dtype=np.float32,
        )
        token_ids = np.asarray(
            [
                [1, 2, 3],
                [1, 2, 4],
                [5, 6, 3],
                [4, 6, 5],
            ],
            dtype=np.int64,
        )
        queries = np.asarray(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [[-1, 0, 0, 0], [0, -1, 0, 0]],
            ],
            dtype=np.float32,
        )
        index = TachiomTacIndex.build(
            doc_ids=("doc-a", "doc-b", "doc-c", "doc-d"),
            documents=documents,
            token_ids=token_ids,
            config=TachiomTacConfig(
                centroid_count=6,
                centroids_per_query_vector=1,
                candidate_k=2,
            ),
            final_k=1,
        )
        candidate_positions = index.candidate_positions_batch(queries)
        reference_positions = exact_search_positions(
            documents=documents,
            queries=queries,
            final_k=1,
        )

        self.assertGreater(
            mean_candidate_set_recall_at_k(
                candidate_positions_by_query=candidate_positions,
                reference_positions_by_query=reference_positions,
                k=1,
            ),
            0.0,
        )

    def test_candidate_set_recall_uses_whole_window_not_ordered_prefix(self) -> None:
        self.assertEqual(
            mean_candidate_set_recall_at_k(
                candidate_positions_by_query=((9, 8, 7, 6, 5, 4, 3, 2, 1, 0),),
                reference_positions_by_query=((0, 1),),
                k=2,
            ),
            1.0,
        )
        self.assertEqual(
            mean_recall_at_k(
                candidate_positions_by_query=((9, 8, 7, 6, 5, 4, 3, 2, 1, 0),),
                reference_positions_by_query=((0, 1),),
                k=2,
            ),
            0.0,
        )

    def test_candidate_pruning_keeps_final_k_and_drops_score_tail(self) -> None:
        documents_array = np.asarray(
            [
                [[1.0, 0.0]],
                [[0.8, 0.0]],
                [[0.2, 0.0]],
                [[-1.0, 0.0]],
            ],
            dtype=np.float32,
        )
        token_ids = np.asarray([[10], [20], [30], [40]], dtype=np.int64)
        queries = np.asarray([[[1.0, 0.0]]], dtype=np.float32)

        unpruned = TachiomTacIndex.build(
            doc_ids=("doc-a", "doc-b", "doc-c", "doc-d"),
            documents=documents_array,
            token_ids=token_ids,
            config=TachiomTacConfig(centroid_count=4, candidate_k=3),
            final_k=2,
        )
        pruned = TachiomTacIndex.build(
            doc_ids=("doc-a", "doc-b", "doc-c", "doc-d"),
            documents=documents_array,
            token_ids=token_ids,
            config=TachiomTacConfig(
                centroid_count=4,
                candidate_k=3,
                candidate_pruning_alpha=0.5,
            ),
            final_k=2,
        )

        self.assertEqual(unpruned.candidate_positions_batch(queries), ((0, 1, 2),))
        self.assertEqual(
            pruned.candidate_positions_batch(queries, final_k=2),
            ((0, 1),),
        )
        with self.assertRaisesRegex(ValueError, "final_k is required"):
            pruned.candidate_positions_batch(queries)
        self.assertEqual(
            pruned.search_batch_positions(queries, final_k=2),
            ((0, 1),),
        )

    def test_candidate_pruning_alpha_must_be_probability_like(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate_pruning_alpha"):
            TachiomTacConfig(
                centroid_count=4,
                candidate_k=2,
                candidate_pruning_alpha=1.0,
            ).validate(final_k=1)

    def test_centroid_hnsw_can_match_exact_tac_on_small_dense_graph(self) -> None:
        documents_array = np.asarray(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [[1, 0, 0, 0], [0, 0, 1, 0]],
                [[0, 1, 0, 0], [0, 0, 1, 0]],
                [[-1, 0, 0, 0], [0, -1, 0, 0]],
            ],
            dtype=np.float32,
        )
        token_ids = np.asarray(
            [[10, 20], [10, 30], [20, 30], [40, 50]],
            dtype=np.int64,
        )
        queries = np.asarray(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [[0, 0, 1, 0], [1, 0, 0, 0]],
            ],
            dtype=np.float32,
        )
        tac_index = TachiomTacIndex.build(
            doc_ids=("doc-a", "doc-b", "doc-c", "doc-d"),
            documents=documents_array,
            token_ids=token_ids,
            config=TachiomTacConfig(
                centroid_count=8,
                centroids_per_query_vector=4,
                candidate_k=3,
            ),
            final_k=2,
        )
        hnsw_index = TachiomTacHnswIndex.from_tac_index(
            tac_index,
            config=TachiomHnswConfig(
                max_neighbors=8,
                ef_construction=8,
                ef_search=8,
            ),
        )

        self.assertGreater(hnsw_index.graph.graph_bytes, 0)
        self.assertEqual(
            hnsw_index.candidate_positions_batch(queries, final_k=2),
            tac_index.candidate_positions_batch(queries, final_k=2),
        )
        self.assertEqual(
            hnsw_index.search_batch_positions(queries, final_k=2),
            tac_index.search_batch_positions(queries, final_k=2),
        )

    def test_from_late_index_requires_aligned_token_ids(self) -> None:
        late_index = documents(
            ["doc-a"],
            [np.asarray([[1, 0, 0, 0]], dtype=np.float32)],
        ).pack()

        with self.assertRaisesRegex(ValueError, "requires index token_ids"):
            TachiomTacIndex.from_late_index(
                late_index,
                config=TachiomTacConfig(centroid_count=1, candidate_k=1),
                final_k=1,
            )

    def test_from_late_index_handles_ragged_document_vector_counts(self) -> None:
        late_index = documents(
            ["doc-a", "doc-b", "doc-c"],
            [
                np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
                np.asarray([[0, 0, 1, 0]], dtype=np.float32),
                np.asarray(
                    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    dtype=np.float32,
                ),
            ],
            token_ids=[[10, 20], [30], [10, 30, 40]],
        ).pack()
        queries = np.asarray(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [[0, 0, 1, 0], [0, 0, 0, 1]],
            ],
            dtype=np.float32,
        )
        tac_index = TachiomTacIndex.from_late_index(
            late_index,
            config=TachiomTacConfig(centroid_count=6, candidate_k=3),
            final_k=2,
        )

        self.assertEqual(
            tac_index.search_batch_positions(queries, final_k=2),
            ((0, 2), (2, 1)),
        )
        hits = tac_index.search_query(query(queries[1]), final_k=2)
        self.assertEqual([hit.doc_id for hit in hits], ["doc-c", "doc-b"])
        self.assertEqual([hit.score for hit in hits], [2.0, 1.0])
        with self.assertRaisesRegex(ValueError, "not regular"):
            _ = tac_index.document_vector_count

    def test_residual_pq_refine_preserves_zero_residual_fixture(self) -> None:
        documents_array = np.asarray(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [[1, 0, 0, 0], [0, 0, 1, 0]],
                [[0, 1, 0, 0], [0, 0, 1, 0]],
            ],
            dtype=np.float32,
        )
        token_ids = np.asarray(
            [
                [10, 20],
                [10, 30],
                [20, 30],
            ],
            dtype=np.int64,
        )
        queries = np.asarray(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [[0, 0, 1, 0], [1, 0, 0, 0]],
            ],
            dtype=np.float32,
        )
        tac_index = TachiomTacIndex.build(
            doc_ids=("doc-a", "doc-b", "doc-c"),
            documents=documents_array,
            token_ids=token_ids,
            config=TachiomTacConfig(centroid_count=6, candidate_k=3),
            final_k=2,
        )
        pq_index = TachiomResidualPqIndex.from_tac_index(
            tac_index,
            config=TachiomResidualPqConfig(
                subspace_count=2,
                codebook_size=4,
                kmeans_iterations=2,
            ),
        )

        self.assertEqual(pq_index.effective_codebook_size, 4)
        self.assertEqual(pq_index.pq_codes.shape, (6, 2))
        self.assertEqual(pq_index.pq_codebooks.shape, (2, 4, 2))
        self.assertTrue(np.all(np.isfinite(pq_index.residual_norms)))
        self.assertEqual(
            pq_index.candidate_positions_batch(queries),
            tac_index.candidate_positions_batch(queries),
        )
        self.assertEqual(
            pq_index.search_batch_positions(queries, final_k=2),
            exact_search_positions(
                documents=documents_array,
                queries=queries,
                final_k=2,
            ),
        )
        self.assertLess(pq_index.index_bytes, tac_index.index_bytes)

    def test_hnsw_residual_pq_composition_uses_hnsw_candidates(self) -> None:
        documents_array = np.asarray(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [[1, 0, 0, 0], [0, 0, 1, 0]],
                [[0, 1, 0, 0], [0, 0, 1, 0]],
            ],
            dtype=np.float32,
        )
        token_ids = np.asarray(
            [[10, 20], [10, 30], [20, 30]],
            dtype=np.int64,
        )
        queries = np.asarray(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [[0, 0, 1, 0], [1, 0, 0, 0]],
            ],
            dtype=np.float32,
        )
        tac_index = TachiomTacIndex.build(
            doc_ids=("doc-a", "doc-b", "doc-c"),
            documents=documents_array,
            token_ids=token_ids,
            config=TachiomTacConfig(centroid_count=6, candidate_k=3),
            final_k=2,
        )
        full_index = TachiomTacHnswResidualPqIndex.from_tac_index(
            tac_index,
            hnsw_config=TachiomHnswConfig(
                max_neighbors=4,
                ef_construction=4,
                ef_search=4,
            ),
            pq_config=TachiomResidualPqConfig(
                subspace_count=2,
                codebook_size=4,
                kmeans_iterations=2,
            ),
        )

        self.assertEqual(
            full_index.candidate_positions_batch(queries, final_k=2),
            full_index.hnsw_index.candidate_positions_batch(queries, final_k=2),
        )
        self.assertEqual(
            full_index.search_batch_positions(queries, final_k=2),
            exact_search_positions(
                documents=documents_array,
                queries=queries,
                final_k=2,
            ),
        )
        self.assertGreater(full_index.index_bytes, full_index.pq_index.index_bytes)

    def test_residual_pq_config_requires_subspaces_dividing_vector_dim(self) -> None:
        documents_array = np.asarray(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [[0, 0, 1, 0], [0, 0, 0, 1]],
            ],
            dtype=np.float32,
        )
        token_ids = np.asarray([[1, 2], [3, 4]], dtype=np.int64)
        tac_index = TachiomTacIndex.build(
            doc_ids=("doc-a", "doc-b"),
            documents=documents_array,
            token_ids=token_ids,
            config=TachiomTacConfig(centroid_count=4, candidate_k=2),
            final_k=1,
        )

        with self.assertRaisesRegex(ValueError, "subspace_count must divide"):
            TachiomResidualPqIndex.from_tac_index(
                tac_index,
                config=TachiomResidualPqConfig(subspace_count=3),
            )

    def test_residual_pq_training_sample_reduces_effective_codebook(self) -> None:
        documents_array = np.asarray(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ],
            dtype=np.float32,
        )
        token_ids = np.asarray([[1, 2, 3], [1, 3, 4]], dtype=np.int64)
        tac_index = TachiomTacIndex.build(
            doc_ids=("doc-a", "doc-b"),
            documents=documents_array,
            token_ids=token_ids,
            config=TachiomTacConfig(centroid_count=6, candidate_k=2),
            final_k=1,
        )
        pq_index = TachiomResidualPqIndex.from_tac_index(
            tac_index,
            config=TachiomResidualPqConfig(
                subspace_count=2,
                codebook_size=4,
                training_sample_count=2,
            ),
        )

        self.assertEqual(pq_index.effective_codebook_size, 2)
        self.assertEqual(pq_index.pq_codes.shape, (6, 2))
        self.assertLessEqual(int(pq_index.pq_codes.max()), 1)

    def test_residual_pq_handles_token_ids_trimmed_by_centroid_budget(self) -> None:
        documents_array = np.asarray(
            [
                [[1, 0, 0, 0]],
                [[0, 1, 0, 0]],
                [[0, 0, 1, 0]],
            ],
            dtype=np.float32,
        )
        token_ids = np.asarray([[10], [20], [30]], dtype=np.int64)
        tac_index = TachiomTacIndex.build(
            doc_ids=("doc-a", "doc-b", "doc-c"),
            documents=documents_array,
            token_ids=token_ids,
            config=TachiomTacConfig(centroid_count=2, candidate_k=3),
            final_k=1,
        )

        pq_index = TachiomResidualPqIndex.from_tac_index(
            tac_index,
            config=TachiomResidualPqConfig(
                subspace_count=2,
                codebook_size=2,
                kmeans_iterations=2,
            ),
        )

        self.assertEqual(pq_index.token_centroid_positions.shape, (3,))
        self.assertTrue(
            np.all(pq_index.token_centroid_positions < tac_index.centroid_count)
        )

    @unittest.skipUnless(
        _mojo_backend_available.__func__(), "Mojo TAC requires Mojo"
    )
    def test_mojo_tac_candidate_pruning_matches_python_reference(self) -> None:
        documents_array = np.zeros((4, 1, 128), dtype=np.float32)
        documents_array[0, 0, 0] = 1.0
        documents_array[1, 0, 0] = 0.8
        documents_array[2, 0, 0] = 0.2
        documents_array[3, 0, 0] = -1.0
        token_ids = np.asarray([[10], [20], [30], [40]], dtype=np.int64)
        queries = np.zeros((1, 1, 128), dtype=np.float32)
        queries[0, 0, 0] = 1.0
        python_index = TachiomTacIndex.build(
            doc_ids=("doc-a", "doc-b", "doc-c", "doc-d"),
            documents=documents_array,
            token_ids=token_ids,
            config=TachiomTacConfig(
                centroid_count=4,
                candidate_k=3,
                candidate_pruning_alpha=0.5,
            ),
            final_k=2,
        )
        mojo_index = TachiomTacMojoIndex.from_tac_index(python_index)
        pq_index = TachiomResidualPqIndex.from_tac_index(
            python_index,
            config=TachiomResidualPqConfig(
                subspace_count=32,
                codebook_size=4,
                kmeans_iterations=2,
            ),
        )
        pq_mojo_index = TachiomResidualPqMojoIndex.from_pq_index(pq_index)

        self.assertEqual(
            mojo_index.candidate_positions_batch(queries, final_k=2),
            python_index.candidate_positions_batch(queries, final_k=2),
        )
        self.assertEqual(
            mojo_index.search_batch_positions(queries, final_k=2),
            python_index.search_batch_positions(queries, final_k=2),
        )
        self.assertEqual(
            pq_mojo_index.candidate_positions_batch(queries, final_k=2),
            pq_index.candidate_positions_batch(queries, final_k=2),
        )
        self.assertEqual(
            pq_mojo_index.search_batch_positions(queries, final_k=2),
            pq_index.search_batch_positions(queries, final_k=2),
        )

    @unittest.skipUnless(
        _mojo_backend_available.__func__(), "Mojo TAC requires Mojo"
    )
    def test_mojo_tac_hnsw_matches_python_hnsw_on_dim128_fixture(self) -> None:
        documents_array = np.zeros((4, 1, 128), dtype=np.float32)
        documents_array[0, 0, 0] = 1.0
        documents_array[1, 0, 0] = 0.8
        documents_array[2, 0, 0] = 0.2
        documents_array[3, 0, 0] = -1.0
        token_ids = np.asarray([[10], [20], [30], [40]], dtype=np.int64)
        queries = np.zeros((1, 1, 128), dtype=np.float32)
        queries[0, 0, 0] = 1.0
        tac_index = TachiomTacIndex.build(
            doc_ids=("doc-a", "doc-b", "doc-c", "doc-d"),
            documents=documents_array,
            token_ids=token_ids,
            config=TachiomTacConfig(
                centroid_count=4,
                candidate_k=3,
                candidate_pruning_alpha=0.5,
            ),
            final_k=2,
        )
        hnsw_index = TachiomTacHnswIndex.from_tac_index(
            tac_index,
            config=TachiomHnswConfig(
                max_neighbors=4,
                ef_construction=4,
                ef_search=4,
            ),
        )
        mojo_index = TachiomTacHnswMojoIndex.from_hnsw_index(hnsw_index)

        self.assertEqual(
            mojo_index.candidate_positions_batch(queries, final_k=2),
            hnsw_index.candidate_positions_batch(queries, final_k=2),
        )
        self.assertEqual(
            mojo_index.search_batch_positions(queries, final_k=2),
            hnsw_index.search_batch_positions(queries, final_k=2),
        )

    @unittest.skipUnless(
        _mojo_backend_available.__func__(), "Mojo TAC requires Mojo"
    )
    def test_mojo_tac_matches_python_tac_on_dim128_fixture(self) -> None:
        documents_array = np.zeros((4, 3, 128), dtype=np.float32)
        documents_array[0, 0, 0] = 1.0
        documents_array[0, 1, 1] = 1.0
        documents_array[0, 2, 2] = 1.0
        documents_array[1, 0, 0] = 1.0
        documents_array[1, 1, 1] = 0.5
        documents_array[1, 2, 2] = 0.5
        documents_array[2, 0, 3] = 1.0
        documents_array[2, 1, 4] = 1.0
        documents_array[2, 2, 5] = 1.0
        documents_array[3, 0, 0] = -1.0
        documents_array[3, 1, 1] = -1.0
        documents_array[3, 2, 2] = -1.0
        token_ids = np.asarray(
            [[10, 20, 30], [10, 20, 30], [40, 50, 60], [10, 20, 30]],
            dtype=np.int64,
        )
        queries = np.zeros((2, 2, 128), dtype=np.float32)
        queries[0, 0, 0] = 1.0
        queries[0, 1, 1] = 1.0
        queries[1, 0, 3] = 1.0
        queries[1, 1, 4] = 1.0
        python_index = TachiomTacIndex.build(
            doc_ids=("doc-a", "doc-b", "doc-c", "doc-d"),
            documents=documents_array,
            token_ids=token_ids,
            config=TachiomTacConfig(
                centroid_count=6,
                centroids_per_query_vector=2,
                candidate_k=3,
            ),
            final_k=2,
        )
        mojo_index = TachiomTacMojoIndex.from_tac_index(python_index)
        i8_index = TachiomTacI8MojoIndex.from_tac_index(python_index)
        pq_index = TachiomResidualPqIndex.from_tac_index(
            python_index,
            config=TachiomResidualPqConfig(
                subspace_count=32,
                codebook_size=4,
                kmeans_iterations=2,
            ),
        )
        pq_mojo_index = TachiomResidualPqMojoIndex.from_pq_index(pq_index)

        self.assertEqual(
            mojo_index.candidate_positions_batch(queries),
            python_index.candidate_positions_batch(queries),
        )
        self.assertEqual(
            i8_index.candidate_positions_batch(queries),
            python_index.candidate_positions_batch(queries),
        )
        self.assertEqual(
            mojo_index.search_batch_positions(queries, final_k=2),
            python_index.search_batch_positions(queries, final_k=2),
        )
        self.assertEqual(
            i8_index.search_batch_positions(queries, final_k=2),
            python_index.search_batch_positions(queries, final_k=2),
        )
        self.assertEqual(
            pq_mojo_index.candidate_positions_batch(queries),
            pq_index.candidate_positions_batch(queries),
        )
        self.assertEqual(
            pq_mojo_index.search_batch_positions(queries, final_k=2),
            pq_index.search_batch_positions(queries, final_k=2),
        )
        self.assertLess(pq_mojo_index.index_bytes, python_index.index_bytes)
        self.assertLess(i8_index.index_bytes, python_index.index_bytes)
        profiles = mojo_index.profile_candidate_generation_batch(
            queries,
            final_k=2,
            measurement_iterations=1,
        )
        self.assertEqual(len(profiles), 2)
        self.assertEqual(profiles[0]["document_count"], 4)
        self.assertEqual(profiles[0]["query_vector_count"], 2)
        self.assertEqual(profiles[0]["candidate_k"], 3)
        self.assertEqual(profiles[0]["final_k"], 2)
        self.assertGreater(profiles[0]["full_search_mean_seconds"], 0.0)
        self.assertGreater(profiles[0]["exact_rerank_mean_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
