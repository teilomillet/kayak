from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from kayak_bridge.cache_paths import REPO_ROOT
from kayak_bridge.encoded_snapshot import (
    EncodedDocumentShardWriter,
    write_snapshot_manifest,
)
from kayak_bridge.tachiom_pq import TachiomResidualPqConfig
from kayak_bridge.tachiom_hnsw import TachiomHnswConfig
from kayak_bridge.tachiom_streaming_hnsw import build_streaming_tachiom_hnsw_graph
from kayak_bridge.tachiom_streaming_index import (
    StreamingTachiomBuildConfig,
    build_streaming_tachiom_index,
)
from kayak_bridge.tachiom_streaming_benchmark import (
    _same_shape_query_groups,
    _with_centroids_per_query_vector,
    _with_candidate_pruning_alpha,
    _with_hnsw_ef_search,
)
from kayak_bridge.tachiom_streaming_search import (
    load_streaming_tachiom_hnsw_pq_mojo_address_index,
    load_streaming_tachiom_hnsw_pq_mojo_index,
    load_streaming_tachiom_hnsw_pq_index,
    load_streaming_tachiom_pq_index,
)
from kayak_bridge.tachiom_types import TachiomTacConfig


def _write_streaming_snapshot(root: Path) -> Path:
    snapshot = root / "snapshot"
    snapshot.mkdir()
    (snapshot / "shards").mkdir()
    writer = EncodedDocumentShardWriter(
        snapshot_root=snapshot,
        shard_index=0,
        vector_dim=4,
        vector_dtype="float16",
        token_id_dtype="uint32",
    )
    writer.append(
        doc_id="d0",
        vectors=np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
            dtype=np.float32,
        ),
        token_ids=np.asarray([10, 10], dtype=np.int64),
    )
    writer.append(
        doc_id="d1",
        vectors=np.asarray(
            [[0.0, 1.0, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]],
            dtype=np.float32,
        ),
        token_ids=np.asarray([20, 20], dtype=np.int64),
    )
    shard = writer.close()
    write_snapshot_manifest(
        snapshot_root=snapshot,
        dataset_id="dataset://streaming",
        model_name="unit-model",
        vector_dim=4,
        vector_dtype="float16",
        token_id_dtype="uint32",
        shards=[shard],
        source={"unit": True},
    )
    return snapshot


def _write_streaming_snapshot_dim128(root: Path) -> Path:
    snapshot = root / "snapshot"
    snapshot.mkdir()
    (snapshot / "shards").mkdir()
    writer = EncodedDocumentShardWriter(
        snapshot_root=snapshot,
        shard_index=0,
        vector_dim=128,
        vector_dtype="float16",
        token_id_dtype="uint32",
    )
    left = np.zeros((2, 128), dtype=np.float32)
    left[0, 0] = 1.0
    left[1, 1] = 1.0
    right = np.zeros((2, 128), dtype=np.float32)
    right[0, 2] = 1.0
    right[1, 3] = 1.0
    writer.append(
        doc_id="d0",
        vectors=left,
        token_ids=np.asarray([10, 20], dtype=np.int64),
    )
    writer.append(
        doc_id="d1",
        vectors=right,
        token_ids=np.asarray([30, 40], dtype=np.int64),
    )
    shard = writer.close()
    write_snapshot_manifest(
        snapshot_root=snapshot,
        dataset_id="dataset://streaming-dim128",
        model_name="unit-model",
        vector_dim=128,
        vector_dtype="float16",
        token_id_dtype="uint32",
        shards=[shard],
        source={"unit": True},
    )
    return snapshot


class TachiomStreamingIndexTests(unittest.TestCase):
    @staticmethod
    def _mojo_backend_available() -> bool:
        import shutil

        if shutil.which("mojo") is not None:
            return True
        return (REPO_ROOT / ".pixi" / "envs" / "default" / "bin" / "mojo").exists()

    def test_streaming_builder_writes_tac_and_pq_payloads(self) -> None:
        with TemporaryDirectory() as root:
            root_path = Path(root)
            snapshot = _write_streaming_snapshot(root_path)
            output = root_path / "index"
            summary = build_streaming_tachiom_index(
                snapshot_root=snapshot,
                output_root=output,
                config=StreamingTachiomBuildConfig(
                    tac=TachiomTacConfig(
                        centroid_count=4,
                        micro_token_threshold=1,
                        small_token_threshold=2,
                        active_token_floor=1,
                        min_vectors_per_centroid=1,
                        kmeans_iterations=2,
                        centroids_per_query_vector=2,
                        candidate_k=2,
                    ),
                    pq=TachiomResidualPqConfig(
                        subspace_count=2,
                        codebook_size=2,
                        kmeans_iterations=2,
                        training_sample_count=4,
                    ),
                    centroid_samples_per_centroid=2,
                    posting_partition_count=2,
                ),
            )
            posting_offsets = np.load(output / "centroid_doc_posting_offsets.u64.npy")
            postings = np.fromfile(output / "centroid_doc_postings.u32", dtype=np.uint32)
            codes = np.fromfile(output / "pq_codes.u8", dtype=np.uint8)
            index = load_streaming_tachiom_pq_index(output)
            pruned_index = _with_candidate_pruning_alpha(index, 0.25)
            unpruned_index = _with_candidate_pruning_alpha(pruned_index, None)
            rankings = index.search_batch_positions(
                np.asarray(
                    [
                        [[1.0, 0.0, 0.0, 0.0]],
                        [[0.0, 1.0, 0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
                final_k=1,
            )
            graph_summary = build_streaming_tachiom_hnsw_graph(
                index_root=output,
                config=TachiomHnswConfig(
                    max_neighbors=2,
                    ef_construction=2,
                    ef_search=4,
                    level_probability=0.25,
                ),
            )
            hnsw_index = load_streaming_tachiom_hnsw_pq_index(output)
            pruned_hnsw_index = _with_candidate_pruning_alpha(hnsw_index, 0.25)
            wider_hnsw_index = _with_centroids_per_query_vector(hnsw_index, 3)
            deeper_hnsw_index = _with_hnsw_ef_search(hnsw_index, 8)
            hnsw_rankings = hnsw_index.search_batch_positions(
                np.asarray(
                    [
                        [[1.0, 0.0, 0.0, 0.0]],
                        [[0.0, 1.0, 0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
                final_k=1,
            )

            self.assertEqual(summary.document_count, 2)
            self.assertEqual(summary.document_vector_count, 4)
            self.assertEqual(summary.centroid_count, 4)
            self.assertEqual(summary.sample_vector_count, 4)
            self.assertTrue(summary.pq_enabled)
            self.assertLess(summary.index_payload_bytes, summary.build_payload_bytes)
            self.assertEqual(index.index_bytes, summary.index_payload_bytes)
            self.assertEqual(pruned_index.candidate_pruning_alpha, 0.25)
            self.assertIsNone(unpruned_index.candidate_pruning_alpha)
            self.assertEqual(pruned_hnsw_index.base_index.candidate_pruning_alpha, 0.25)
            self.assertIs(pruned_hnsw_index.graph, hnsw_index.graph)
            self.assertEqual(
                wider_hnsw_index.base_index.centroids_per_query_vector,
                3,
            )
            self.assertIs(wider_hnsw_index.graph, hnsw_index.graph)
            self.assertEqual(deeper_hnsw_index.graph.ef_search, 8)
            self.assertIs(deeper_hnsw_index.base_index, hnsw_index.base_index)
            self.assertEqual(posting_offsets.shape, (5,))
            self.assertEqual(int(posting_offsets[-1]), int(postings.shape[0]))
            self.assertEqual(codes.shape[0], 8)
            self.assertEqual(rankings, ((0,), (1,)))
            self.assertEqual(graph_summary.centroid_count, summary.centroid_count)
            self.assertGreater(graph_summary.edge_count, 0)
            self.assertEqual(hnsw_rankings, ((0,), (1,)))

    def test_same_shape_query_groups_respect_batch_cap(self) -> None:
        queries = [
            np.zeros((2, 4), dtype=np.float32),
            np.zeros((1, 4), dtype=np.float32),
            np.zeros((2, 4), dtype=np.float32),
            np.zeros((2, 4), dtype=np.float32),
            np.zeros((1, 4), dtype=np.float32),
        ]

        groups = _same_shape_query_groups(queries, max_query_batch_size=2)

        self.assertEqual(groups, ((0, 2), (3,), (1, 4)))

    @unittest.skipUnless(
        _mojo_backend_available.__func__(), "Mojo streaming TAC/HNSW/PQ requires Mojo"
    )
    def test_streaming_hnsw_pq_mojo_matches_python_on_dim128_fixture(self) -> None:
        with TemporaryDirectory() as root:
            root_path = Path(root)
            snapshot = _write_streaming_snapshot_dim128(root_path)
            output = root_path / "index"
            build_streaming_tachiom_index(
                snapshot_root=snapshot,
                output_root=output,
                config=StreamingTachiomBuildConfig(
                    tac=TachiomTacConfig(
                        centroid_count=4,
                        micro_token_threshold=1,
                        small_token_threshold=2,
                        active_token_floor=1,
                        min_vectors_per_centroid=1,
                        kmeans_iterations=2,
                        centroids_per_query_vector=2,
                        candidate_k=2,
                        candidate_pruning_alpha=0.5,
                    ),
                    pq=TachiomResidualPqConfig(
                        subspace_count=32,
                        codebook_size=2,
                        kmeans_iterations=2,
                        training_sample_count=4,
                    ),
                    centroid_samples_per_centroid=2,
                    posting_partition_count=2,
                ),
            )
            build_streaming_tachiom_hnsw_graph(
                index_root=output,
                config=TachiomHnswConfig(
                    max_neighbors=2,
                    ef_construction=2,
                    ef_search=4,
                    level_probability=0.25,
                ),
            )
            python_index = load_streaming_tachiom_hnsw_pq_index(output)
            mojo_index = load_streaming_tachiom_hnsw_pq_mojo_index(output)
            address_mojo_index = load_streaming_tachiom_hnsw_pq_mojo_address_index(
                output
            )
            queries = np.zeros((2, 1, 128), dtype=np.float32)
            queries[0, 0, 0] = 1.0
            queries[1, 0, 2] = 1.0

            self.assertEqual(
                mojo_index.candidate_positions_batch(queries, final_k=1),
                python_index.candidate_positions_batch(queries, final_k=1),
            )
            self.assertEqual(
                mojo_index.search_batch_positions(queries, final_k=1),
                python_index.search_batch_positions(queries, final_k=1),
            )
            self.assertEqual(
                address_mojo_index.candidate_positions_batch(queries, final_k=1),
                python_index.candidate_positions_batch(queries, final_k=1),
            )
            self.assertEqual(
                address_mojo_index.search_batch_positions(queries, final_k=1),
                python_index.search_batch_positions(queries, final_k=1),
            )
            self.assertEqual(
                mojo_index.index_kind,
                "streaming_token_aware_hnsw_centroid_postings_residual_pq_sparse_rerank_mojo",
            )
            self.assertEqual(
                address_mojo_index.index_kind,
                "streaming_token_aware_hnsw_centroid_postings_residual_pq_sparse_rerank_mojo_address",
            )
            profiles = mojo_index.profile_search_batch(
                queries,
                final_k=1,
                measurement_iterations=1,
            )
            self.assertEqual(len(profiles), 2)
            self.assertEqual(profiles[0]["document_count"], 2)
            self.assertEqual(profiles[0]["query_vector_count"], 1)
            self.assertEqual(profiles[0]["centroid_count"], 4)
            self.assertEqual(profiles[0]["candidate_k"], 2)
            self.assertEqual(profiles[0]["final_k"], 1)
            self.assertEqual(profiles[0]["ef_search"], 4)
            self.assertGreater(profiles[0]["full_search_mean_seconds"], 0.0)
            self.assertGreater(profiles[0]["hnsw_traversal_mean_seconds"], 0.0)
            self.assertGreater(
                profiles[0]["candidate_score_accumulation_mean_seconds"],
                0.0,
            )
            self.assertGreater(profiles[0]["rerank_scoring_mean_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
