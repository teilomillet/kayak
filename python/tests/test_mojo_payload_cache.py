from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import kayak
from kayak_bridge.mojo_payload_cache import (
    MOJO_INDEX_PAYLOAD_CACHE_SIZE,
    cached_index_payload,
    clear_mojo_index_payload_cache,
)


class _FakeMojoModule:
    def exact_scores_hybrid_flat_dim128_with_flat_query(
        self,
        query_payload: list[float],
        doc_ids: list[str],
        doc_offsets: list[int],
        packed_vectors: list[list[float]],
        flat_token_values: list[float],
    ) -> list[float]:
        del query_payload, doc_offsets, packed_vectors, flat_token_values
        return [0.0 for _ in doc_ids]


def _build_index(seed: int) -> kayak.LateIndex:
    rng = np.random.default_rng(seed)
    return (
        kayak.documents(
            [f"doc-{seed}-a", f"doc-{seed}-b"],
            [
                rng.standard_normal((2, 128)).astype(np.float32),
                rng.standard_normal((3, 128)).astype(np.float32),
            ],
        )
        .pack()
        .to_layout("hybrid_flat_dim128")
    )


class MojoPayloadCacheTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_mojo_index_payload_cache()

    def test_reuses_payload_for_same_index_object(self) -> None:
        index = _build_index(1)

        first = cached_index_payload(index)
        second = cached_index_payload(index)

        self.assertIs(first, second)
        self.assertEqual(first.doc_ids, list(index.doc_ids))
        self.assertIsNotNone(first.flat_token_values)

    def test_cache_is_scoped_to_index_identity(self) -> None:
        index = _build_index(2)
        equivalent_index = index.with_texts(index.doc_texts)

        first = cached_index_payload(index)
        second = cached_index_payload(equivalent_index)

        self.assertIsNot(first, second)
        self.assertEqual(first.doc_ids, second.doc_ids)

    def test_eviction_discards_oldest_payload(self) -> None:
        indexes = [
            _build_index(seed)
            for seed in range(MOJO_INDEX_PAYLOAD_CACHE_SIZE + 1)
        ]

        payloads = [cached_index_payload(index) for index in indexes]

        cached_again = cached_index_payload(indexes[0])

        self.assertIsNot(payloads[0], cached_again)

    def test_batch_dispatch_reuses_cached_hybrid_payload(self) -> None:
        index = _build_index(10)
        query_batch = kayak.query_batch(
            [
                np.ones((2, 128), dtype=np.float32),
                np.ones((3, 128), dtype=np.float32),
            ]
        ).to_layout("flat_dim128")

        with patch("kayak_bridge.mojo_payload_cache.index_payload") as wrapped:
            from kayak_bridge.mojo_payloads import index_payload

            wrapped.side_effect = index_payload
            cached_index_payload(index)
            with patch(
                "kayak_bridge.batch_dispatch.load_mojo_exact_cpu_module",
                return_value=_FakeMojoModule(),
            ):
                kayak.maxsim_batch(
                    query_batch,
                    index,
                    backend=kayak.MOJO_EXACT_CPU_BACKEND,
                )

        self.assertEqual(wrapped.call_count, 1)


if __name__ == "__main__":
    unittest.main()
