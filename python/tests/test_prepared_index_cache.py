from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import kayak
import kayak_bridge.prepared_index_storage_artifact as prepared_storage_artifact
from kayak_bridge.prepared_index_cache import (
    PREPARED_PACKED_INDEX_CACHE_SIZE,
    clear_prepared_packed_index_cache,
    prepared_packed_index_object,
)


class _FakePreparedModule:
    def __init__(self) -> None:
        self.prepared_calls = 0
        self.prepared_objects: list[object] = []

    def prepare_packed_index(
        self,
        doc_ids: list[str],
        doc_offsets: list[int],
        packed_vectors: list[list[float]],
    ) -> object:
        del doc_ids, doc_offsets, packed_vectors
        self.prepared_calls += 1
        prepared = object()
        self.prepared_objects.append(prepared)
        return prepared


class _FakePreparedStorageModule:
    def __init__(self) -> None:
        self.prepared_from_storage_calls = 0
        self.prepared_calls = 0
        self.prepared_objects: list[object] = []
        self.storage_roots: list[Path] = []

    def prepare_packed_index_from_storage(
        self, root: str, vector_dim: int
    ) -> object:
        self.prepared_from_storage_calls += 1
        self.storage_roots.append(Path(root))
        self.vector_dim = vector_dim
        prepared = object()
        self.prepared_objects.append(prepared)
        return prepared

    def prepare_packed_index(
        self,
        doc_ids: list[str],
        doc_offsets: list[int],
        packed_vectors: list[list[float]],
    ) -> object:
        del doc_ids, doc_offsets, packed_vectors
        self.prepared_calls += 1
        raise AssertionError("direct list payload path should not be used")


def _build_index(seed: int) -> kayak.LateIndex:
    rng = np.random.default_rng(seed)
    return kayak.documents(
        [f"doc-{seed}-a", f"doc-{seed}-b"],
        [
            rng.standard_normal((2, 128)).astype(np.float32),
            rng.standard_normal((3, 128)).astype(np.float32),
        ],
    ).pack()


class PreparedIndexCacheTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_prepared_packed_index_cache()

    def test_reuses_prepared_object_for_same_index_object(self) -> None:
        module = _FakePreparedModule()
        index = _build_index(1)

        first = prepared_packed_index_object(index, module=module)
        second = prepared_packed_index_object(index, module=module)

        self.assertIs(first, second)
        self.assertEqual(module.prepared_calls, 1)

    def test_cache_is_scoped_to_module_identity(self) -> None:
        module = _FakePreparedModule()
        other_module = _FakePreparedModule()
        index = _build_index(2)

        first = prepared_packed_index_object(index, module=module)
        second = prepared_packed_index_object(index, module=other_module)

        self.assertIsNot(first, second)
        self.assertEqual(module.prepared_calls, 1)
        self.assertEqual(other_module.prepared_calls, 1)

    def test_eviction_discards_oldest_prepared_object(self) -> None:
        module = _FakePreparedModule()
        indexes = [
            _build_index(seed)
            for seed in range(PREPARED_PACKED_INDEX_CACHE_SIZE + 1)
        ]

        prepared_objects = [
            prepared_packed_index_object(index, module=module)
            for index in indexes
        ]

        cached_again = prepared_packed_index_object(indexes[0], module=module)

        self.assertEqual(module.prepared_calls, len(indexes) + 1)
        self.assertIsNot(prepared_objects[0], cached_again)

    def test_storage_preparation_writes_native_artifact(self) -> None:
        module = _FakePreparedStorageModule()
        index = _build_index(11)

        prepared = prepared_packed_index_object(index, module=module)

        self.assertIs(prepared, module.prepared_objects[0])
        self.assertEqual(module.prepared_from_storage_calls, 1)
        self.assertEqual(module.prepared_calls, 0)
        self.assertEqual(module.vector_dim, 128)
        root = module.storage_roots[0]
        self.assertTrue((root / "doc_ids.tsv").exists())
        doc_offsets_path = root / "doc_offsets.bin"
        token_values_path = root / "token_values.bin"
        self.assertTrue(doc_offsets_path.exists())
        self.assertTrue(token_values_path.exists())
        self.assertEqual(
            doc_offsets_path.stat().st_size,
            (index.document_count + 1) * np.dtype(np.int64).itemsize,
        )
        expected_size = (
            index.total_vector_count * index.vector_dim * np.dtype(np.float32).itemsize
        )
        self.assertEqual(token_values_path.stat().st_size, expected_size)

        clear_prepared_packed_index_cache()
        self.assertFalse(root.exists())

    def test_storage_preparation_splits_large_token_value_payload(self) -> None:
        module = _FakePreparedStorageModule()
        index = _build_index(13)

        with patch.object(
            prepared_storage_artifact,
            "MAX_PREPARED_TOKEN_VALUES_PART_BYTES",
            512,
        ):
            prepared = prepared_packed_index_object(index, module=module)

        self.assertIs(prepared, module.prepared_objects[0])
        root = module.storage_roots[0]
        parts_manifest = root / "token_values.parts.tsv"
        self.assertTrue(parts_manifest.exists())
        part_names = parts_manifest.read_text(encoding="utf-8").splitlines()
        self.assertGreater(len(part_names), 1)
        self.assertFalse((root / "token_values.bin").exists())
        total_size = sum((root / part_name).stat().st_size for part_name in part_names)
        expected_size = (
            index.total_vector_count * index.vector_dim * np.dtype(np.float32).itemsize
        )
        self.assertEqual(total_size, expected_size)


if __name__ == "__main__":
    unittest.main()
