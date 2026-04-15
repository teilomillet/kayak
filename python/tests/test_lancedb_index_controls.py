from __future__ import annotations

import unittest

from kayak_bridge.lancedb_index_controls import (
    LanceDbIndexBuildControls,
    LanceDbIndexedQueryControls,
    validate_index_controls,
)


class LanceDbIndexControlsTests(unittest.TestCase):
    def test_rejects_indexed_query_controls_for_scan_runs(self) -> None:
        with self.assertRaisesRegex(ValueError, "build_index=True"):
            validate_index_controls(
                build_index=False,
                index_build_controls=None,
                indexed_query_controls=LanceDbIndexedQueryControls(refine_factor=2),
            )

    def test_rejects_non_positive_override_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "indexed_refine_factor"):
            LanceDbIndexedQueryControls(refine_factor=0).validated()

        with self.assertRaisesRegex(ValueError, "index_num_partitions"):
            LanceDbIndexBuildControls(num_partitions=-1).validated()

    def test_build_controls_expose_only_explicit_overrides(self) -> None:
        controls = LanceDbIndexBuildControls(
            num_partitions=8,
            num_sub_vectors=None,
            target_partition_size=512,
        ).validated()

        self.assertEqual(
            controls.create_index_kwargs(),
            {"num_partitions": 8, "target_partition_size": 512},
        )


if __name__ == "__main__":
    unittest.main()
