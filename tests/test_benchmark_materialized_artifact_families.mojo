from std.testing import TestSuite, assert_equal

from kayak import StageArtifactMaterialization
from kayak.benchmarks.materialized_artifact_families import (
    materialized_artifact_families,
)


def test_materialized_artifact_families_deduplicates_in_first_seen_order() raises:
    var families = materialized_artifact_families(
        [
            StageArtifactMaterialization("document_text", 1, 2, 7, 0, 48),
            StageArtifactMaterialization("late_interaction", 1, 2, 7, 7, 224),
            StageArtifactMaterialization("document_text", 1, 1, 3, 0, 24),
        ]
    )

    assert_equal(len(families), 2)
    assert_equal(families[0], "document_text")
    assert_equal(families[1], "late_interaction")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
