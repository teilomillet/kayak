from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionId,
    NamespaceId,
    SnapshotId,
    TenantId,
    centroid_posting_imputed_flat_scores_for_segment,
    centroid_posting_imputed_scores_for_segment,
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
    loaded_segment_stored_centroid_postings_index,
)
from kayak.benchmarks import (
    high_centroid_synthetic_hard_recall_profile,
    make_synthetic_hard_recall_fixture,
)


def unique_collection_root(prefix: String) -> Path:
    var suffix = 0

    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1


def test_high_centroid_imputed_probe_exceeds_current_imputed_bound() raises:
    var profile = high_centroid_synthetic_hard_recall_profile()
    var fixture = make_synthetic_hard_recall_fixture(profile)
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-high-centroid-imputed-probe"),
        CollectionId("high-centroid-imputed-probe"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        fixture.stored_index,
        0,
        fixture.stored_index.index.vector_dim,
        profile.centroid_head_posting_cap,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )
    var index = loaded_segment_stored_centroid_postings_index(
        snapshot.segments[0]
    ).index.copy()

    assert_equal(index.centroid_count > 128, True)


def test_high_centroid_imputed_flat_stage_matches_imputed_stage_scores() raises:
    var profile = high_centroid_synthetic_hard_recall_profile()
    var fixture = make_synthetic_hard_recall_fixture(profile)
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-high-centroid-imputed-equality"),
        CollectionId("high-centroid-imputed-equality"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        fixture.stored_index,
        0,
        fixture.stored_index.index.vector_dim,
        profile.centroid_head_posting_cap,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )
    var index = loaded_segment_stored_centroid_postings_index(
        snapshot.segments[0]
    ).index.copy()
    var query = fixture.stored_task.task.queries[0].query.copy()
    var imputed_scores = centroid_posting_imputed_scores_for_segment(
        query.token_vectors,
        index,
        profile.final_k,
    )
    var imputed_flat_scores = centroid_posting_imputed_flat_scores_for_segment(
        query,
        index,
        profile.final_k,
    )

    assert_equal(len(imputed_flat_scores), len(imputed_scores))
    for score_index in range(len(imputed_scores)):
        assert_equal(imputed_flat_scores[score_index], imputed_scores[score_index])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
