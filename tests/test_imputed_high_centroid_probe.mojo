from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    ScoreScalar,
    VectorScalar,
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
from kayak.index import CentroidPostingIndex
from kayak.scoring.dot import dot_product
from kayak.planning.centroid_primitives import ScoredCentroidSelection
from kayak.planning.centroid_postings_imputed_stage import (
    centroid_selection_for_query_token,
    centroid_token_count,
    effective_imputed_centroid_bound,
    effective_imputed_centroid_nprobe,
    warp_like_t_prime,
)
from kayak.planning.centroid_postings_imputed_flat_stage import (
    centroid_selection_for_flat_query_token_generic,
)


def reference_insert_descending_centroid_match(
    mut centroid_indices: List[Int],
    mut centroid_scores: List[ScoreScalar],
    centroid_index: Int,
    centroid_score: ScoreScalar,
    k: Int,
):
    if k <= 0:
        return

    var insert_at = 0
    while (
        insert_at < len(centroid_scores)
        and centroid_scores[insert_at] >= centroid_score
    ):
        insert_at += 1

    if insert_at >= k:
        if len(centroid_scores) < k:
            centroid_indices.append(centroid_index)
            centroid_scores.append(centroid_score)
        return

    if len(centroid_scores) < k:
        centroid_indices.append(centroid_index)
        centroid_scores.append(centroid_score)

    var current = len(centroid_scores) - 1
    while current > insert_at:
        centroid_indices[current] = centroid_indices[current - 1]
        centroid_scores[current] = centroid_scores[current - 1]
        current -= 1

    centroid_indices[insert_at] = centroid_index
    centroid_scores[insert_at] = centroid_score


def reference_centroid_selection_for_query_token(
    read query_token: List[VectorScalar],
    read index: CentroidPostingIndex,
    final_k: Int,
) -> ScoredCentroidSelection:
    var bound = effective_imputed_centroid_bound(index.centroid_count)
    var sorted_centroid_indices = List[Int]()
    var sorted_centroid_scores = List[ScoreScalar]()

    for centroid_index in range(index.centroid_count):
        reference_insert_descending_centroid_match(
            sorted_centroid_indices,
            sorted_centroid_scores,
            centroid_index,
            dot_product(query_token, index.centroid_vectors[centroid_index]),
            bound,
        )

    var nprobe = effective_imputed_centroid_nprobe(bound)
    var selected_centroid_indices = List[Int]()
    var selected_centroid_scores = List[ScoreScalar]()

    for selection_index in range(nprobe):
        selected_centroid_indices.append(sorted_centroid_indices[selection_index])
        selected_centroid_scores.append(sorted_centroid_scores[selection_index])

    var t_prime = warp_like_t_prime(index, final_k)
    var cumulative_size = 0
    var missing_similarity_estimate = ScoreScalar(0.0)

    for sorted_index in range(bound):
        var centroid_index = sorted_centroid_indices[sorted_index]
        cumulative_size += centroid_token_count(index, centroid_index)
        missing_similarity_estimate = sorted_centroid_scores[sorted_index]
        if cumulative_size >= t_prime:
            break

    return ScoredCentroidSelection(
        selected_centroid_indices^,
        selected_centroid_scores^,
        missing_similarity_estimate,
    )


def assert_scored_centroid_selection_equal(
    read lhs: ScoredCentroidSelection,
    read rhs: ScoredCentroidSelection,
) raises:
    assert_equal(lhs.baseline_correction, rhs.baseline_correction)
    assert_equal(lhs.centroid_indices, rhs.centroid_indices)
    assert_equal(lhs.centroid_scores, rhs.centroid_scores)


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


def test_high_centroid_imputed_selector_matches_reference() raises:
    var profile = high_centroid_synthetic_hard_recall_profile()
    var fixture = make_synthetic_hard_recall_fixture(profile)
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-high-centroid-imputed-selection-reference"),
        CollectionId("high-centroid-imputed-selection-reference"),
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
    var query_token = fixture.stored_task.task.queries[0].query.token_vectors[0].copy()

    assert_scored_centroid_selection_equal(
        centroid_selection_for_query_token(query_token, index, profile.final_k),
        reference_centroid_selection_for_query_token(
            query_token,
            index,
            profile.final_k,
        ),
    )


def test_high_centroid_imputed_flat_selector_matches_nested_selector() raises:
    var profile = high_centroid_synthetic_hard_recall_profile()
    var fixture = make_synthetic_hard_recall_fixture(profile)
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-high-centroid-imputed-flat-selection"),
        CollectionId("high-centroid-imputed-flat-selection"),
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
    var flat_query_values = List[VectorScalar]()
    for token_vector in query.token_vectors:
        for value in token_vector:
            flat_query_values.append(value)

    assert_scored_centroid_selection_equal(
        centroid_selection_for_flat_query_token_generic(
            flat_query_values,
            0,
            index,
            profile.final_k,
        ),
        centroid_selection_for_query_token(
            query.token_vectors[0],
            index,
            profile.final_k,
        ),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
