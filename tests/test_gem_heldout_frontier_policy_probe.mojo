from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak.benchmarks.gem_heldout_ablation_json import (
    gem_heldout_ablation_variant_spec,
)
from kayak.benchmarks.gem_heldout_frontier_policy_probe import (
    GEM_HELDOUT_FRONTIER_POLICY_GLOBAL_BEST_FIRST,
    GEM_HELDOUT_FRONTIER_POLICY_HYBRID_BEST_HEAD_FAIR_ROUND,
    GEM_HELDOUT_FRONTIER_POLICY_HYBRID_BEST_HEAD_PER_ENTRY,
    GEM_HELDOUT_FRONTIER_POLICY_HYBRID_BEST_HEAD_QUOTA2_ROUND,
    GEM_HELDOUT_FRONTIER_POLICY_LOCAL_PER_ENTRY,
    build_gem_heldout_frontier_policy_probe_summary,
    gem_heldout_frontier_policy_probe_frontier_policies,
    gem_heldout_frontier_policy_probe_summaries_json,
)
from kayak.benchmarks.gem_heldout_querytime_probe import (
    materialized_gem_heldout_querytime_probe_collection_root,
)
from kayak.benchmarks.synthetic_hard_recall_fixture import (
    make_synthetic_hard_recall_fixture,
)
from kayak.benchmarks import (
    build_gem_heldout_query_split,
    default_gem_heldout_training_query_count,
    frontier_gem_graph_build_config,
    smoke_synthetic_hard_recall_profile,
)
from kayak.collections import (
    SnapshotId,
    load_resolved_collection_snapshot,
    loaded_segment_stored_gem_graph_index,
)
from kayak.runtime import ExactCpuBackend
from kayak.scoring import ExactScoringConfig


def unique_collection_root(prefix: String) -> Path:
    var suffix = 0
    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1


def single_core_backend() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_parallel_scoring = False
    return ExactCpuBackend(config^)


def test_frontier_policy_probe_lists_hybrid_policy() raises:
    var policies = gem_heldout_frontier_policy_probe_frontier_policies()
    assert_equal(len(policies), 5)
    assert_equal(policies[0], GEM_HELDOUT_FRONTIER_POLICY_LOCAL_PER_ENTRY)
    assert_equal(policies[1], GEM_HELDOUT_FRONTIER_POLICY_GLOBAL_BEST_FIRST)
    assert_equal(
        policies[2],
        GEM_HELDOUT_FRONTIER_POLICY_HYBRID_BEST_HEAD_PER_ENTRY,
    )
    assert_equal(
        policies[3],
        GEM_HELDOUT_FRONTIER_POLICY_HYBRID_BEST_HEAD_FAIR_ROUND,
    )
    assert_equal(
        policies[4],
        GEM_HELDOUT_FRONTIER_POLICY_HYBRID_BEST_HEAD_QUOTA2_ROUND,
    )


def test_smoke_frontier_policy_probe_summary_supports_hybrid_policies() raises:
    var fixture = make_synthetic_hard_recall_fixture(
        smoke_synthetic_hard_recall_profile()
    )
    var split = build_gem_heldout_query_split(
        fixture.stored_task,
        default_gem_heldout_training_query_count(fixture.stored_task),
    )
    var variant = gem_heldout_ablation_variant_spec(
        "baseline",
        frontier_gem_graph_build_config(
            fixture.stored_index,
            fixture.stored_task.task.nominal_query_vector_count,
        ),
        split.training_pairs,
    )
    var snapshot_id = SnapshotId("snapshot-0001")
    var collection_root = unique_collection_root(
        "kayak-gem-heldout-frontier-policy-probe"
    )
    var collection_path = materialized_gem_heldout_querytime_probe_collection_root(
        collection_root,
        "gem-heldout-frontier-policy-probe",
        fixture.stored_index,
        variant,
        snapshot_id,
    )
    var snapshot = load_resolved_collection_snapshot(collection_path, snapshot_id)
    var stored_gem = loaded_segment_stored_gem_graph_index(snapshot.segments[0])
    var per_entry_summary = build_gem_heldout_frontier_policy_probe_summary(
        single_core_backend(),
        split,
        snapshot,
        stored_gem,
        variant.variant_kind,
        GEM_HELDOUT_FRONTIER_POLICY_HYBRID_BEST_HEAD_PER_ENTRY,
        1,
        variant.build_config.cluster_top_k_per_query_token,
        32,
    )
    var fair_round_summary = build_gem_heldout_frontier_policy_probe_summary(
        single_core_backend(),
        split,
        snapshot,
        stored_gem,
        variant.variant_kind,
        GEM_HELDOUT_FRONTIER_POLICY_HYBRID_BEST_HEAD_FAIR_ROUND,
        1,
        variant.build_config.cluster_top_k_per_query_token,
        32,
    )
    var quota2_round_summary = build_gem_heldout_frontier_policy_probe_summary(
        single_core_backend(),
        split,
        snapshot,
        stored_gem,
        variant.variant_kind,
        GEM_HELDOUT_FRONTIER_POLICY_HYBRID_BEST_HEAD_QUOTA2_ROUND,
        1,
        variant.build_config.cluster_top_k_per_query_token,
        32,
    )

    for summary in [
        per_entry_summary.copy(),
        fair_round_summary.copy(),
        quota2_round_summary.copy(),
    ]:
        assert_equal(
            summary.frontier_policy
                == GEM_HELDOUT_FRONTIER_POLICY_HYBRID_BEST_HEAD_PER_ENTRY
                or summary.frontier_policy
                    == GEM_HELDOUT_FRONTIER_POLICY_HYBRID_BEST_HEAD_FAIR_ROUND
                or summary.frontier_policy
                    == GEM_HELDOUT_FRONTIER_POLICY_HYBRID_BEST_HEAD_QUOTA2_ROUND,
            True,
        )
        assert_equal(summary.candidate_k, 1)
        assert_equal(summary.search_cluster_top_k_per_query_token, 2)
        assert_equal(summary.search_beam_width, 32)
        assert_equal(summary.mean_candidate_recall_at_final_k >= 0.0, True)
        assert_equal(summary.mean_candidate_recall_at_final_k <= 1.0, True)
        assert_equal(summary.mean_recall_at_k >= 0.0, True)
        assert_equal(summary.mean_recall_at_k <= 1.0, True)
        assert_equal(summary.success_rate_at_k >= 0.0, True)
        assert_equal(summary.success_rate_at_k <= 1.0, True)
        assert_equal(summary.mean_stage1_graph_visited_vertex_count >= 0.0, True)
        assert_equal(summary.mean_stage1_graph_expanded_edge_count >= 0.0, True)
        assert_equal(summary.mean_stage1_graph_max_frontier_size >= 0.0, True)

    var json = gem_heldout_frontier_policy_probe_summaries_json(
        [
            per_entry_summary.copy(),
            fair_round_summary.copy(),
            quota2_round_summary.copy(),
        ]
    )
    assert_equal(
        json.find("\"frontier_policy\":\"hybrid_best_head_per_entry\"") != -1,
        True,
    )
    assert_equal(
        json.find("\"frontier_policy\":\"hybrid_best_head_fair_round\"") != -1,
        True,
    )
    assert_equal(
        json.find("\"frontier_policy\":\"hybrid_best_head_quota2_round\"") != -1,
        True,
    )
    assert_equal(json.find("\"mean_candidate_recall_at_final_k\":") != -1, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
