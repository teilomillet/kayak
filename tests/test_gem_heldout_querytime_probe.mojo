from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak.benchmarks import (
    build_gem_heldout_query_split,
    build_gem_heldout_querytime_probe_summary,
    default_gem_heldout_training_query_count,
    gem_heldout_querytime_probe_summaries_json,
    frontier_gem_graph_build_config,
    smoke_synthetic_hard_recall_profile,
)
from kayak.benchmarks.gem_heldout_ablation_json import (
    gem_heldout_ablation_variant_spec,
)
from kayak.benchmarks.synthetic_hard_recall_fixture import (
    make_synthetic_hard_recall_fixture,
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


def test_smoke_querytime_probe_summary_reports_reachability_fields() raises:
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
    var summary = build_gem_heldout_querytime_probe_summary(
        single_core_backend(),
        fixture.stored_index,
        split,
        unique_collection_root("kayak-gem-heldout-querytime-probe"),
        "gem-heldout-querytime-probe",
        variant,
        1,
        variant.build_config.cluster_top_k_per_query_token,
        32,
    )
    var json = gem_heldout_querytime_probe_summaries_json([summary.copy()])

    assert_equal(summary.variant_kind, "baseline")
    assert_equal(summary.construction_neighbor_count > 0, True)
    assert_equal(summary.degree_limit > 0, True)
    assert_equal(summary.shortcut_candidate_k > 0, True)
    assert_equal(summary.shortcuts_enabled, False)
    assert_equal(summary.artifact_shortcut_edge_count, 0)
    assert_equal(summary.candidate_k, 1)
    assert_equal(summary.search_cluster_top_k_per_query_token, 2)
    assert_equal(summary.search_beam_width, 32)
    assert_equal(summary.evaluation_positive_entry_rate >= 0.0, True)
    assert_equal(summary.evaluation_positive_reachable_rate >= 0.0, True)
    assert_equal(summary.evaluation_positive_mean_reachable_hop_count >= 0.0, True)
    assert_equal(summary.evaluation_positive_within_1_hop_rate >= 0.0, True)
    assert_equal(
        summary.evaluation_positive_within_1_hop_rate
            <= summary.evaluation_positive_within_2_hop_rate,
        True,
    )
    assert_equal(
        summary.evaluation_positive_within_2_hop_rate
            <= summary.evaluation_positive_within_4_hop_rate,
        True,
    )
    assert_equal(
        summary.evaluation_positive_within_4_hop_rate
            <= summary.evaluation_positive_reachable_rate,
        True,
    )
    assert_equal(
        summary.evaluation_positive_entry_rate
            <= summary.evaluation_positive_reachable_rate,
        True,
    )
    assert_equal(summary.mean_stage1_graph_visited_vertex_count > 0.0, True)
    assert_equal(
        json.find("\"construction_neighbor_count\":") != -1,
        True,
    )
    assert_equal(
        json.find("\"degree_limit\":") != -1,
        True,
    )
    assert_equal(
        json.find("\"shortcut_candidate_k\":") != -1,
        True,
    )
    assert_equal(
        json.find("\"artifact_shortcut_edge_count\":") != -1,
        True,
    )
    assert_equal(
        json.find("\"search_cluster_top_k_per_query_token\":") != -1,
        True,
    )
    assert_equal(json.find("\"evaluation_positive_reachable_rate\":") != -1, True)
    assert_equal(
        json.find("\"evaluation_positive_mean_reachable_hop_count\":") != -1,
        True,
    )
    assert_equal(
        json.find("\"evaluation_positive_within_2_hop_rate\":") != -1,
        True,
    )
    assert_equal(json.find("\"mean_recall_at_k\":") != -1, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
