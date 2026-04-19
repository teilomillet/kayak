from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    DEFAULT_GEM_GRAPH_QUERY_BEAM_WIDTH,
    DEFAULT_GEM_GRAPH_QUERY_CLUSTER_TOP_K,
    GEM_GRAPH_ADAPTIVE_LABEL_POLICY_FIRST_RELEVANT_CLUSTER_RANK,
    GEM_GRAPH_ADAPTIVE_LABEL_POLICY_RELEVANT_CLUSTER_COVERAGE,
    GemGraphBuildConfig,
)
from kayak.benchmarks import (
    GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF,
    GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_SHORTCUTS,
    GEM_HELDOUT_VARIANT_BASELINE,
    GEM_HELDOUT_VARIANT_SHORTCUTS,
    build_gem_heldout_adaptive_diagnostics,
    build_gem_heldout_ablation_summary,
    build_gem_heldout_query_split,
    default_gem_heldout_training_query_count,
    gem_heldout_candidate_window_sizes,
    gem_heldout_ablation_summary_json,
    smoke_synthetic_hard_recall_profile,
    standard_gem_heldout_ablation_variant_specs,
)
from kayak.benchmarks.synthetic_hard_recall_fixture import (
    make_synthetic_hard_recall_fixture,
)
from kayak.runtime import ExactCpuBackend


def unique_collection_root(prefix: String) -> Path:
    var suffix = 0

    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1

def test_gem_heldout_query_split_is_disjoint_and_evenly_spaced() raises:
    var fixture = make_synthetic_hard_recall_fixture(
        smoke_synthetic_hard_recall_profile()
    )
    var split = build_gem_heldout_query_split(fixture.stored_task, 2)

    assert_equal(split.training_selection_policy, "evenly_spaced_queries")
    assert_equal(split.positive_selection_policy, "first_relevant_doc_id")
    assert_equal(len(split.training_pairs), 2)
    assert_equal(len(split.evaluation_task.task.queries), 2)
    assert_equal(split.training_query_ids[0], "query-0")
    assert_equal(split.training_query_ids[1], "query-2")
    assert_equal(split.evaluation_query_ids[0], "query-1")
    assert_equal(split.evaluation_query_ids[1], "query-3")
    assert_equal(split.evaluation_task.task.slice_name.find("__gem_heldout_eval") != -1, True)
    assert_equal(split.training_pairs[0].positive_doc_id.find("doc-c") == 0, True)


def test_standard_gem_heldout_variant_specs_enable_supervision_only_when_needed() raises:
    var fixture = make_synthetic_hard_recall_fixture(
        smoke_synthetic_hard_recall_profile()
    )
    var split = build_gem_heldout_query_split(
        fixture.stored_task,
        default_gem_heldout_training_query_count(fixture.stored_task),
    )
    var variants = standard_gem_heldout_ablation_variant_specs(
        GemGraphBuildConfig(6, 4, 2),
        split,
    )

    assert_equal(len(variants), 4)
    assert_equal(variants[0].variant_kind, GEM_HELDOUT_VARIANT_BASELINE)
    assert_equal(len(variants[0].build_config.training_pairs), 0)
    assert_equal(
        variants[1].variant_kind,
        GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF,
    )
    assert_equal(variants[1].build_config.enable_adaptive_cluster_cutoff, True)
    assert_equal(variants[1].build_config.adaptive_cluster_cutoff_max, 4)
    assert_equal(len(variants[1].build_config.training_pairs) > 0, True)
    assert_equal(variants[2].variant_kind, GEM_HELDOUT_VARIANT_SHORTCUTS)
    assert_equal(variants[2].build_config.enable_shortcuts, True)
    assert_equal(
        variants[3].variant_kind,
        GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_SHORTCUTS,
    )
    assert_equal(variants[3].build_config.enable_adaptive_cluster_cutoff, True)
    assert_equal(variants[3].build_config.enable_shortcuts, True)


def test_gem_heldout_candidate_window_sizes_support_smoke_and_full_modes() raises:
    var full_sizes = gem_heldout_candidate_window_sizes(2, 32, True, 0)
    var smoke_sizes = gem_heldout_candidate_window_sizes(2, 32, False, 0)
    var limited_smoke_sizes = gem_heldout_candidate_window_sizes(2, 32, False, 2)

    assert_equal(full_sizes[0], 2)
    assert_equal(full_sizes[len(full_sizes) - 1], 32)
    assert_equal(smoke_sizes[0], 2)
    assert_equal(smoke_sizes[len(smoke_sizes) - 1], 32)
    assert_equal(len(limited_smoke_sizes), 2)
    assert_equal(limited_smoke_sizes[0], 2)


def test_smoke_adaptive_diagnostics_capture_label_and_limit_collapse() raises:
    var fixture = make_synthetic_hard_recall_fixture(
        smoke_synthetic_hard_recall_profile()
    )
    var split = build_gem_heldout_query_split(fixture.stored_task, 2)
    var variants = standard_gem_heldout_ablation_variant_specs(
        GemGraphBuildConfig(6, 4, 2),
        split,
    )
    var diagnostics = build_gem_heldout_adaptive_diagnostics(
        fixture.stored_index.index,
        variants[1].build_config,
    )

    assert_equal(diagnostics.training_label_count, 2)
    assert_equal(diagnostics.training_label_mean, 1.0)
    assert_equal(diagnostics.training_label_min, 1)
    assert_equal(diagnostics.training_label_max, 1)
    assert_equal(diagnostics.training_label_one_share, 1.0)
    assert_equal(
        diagnostics.predicted_profile_limit_count,
        fixture.stored_index.index.document_count,
    )
    assert_equal(diagnostics.predicted_profile_limit_mean, 1.0)
    assert_equal(diagnostics.predicted_profile_limit_min, 1)
    assert_equal(diagnostics.predicted_profile_limit_max, 1)
    assert_equal(diagnostics.predicted_profile_limit_one_share, 1.0)


def test_smoke_adaptive_diagnostics_capture_coverage_policy_widening() raises:
    var fixture = make_synthetic_hard_recall_fixture(
        smoke_synthetic_hard_recall_profile()
    )
    var split = build_gem_heldout_query_split(fixture.stored_task, 2)
    var diagnostics = build_gem_heldout_adaptive_diagnostics(
        fixture.stored_index.index,
        GemGraphBuildConfig(
            6,
            4,
            2,
            6,
            8,
            True,
            4,
            3,
            2,
            False,
            6,
            32,
            split.training_pairs.copy(),
            GEM_GRAPH_ADAPTIVE_LABEL_POLICY_RELEVANT_CLUSTER_COVERAGE,
        ),
    )

    assert_equal(diagnostics.training_label_count, 2)
    assert_equal(diagnostics.training_label_mean > 1.0, True)
    assert_equal(diagnostics.training_label_one_share, 0.0)
    assert_equal(diagnostics.predicted_profile_limit_count > 0, True)
    assert_equal(diagnostics.predicted_profile_limit_mean > 1.0, True)
    assert_equal(diagnostics.predicted_profile_limit_one_share < 1.0, True)


def test_gem_heldout_ablation_summary_json_contains_variant_and_graph_fields() raises:
    var fixture = make_synthetic_hard_recall_fixture(
        smoke_synthetic_hard_recall_profile()
    )
    var split = build_gem_heldout_query_split(fixture.stored_task, 2)
    var variants = standard_gem_heldout_ablation_variant_specs(
        GemGraphBuildConfig(6, 4, 2),
        split,
    )
    var baseline_summary = build_gem_heldout_ablation_summary(
        ExactCpuBackend(),
        fixture.stored_index,
        split,
        unique_collection_root("kayak-gem-heldout-ablation-baseline"),
        "gem-heldout-ablation-baseline",
        variants[0].copy(),
        4,
    )
    var summary = build_gem_heldout_ablation_summary(
        ExactCpuBackend(),
        fixture.stored_index,
        split,
        unique_collection_root("kayak-gem-heldout-ablation"),
        "gem-heldout-ablation",
        variants[3].copy(),
        4,
    )
    var json = gem_heldout_ablation_summary_json(summary)

    assert_equal(summary.variant_kind, GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_SHORTCUTS)
    assert_equal(summary.training_query_count, 2)
    assert_equal(summary.evaluation_query_count, 2)
    assert_equal(
        summary.search_cluster_top_k_per_query_token,
        DEFAULT_GEM_GRAPH_QUERY_CLUSTER_TOP_K,
    )
    assert_equal(summary.search_beam_width, DEFAULT_GEM_GRAPH_QUERY_BEAM_WIDTH)
    assert_equal(summary.adaptive_cluster_cutoff_enabled, True)
    assert_equal(
        summary.adaptive_label_policy,
        GEM_GRAPH_ADAPTIVE_LABEL_POLICY_FIRST_RELEVANT_CLUSTER_RANK,
    )
    assert_equal(summary.shortcuts_enabled, True)
    assert_equal(summary.artifact_cluster_count > 0, True)
    assert_equal(summary.artifact_graph_edge_count > 0, True)
    assert_equal(summary.stage1_byte_size > 0, True)
    assert_equal(summary.mean_document_profile_limit > 0.0, True)
    assert_equal(summary.max_document_profile_limit >= summary.min_document_profile_limit, True)
    assert_equal(summary.mean_training_positive_profile_limit > 0.0, True)
    assert_equal(summary.mean_evaluation_positive_profile_limit > 0.0, True)
    assert_equal(summary.training_positive_profile_hit_rate >= 0.0, True)
    assert_equal(summary.evaluation_positive_profile_hit_rate >= 0.0, True)
    assert_equal(summary.evaluation_positive_entry_rate >= 0.0, True)
    assert_equal(summary.evaluation_positive_reachable_rate >= 0.0, True)
    assert_equal(
        summary.evaluation_positive_entry_rate
            <= summary.evaluation_positive_reachable_rate,
        True,
    )
    assert_equal(baseline_summary.adaptive_training_label_count, 0)
    assert_equal(summary.adaptive_training_label_count, 2)
    assert_equal(summary.adaptive_training_label_one_share, 1.0)
    assert_equal(summary.adaptive_predicted_profile_limit_count > 0, True)
    assert_equal(summary.adaptive_predicted_profile_limit_one_share, 1.0)
    assert_equal(summary.mean_document_profile_limit < baseline_summary.mean_document_profile_limit, True)
    assert_equal(
        summary.evaluation_positive_profile_hit_rate
            <= baseline_summary.evaluation_positive_profile_hit_rate,
        True,
    )
    assert_equal(
        json.find("\"variant_kind\":\"adaptive_cutoff_shortcuts\"") != -1,
        True,
    )
    assert_equal(json.find("\"training_query_count\":2") != -1, True)
    assert_equal(json.find("\"shortcuts_enabled\":true") != -1, True)
    assert_equal(
        json.find("\"adaptive_label_policy\":\"first_relevant_cluster_rank\"") != -1,
        True,
    )
    assert_equal(json.find("\"artifact_graph_edge_count\":") != -1, True)
    assert_equal(json.find("\"mean_document_profile_limit\":") != -1, True)
    assert_equal(json.find("\"mean_training_positive_profile_limit\":") != -1, True)
    assert_equal(json.find("\"evaluation_positive_profile_hit_rate\":") != -1, True)
    assert_equal(json.find("\"evaluation_positive_reachable_rate\":") != -1, True)
    assert_equal(json.find("\"adaptive_training_label_count\":2") != -1, True)
    assert_equal(json.find("\"adaptive_predicted_profile_limit_one_share\":1.0") != -1, True)
    assert_equal(json.find("\"mean_candidate_recall_at_final_k\":") != -1, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
