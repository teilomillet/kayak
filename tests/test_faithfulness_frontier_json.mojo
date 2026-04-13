from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionId,
    NamespaceId,
    SnapshotId,
    TenantId,
)
from kayak.benchmarks import (
    FaithfulnessFrontierSummary,
    build_faithfulness_frontier_summary_for_plan,
    default_single_core_scale_profiles,
    faithfulness_frontier_summary_json,
    make_single_core_scale_fixture,
)
from kayak.collections import (
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
)
from kayak.planning import (
    best_effort_faithfulness_policy,
    centroid_heads_search_plan,
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


def test_faithfulness_frontier_summary_json_contains_budget_fields() raises:
    var json = faithfulness_frontier_summary_json(
        FaithfulnessFrontierSummary(
            "mock://dataset",
            "mock-model",
            "synthetic_scale",
            "docs_64",
            "mock_collection",
            "snapshot-0001",
            centroid_heads_search_plan(
                2,
                16,
                best_effort_faithfulness_policy(),
            ).candidate_generator,
            2,
            16,
            4,
            64,
            8,
            8,
            64,
            512,
            512,
            64,
            0.001,
            0.002,
            16.0,
            0.75,
            0.5,
            0.5,
            0.5,
            1.0,
            64,
            128,
            4096,
            64.0,
            64.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    )

    assert_equal(
        json.find("\"requested_stage1_vector_budget\":64") != -1,
        True,
    )
    assert_equal(json.find("\"posting_cap\":8") != -1, True)
    assert_equal(
        json.find("\"stage1_bytes_per_vector\":64.0") != -1,
        True,
    )
    assert_equal(
        json.find("\"candidate_generator_family\":\"centroid\"") != -1,
        True,
    )
    assert_equal(
        json.find("\"stage1_interaction_semantics\":\"approximate_late_interaction\"")
            != -1,
        True,
    )
    assert_equal(
        json.find("\"mean_stage1_graph_visited_vertex_count\":0.0") != -1,
        True,
    )


def test_build_faithfulness_frontier_summary_reports_stage1_shape() raises:
    var profile = default_single_core_scale_profiles()[0].copy()
    var fixture = make_single_core_scale_fixture(profile)
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-faithfulness-frontier-summary"),
        CollectionId("faithfulness-frontier-summary"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        fixture.stored_index,
        profile.document_vector_count,
        profile.vector_dim,
        8,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )
    var summary = build_faithfulness_frontier_summary_for_plan(
        single_core_backend(),
        fixture.stored_task,
        snapshot,
        centroid_heads_search_plan(
            profile.final_k,
            profile.candidate_k,
            best_effort_faithfulness_policy(),
        ),
        profile.query_vector_count,
        profile.vector_dim,
        8,
    )

    assert_equal(summary.family, "synthetic_scale")
    assert_equal(summary.slice_name, profile.slice_name)
    assert_equal(summary.query_vector_budget, profile.query_vector_count)
    assert_equal(summary.requested_stage1_vector_budget, profile.vector_dim)
    assert_equal(summary.posting_cap, 8)
    assert_equal(summary.document_count, profile.document_count)
    assert_equal(summary.stage1_vector_count > 0, True)
    assert_equal(summary.stage1_byte_size > 0, True)
    assert_equal(summary.mean_candidate_hit_count >= Float64(profile.final_k), True)
    assert_equal(summary.mean_stage1_graph_visited_vertex_count, 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
