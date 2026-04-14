from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    FaithfulnessFrontierSummary,
    build_faithfulness_frontier_summary_for_plan,
    default_single_core_scale_profiles,
    frontier_gem_graph_build_config,
    faithfulness_frontier_summaries_json,
    make_single_core_scale_fixture,
    standard_candidate_window_sizes,
    standard_posting_cap_sizes,
)
from kayak.collections import (
    CollectionId,
    NamespaceId,
    SnapshotId,
    TenantId,
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
)
from kayak.planning import (
    best_effort_faithfulness_policy,
    centroid_heads_search_plan,
    centroid_postings_head_auto_search_plan,
    centroid_postings_imputed_search_plan,
    centroid_postings_search_plan,
    document_proxy_search_plan,
    exact_full_scan_search_plan,
    gem_graph_search_plan,
)
from kayak.runtime import ExactCpuBackend
from kayak.scoring import ExactScoringConfig


def single_core_backend() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_parallel_scoring = False
    return ExactCpuBackend(config^)


def print_frontier_mean_line(read summary: FaithfulnessFrontierSummary):
    print(
        "Mean: ",
        summary.mean_search_seconds,
        " family=",
        summary.family,
        " slice=",
        summary.slice_name,
        " generator=",
        summary.candidate_generator.kind,
        " candidate_k=",
        summary.candidate_k,
        " posting_cap=",
        summary.posting_cap,
        " recall=",
        summary.mean_candidate_recall_at_final_k,
    )


def documents_per_query_group(document_count: Int, query_count: Int) -> Int:
    return document_count // query_count


def max_frontier_candidate_k(document_count: Int, query_count: Int, baseline_k: Int) -> Int:
    var max_candidate_k = documents_per_query_group(document_count, query_count)
    if baseline_k > max_candidate_k:
        max_candidate_k = baseline_k
    if max_candidate_k > document_count:
        max_candidate_k = document_count
    return max_candidate_k


def append_summary(
    mut summaries: List[FaithfulnessFrontierSummary],
    read summary: FaithfulnessFrontierSummary,
):
    summaries.append(summary.copy())
    print_frontier_mean_line(summary)


def append_frontier_for_profile(
    mut summaries: List[FaithfulnessFrontierSummary],
    read backend: ExactCpuBackend,
    profile_index: Int,
) raises:
    var profile = default_single_core_scale_profiles()[profile_index].copy()
    var fixture = make_single_core_scale_fixture(profile)
    var tenant_id = TenantId("public")
    var namespace_id = NamespaceId("benchmark")
    var snapshot_id = SnapshotId("snapshot-0001")
    var full_proxy_budget = profile.document_vector_count
    var full_centroid_budget = profile.vector_dim
    var gem_config = frontier_gem_graph_build_config(
        fixture.stored_index,
        profile.query_vector_count,
    )
    var max_candidate_k = max_frontier_candidate_k(
        profile.document_count,
        profile.query_count,
        profile.candidate_k,
    )

    var exact_root = ensure_one_segment_collection_mirror(
        Path(".cache/kayak/single_core_faithfulness_exact_" + profile.slice_name),
        CollectionId("single-core-faithfulness-exact-" + profile.slice_name),
        tenant_id,
        namespace_id,
        snapshot_id,
        1,
        fixture.stored_index,
        full_proxy_budget,
        full_centroid_budget,
        0,
        gem_config.fine_cluster_count,
        gem_config.coarse_cluster_count,
        gem_config.cluster_cutoff,
    )
    var exact_snapshot = load_resolved_collection_snapshot(exact_root, snapshot_id)
    append_summary(
        summaries,
        build_faithfulness_frontier_summary_for_plan(
            backend,
            fixture.stored_task,
            exact_snapshot,
            exact_full_scan_search_plan(profile.final_k, profile.final_k),
            profile.query_vector_count,
            profile.document_vector_count,
            0,
        ),
    )

    var proxy_root = ensure_one_segment_collection_mirror(
        Path(".cache/kayak/single_core_faithfulness_proxy_" + profile.slice_name),
        CollectionId("single-core-faithfulness-proxy-" + profile.slice_name),
        tenant_id,
        namespace_id,
        snapshot_id,
        1,
        fixture.stored_index,
        full_proxy_budget,
        full_centroid_budget,
        0,
        gem_config.fine_cluster_count,
        gem_config.coarse_cluster_count,
        gem_config.cluster_cutoff,
    )
    var proxy_snapshot = load_resolved_collection_snapshot(proxy_root, snapshot_id)

    for candidate_k in standard_candidate_window_sizes(
        profile.final_k,
        max_candidate_k,
    ):
        append_summary(
            summaries,
            build_faithfulness_frontier_summary_for_plan(
                backend,
                fixture.stored_task,
                proxy_snapshot,
                document_proxy_search_plan(
                    profile.final_k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
                profile.query_vector_count,
                full_proxy_budget,
                0,
            ),
        )
    var centroid_root = ensure_one_segment_collection_mirror(
        Path(".cache/kayak/single_core_faithfulness_centroid_" + profile.slice_name),
        CollectionId("single-core-faithfulness-centroid-" + profile.slice_name),
        tenant_id,
        namespace_id,
        snapshot_id,
        1,
        fixture.stored_index,
        full_proxy_budget,
        full_centroid_budget,
        0,
        gem_config.fine_cluster_count,
        gem_config.coarse_cluster_count,
        gem_config.cluster_cutoff,
    )
    var centroid_snapshot = load_resolved_collection_snapshot(
        centroid_root,
        snapshot_id,
    )

    for candidate_k in standard_candidate_window_sizes(
        profile.final_k,
        max_candidate_k,
    ):
        append_summary(
            summaries,
            build_faithfulness_frontier_summary_for_plan(
                backend,
                fixture.stored_task,
                centroid_snapshot,
                centroid_postings_search_plan(
                    profile.final_k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
                profile.query_vector_count,
                full_centroid_budget,
                0,
            ),
        )
        append_summary(
            summaries,
            build_faithfulness_frontier_summary_for_plan(
                backend,
                fixture.stored_task,
                centroid_snapshot,
                centroid_postings_imputed_search_plan(
                    profile.final_k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
                profile.query_vector_count,
                full_centroid_budget,
                0,
            ),
        )
        append_summary(
            summaries,
            build_faithfulness_frontier_summary_for_plan(
                backend,
                fixture.stored_task,
                centroid_snapshot,
                gem_graph_search_plan(
                    profile.final_k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
                profile.query_vector_count,
                full_centroid_budget,
                0,
            ),
        )

    for posting_cap in standard_posting_cap_sizes(
        documents_per_query_group(profile.document_count, profile.query_count)
    ):
        var heads_root = ensure_one_segment_collection_mirror(
            Path(
                ".cache/kayak/single_core_faithfulness_heads_"
                + profile.slice_name
                + "_posting_cap_"
                + String(posting_cap)
            ),
            CollectionId(
                "single-core-faithfulness-heads-"
                + profile.slice_name
                + "-"
                + String(posting_cap)
            ),
            tenant_id,
            namespace_id,
            snapshot_id,
            1,
            fixture.stored_index,
            full_proxy_budget,
            full_centroid_budget,
            posting_cap,
            gem_config.fine_cluster_count,
            gem_config.coarse_cluster_count,
            gem_config.cluster_cutoff,
        )
        var heads_snapshot = load_resolved_collection_snapshot(
            heads_root,
            snapshot_id,
        )

        for candidate_k in standard_candidate_window_sizes(
            profile.final_k,
            max_candidate_k,
        ):
            append_summary(
                summaries,
                build_faithfulness_frontier_summary_for_plan(
                    backend,
                    fixture.stored_task,
                    heads_snapshot,
                    centroid_heads_search_plan(
                        profile.final_k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                    ),
                    profile.query_vector_count,
                    full_centroid_budget,
                    posting_cap,
                ),
            )
            append_summary(
                summaries,
                build_faithfulness_frontier_summary_for_plan(
                    backend,
                    fixture.stored_task,
                    heads_snapshot,
                    centroid_postings_head_auto_search_plan(
                        profile.final_k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                    ),
                    profile.query_vector_count,
                    full_centroid_budget,
                    posting_cap,
                ),
            )


def main() raises:
    var backend = single_core_backend()
    var summaries = List[FaithfulnessFrontierSummary]()

    for profile_index in range(len(default_single_core_scale_profiles())):
        append_frontier_for_profile(summaries, backend, profile_index)

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    (output_root / "single_core_faithfulness_frontier.json").write_text(
        faithfulness_frontier_summaries_json(summaries)
    )
    print("wrote ", String(output_root / "single_core_faithfulness_frontier.json"))
