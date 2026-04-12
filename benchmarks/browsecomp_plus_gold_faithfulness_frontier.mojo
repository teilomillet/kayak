from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    FaithfulnessFrontierSummary,
    build_faithfulness_frontier_summary_for_plan,
    faithfulness_frontier_summaries_json,
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
    centroid_postings_search_plan,
    document_proxy_search_plan,
    exact_full_scan_search_plan,
)
from kayak.runtime import ExactCpuBackend
from kayak.scoring import ExactScoringConfig
from kayak.storage import ensure_browsecomp_plus_gold_real_subset_cache


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
        summary.candidate_generator_kind,
        " candidate_k=",
        summary.candidate_k,
        " posting_cap=",
        summary.posting_cap,
        " recall=",
        summary.mean_candidate_recall_at_final_k,
    )


def append_summary(
    mut summaries: List[FaithfulnessFrontierSummary],
    read summary: FaithfulnessFrontierSummary,
):
    summaries.append(summary.copy())
    print_frontier_mean_line(summary)


def main() raises:
    var backend = single_core_backend()
    var summaries = List[FaithfulnessFrontierSummary]()
    var tenant_id = TenantId("public")
    var namespace_id = NamespaceId("benchmark")
    var snapshot_id = SnapshotId("snapshot-0001")
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var full_proxy_budget = task.nominal_document_vector_count
    var full_centroid_budget = cache.stored_index.index.vector_dim

    var base_root = ensure_one_segment_collection_mirror(
        Path(".cache/kayak/browsecomp_plus_gold_frontier_base"),
        CollectionId("browsecomp_plus_gold_frontier_base"),
        tenant_id,
        namespace_id,
        snapshot_id,
        1,
        cache.stored_index,
        full_proxy_budget,
        full_centroid_budget,
    )
    var base_snapshot = load_resolved_collection_snapshot(base_root, snapshot_id)
    append_summary(
        summaries,
        build_faithfulness_frontier_summary_for_plan(
            backend,
            cache.stored_task,
            base_snapshot,
            exact_full_scan_search_plan(task.k, task.k),
            task.nominal_query_vector_count,
            task.nominal_document_vector_count,
            0,
        ),
    )

    for candidate_k in standard_candidate_window_sizes(
        task.k,
        base_snapshot.snapshot.stats.document_count,
    ):
        append_summary(
            summaries,
            build_faithfulness_frontier_summary_for_plan(
                backend,
                cache.stored_task,
                base_snapshot,
                document_proxy_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
                task.nominal_query_vector_count,
                full_proxy_budget,
                0,
            ),
        )
        append_summary(
            summaries,
            build_faithfulness_frontier_summary_for_plan(
                backend,
                cache.stored_task,
                base_snapshot,
                centroid_postings_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
                task.nominal_query_vector_count,
                full_centroid_budget,
                0,
            ),
        )
    for posting_cap in standard_posting_cap_sizes(
        base_snapshot.snapshot.stats.document_count
    ):
        var heads_root = ensure_one_segment_collection_mirror(
            Path(
                ".cache/kayak/browsecomp_plus_gold_frontier_heads_"
                + String(posting_cap)
            ),
            CollectionId("browsecomp_plus_gold_frontier_heads_" + String(posting_cap)),
            tenant_id,
            namespace_id,
            snapshot_id,
            1,
            cache.stored_index,
            full_proxy_budget,
            full_centroid_budget,
            posting_cap,
        )
        var heads_snapshot = load_resolved_collection_snapshot(
            heads_root,
            snapshot_id,
        )

        for candidate_k in standard_candidate_window_sizes(
            task.k,
            heads_snapshot.snapshot.stats.document_count,
        ):
            append_summary(
                summaries,
                build_faithfulness_frontier_summary_for_plan(
                    backend,
                    cache.stored_task,
                    heads_snapshot,
                    centroid_heads_search_plan(
                        task.k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                    ),
                    task.nominal_query_vector_count,
                    full_centroid_budget,
                    posting_cap,
                ),
            )
            append_summary(
                summaries,
                build_faithfulness_frontier_summary_for_plan(
                    backend,
                    cache.stored_task,
                    heads_snapshot,
                    centroid_postings_head_auto_search_plan(
                        task.k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                    ),
                    task.nominal_query_vector_count,
                    full_centroid_budget,
                    posting_cap,
                ),
            )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    (output_root / "browsecomp_plus_gold_faithfulness_frontier.json").write_text(
        faithfulness_frontier_summaries_json(summaries)
    )
    print("wrote ", String(output_root / "browsecomp_plus_gold_faithfulness_frontier.json"))
