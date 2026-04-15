from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    FaithfulnessFrontierSummary,
    build_faithfulness_frontier_summary_for_plan,
    faithfulness_frontier_summaries_json,
    standard_candidate_window_sizes,
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
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import ensure_browsecomp_plus_gold_real_subset_cache


comptime CENTROID_HEAD_POSTING_CAP = 16


def append_summary(
    mut summaries: List[FaithfulnessFrontierSummary],
    read summary: FaithfulnessFrontierSummary,
):
    summaries.append(summary.copy())
    print(
        "Mean: ",
        summary.mean_search_seconds,
        " generator=",
        summary.candidate_generator.kind,
        " candidate_k=",
        summary.candidate_k,
        " posting_cap=",
        summary.posting_cap,
        " ndcg=",
        summary.mean_ndcg_at_k,
        " recall=",
        summary.mean_candidate_recall_at_final_k,
    )


def main() raises:
    var backend = ExactCpuBackend()
    var summaries = List[FaithfulnessFrontierSummary]()
    var tenant_id = TenantId("public")
    var namespace_id = NamespaceId("benchmark")
    var snapshot_id = SnapshotId("snapshot-0001")
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var collection_root = ensure_one_segment_collection_mirror(
        Path(".cache/kayak/head_auto_browsecomp_gold_probe"),
        CollectionId("head_auto_browsecomp_gold_probe"),
        tenant_id,
        namespace_id,
        snapshot_id,
        1,
        cache.stored_index,
        task.nominal_document_vector_count,
        cache.stored_index.index.vector_dim,
        CENTROID_HEAD_POSTING_CAP,
    )
    var snapshot = load_resolved_collection_snapshot(collection_root, snapshot_id)

    for candidate_k in standard_candidate_window_sizes(
        task.k,
        snapshot.snapshot.stats.document_count,
    ):
        append_summary(
            summaries,
            build_faithfulness_frontier_summary_for_plan(
                backend,
                cache.stored_task,
                snapshot,
                centroid_heads_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
                task.nominal_query_vector_count,
                cache.stored_index.index.vector_dim,
                CENTROID_HEAD_POSTING_CAP,
            ),
        )
        append_summary(
            summaries,
            build_faithfulness_frontier_summary_for_plan(
                backend,
                cache.stored_task,
                snapshot,
                centroid_postings_head_auto_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
                task.nominal_query_vector_count,
                cache.stored_index.index.vector_dim,
                CENTROID_HEAD_POSTING_CAP,
            ),
        )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "head_auto_browsecomp_gold_probe.json"
    output_path.write_text(faithfulness_frontier_summaries_json(summaries))
    print("wrote ", String(output_path))
