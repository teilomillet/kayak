from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    VectorBudgetSweepSummary,
    build_vector_budget_sweep_summary,
    build_vector_budget_sweep_summary_for_plan,
    standard_document_vector_budget_sizes,
    standard_query_vector_budget_sizes,
    vector_budget_sweep_summaries_json,
)
from kayak.collections import (
    CollectionId,
    NamespaceId,
    SnapshotId,
    TenantId,
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
)
from kayak.eval import JudgedTask
from kayak.planning import (
    best_effort_faithfulness_policy,
    centroid_postings_search_plan,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import (
    StoredPackedIndex,
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_browsecomp_plus_real_subset_cache,
    ensure_fiqa_real_subset_cache,
    ensure_limit_small_real_subset_cache,
    ensure_scifact_real_subset_cache,
)


def append_vector_budget_summaries_for_dataset(
    mut summaries: List[VectorBudgetSweepSummary],
    read backend: ExactCpuBackend,
    dataset_id: String,
    model_name: String,
    collection_name: String,
    read task: JudgedTask,
    tenant_id: TenantId,
    namespace_id: NamespaceId,
    snapshot_id: SnapshotId,
    generation: Int,
    collection_root_prefix: String,
    read stored_index: StoredPackedIndex,
) raises:
    for document_vector_budget in standard_document_vector_budget_sizes(
        task.nominal_document_vector_count
    ):
        var collection_root = ensure_one_segment_collection_mirror(
            Path(collection_root_prefix + "_doc_budget_" + String(document_vector_budget)),
            CollectionId(collection_name),
            tenant_id,
            namespace_id,
            snapshot_id,
            generation,
            stored_index,
            document_vector_budget,
        )
        var snapshot = load_resolved_collection_snapshot(collection_root, snapshot_id)
        var candidate_k = task.k * 4
        if candidate_k > snapshot.snapshot.stats.document_count:
            candidate_k = snapshot.snapshot.stats.document_count

        for query_vector_budget in standard_query_vector_budget_sizes(
            task.nominal_query_vector_count
        ):
            summaries.append(
                build_vector_budget_sweep_summary(
                    backend,
                    dataset_id,
                    model_name,
                    task,
                    snapshot,
                    query_vector_budget,
                    document_vector_budget,
                    candidate_k,
                )
            )

    for centroid_budget in standard_document_vector_budget_sizes(
        stored_index.index.vector_dim
    ):
        var collection_root = ensure_one_segment_collection_mirror(
            Path(collection_root_prefix + "_centroid_budget_" + String(centroid_budget)),
            CollectionId(collection_name),
            tenant_id,
            namespace_id,
            snapshot_id,
            generation,
            stored_index,
            0,
            centroid_budget,
        )
        var snapshot = load_resolved_collection_snapshot(collection_root, snapshot_id)
        var candidate_k = task.k * 4
        if candidate_k > snapshot.snapshot.stats.document_count:
            candidate_k = snapshot.snapshot.stats.document_count

        for query_vector_budget in standard_query_vector_budget_sizes(
            task.nominal_query_vector_count
        ):
            summaries.append(
                build_vector_budget_sweep_summary_for_plan(
                    backend,
                    dataset_id,
                    model_name,
                    task,
                    snapshot,
                    centroid_postings_search_plan(
                        task.k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                    ),
                    query_vector_budget,
                    centroid_budget,
                )
            )


def main() raises:
    var backend = ExactCpuBackend()
    var summaries = List[VectorBudgetSweepSummary]()
    var tenant_id = TenantId("public")
    var namespace_id = NamespaceId("benchmark")
    var snapshot_id = SnapshotId("snapshot-0001")

    var scifact_cache = ensure_scifact_real_subset_cache()
    append_vector_budget_summaries_for_dataset(
        summaries,
        backend,
        scifact_cache.stored_task.dataset_id,
        scifact_cache.stored_index.model_name,
        "scifact_real_subset",
        scifact_cache.stored_task.task,
        tenant_id,
        namespace_id,
        snapshot_id,
        1,
        ".cache/kayak/scifact_real_subset_collection_proxy",
        scifact_cache.stored_index,
    )

    var fiqa_cache = ensure_fiqa_real_subset_cache()
    append_vector_budget_summaries_for_dataset(
        summaries,
        backend,
        fiqa_cache.stored_task.dataset_id,
        fiqa_cache.stored_index.model_name,
        "fiqa_real_subset",
        fiqa_cache.stored_task.task,
        tenant_id,
        namespace_id,
        snapshot_id,
        1,
        ".cache/kayak/fiqa_real_subset_collection_proxy",
        fiqa_cache.stored_index,
    )

    var limit_small_cache = ensure_limit_small_real_subset_cache()
    append_vector_budget_summaries_for_dataset(
        summaries,
        backend,
        limit_small_cache.stored_task.dataset_id,
        limit_small_cache.stored_index.model_name,
        "limit_small_real_subset",
        limit_small_cache.stored_task.task,
        tenant_id,
        namespace_id,
        snapshot_id,
        1,
        ".cache/kayak/limit_small_real_subset_collection_proxy",
        limit_small_cache.stored_index,
    )

    var browsecomp_cache = ensure_browsecomp_plus_real_subset_cache()
    append_vector_budget_summaries_for_dataset(
        summaries,
        backend,
        browsecomp_cache.stored_task.dataset_id,
        browsecomp_cache.stored_index.model_name,
        "browsecomp_plus_real_subset",
        browsecomp_cache.stored_task.task,
        tenant_id,
        namespace_id,
        snapshot_id,
        1,
        ".cache/kayak/browsecomp_plus_real_subset_collection_proxy",
        browsecomp_cache.stored_index,
    )

    var browsecomp_gold_cache = ensure_browsecomp_plus_gold_real_subset_cache()
    append_vector_budget_summaries_for_dataset(
        summaries,
        backend,
        browsecomp_gold_cache.stored_task.dataset_id,
        browsecomp_gold_cache.stored_index.model_name,
        "browsecomp_plus_gold_real_subset",
        browsecomp_gold_cache.stored_task.task,
        tenant_id,
        namespace_id,
        snapshot_id,
        1,
        ".cache/kayak/browsecomp_plus_gold_real_subset_collection_proxy",
        browsecomp_gold_cache.stored_index,
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    (output_root / "public_vector_budget_sweep.json").write_text(
        vector_budget_sweep_summaries_json(summaries)
    )
    print("wrote ", String(output_root / "public_vector_budget_sweep.json"))
