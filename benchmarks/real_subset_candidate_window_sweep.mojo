from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    CandidateWindowSweepSummary,
    build_candidate_window_sweep_summary,
    build_candidate_window_sweep_summary_for_plan,
    candidate_window_sweep_summaries_json,
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
from kayak.eval import JudgedTask
from kayak.planning import (
    best_effort_faithfulness_policy,
    centroid_heads_search_plan,
    centroid_postings_flat_search_plan,
    centroid_postings_head_auto_search_plan,
    centroid_postings_blockmax_search_plan,
    centroid_postings_head_search_plan,
    centroid_postings_imputed_search_plan,
    centroid_postings_search_plan,
    document_proxy_search_plan,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import (
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_browsecomp_plus_real_subset_cache,
    ensure_fiqa_real_subset_cache,
    ensure_limit_small_real_subset_cache,
    ensure_scifact_real_subset_cache,
)


comptime CENTROID_HEAD_POSTING_CAP = 16


def append_sweep_for_dataset(
    mut summaries: List[CandidateWindowSweepSummary],
    read backend: ExactCpuBackend,
    dataset_id: String,
    model_name: String,
    collection_root: Path,
    read task: JudgedTask,
) raises:
    var snapshot = load_resolved_collection_snapshot(
        collection_root, SnapshotId("snapshot-0001")
    )

    for candidate_k in standard_candidate_window_sizes(
        task.k, snapshot.snapshot.stats.document_count
    ):
        summaries.append(
            build_candidate_window_sweep_summary(
                backend,
                dataset_id,
                model_name,
                task,
                snapshot,
                candidate_k,
            )
        )
        summaries.append(
            build_candidate_window_sweep_summary_for_plan(
                backend,
                dataset_id,
                model_name,
                task,
                snapshot,
                document_proxy_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
                0,
                0,
            )
        )
        summaries.append(
            build_candidate_window_sweep_summary_for_plan(
                backend,
                dataset_id,
                model_name,
                task,
                snapshot,
                centroid_postings_flat_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
                0,
                0,
            )
        )
        summaries.append(
            build_candidate_window_sweep_summary_for_plan(
                backend,
                dataset_id,
                model_name,
                task,
                snapshot,
                centroid_postings_blockmax_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
                0,
                0,
            )
        )
        summaries.append(
            build_candidate_window_sweep_summary_for_plan(
                backend,
                dataset_id,
                model_name,
                task,
                snapshot,
                centroid_postings_head_auto_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
                0,
                0,
            )
        )
        summaries.append(
            build_candidate_window_sweep_summary_for_plan(
                backend,
                dataset_id,
                model_name,
                task,
                snapshot,
                centroid_heads_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
                0,
                0,
            )
        )
        summaries.append(
            build_candidate_window_sweep_summary_for_plan(
                backend,
                dataset_id,
                model_name,
                task,
                snapshot,
                centroid_postings_head_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
                0,
                0,
            )
        )
        summaries.append(
            build_candidate_window_sweep_summary_for_plan(
                backend,
                dataset_id,
                model_name,
                task,
                snapshot,
                centroid_postings_imputed_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
                0,
                0,
            )
        )
        summaries.append(
            build_candidate_window_sweep_summary_for_plan(
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
                0,
                0,
            )
        )


def main() raises:
    var backend = ExactCpuBackend()
    var summaries = List[CandidateWindowSweepSummary]()

    var scifact_cache = ensure_scifact_real_subset_cache()
    append_sweep_for_dataset(
        summaries,
        backend,
        scifact_cache.stored_task.dataset_id,
        scifact_cache.stored_index.model_name,
        ensure_one_segment_collection_mirror(
            Path(".cache/kayak/scifact_real_subset_collection"),
            CollectionId("scifact_real_subset"),
            TenantId("public"),
            NamespaceId("benchmark"),
            SnapshotId("snapshot-0001"),
            1,
            scifact_cache.stored_index,
            0,
            0,
            CENTROID_HEAD_POSTING_CAP,
        ),
        scifact_cache.stored_task.task,
    )

    var fiqa_cache = ensure_fiqa_real_subset_cache()
    append_sweep_for_dataset(
        summaries,
        backend,
        fiqa_cache.stored_task.dataset_id,
        fiqa_cache.stored_index.model_name,
        ensure_one_segment_collection_mirror(
            Path(".cache/kayak/fiqa_real_subset_collection"),
            CollectionId("fiqa_real_subset"),
            TenantId("public"),
            NamespaceId("benchmark"),
            SnapshotId("snapshot-0001"),
            1,
            fiqa_cache.stored_index,
            0,
            0,
            CENTROID_HEAD_POSTING_CAP,
        ),
        fiqa_cache.stored_task.task,
    )

    var limit_small_cache = ensure_limit_small_real_subset_cache()
    append_sweep_for_dataset(
        summaries,
        backend,
        limit_small_cache.stored_task.dataset_id,
        limit_small_cache.stored_index.model_name,
        ensure_one_segment_collection_mirror(
            Path(".cache/kayak/limit_small_real_subset_collection"),
            CollectionId("limit_small_real_subset"),
            TenantId("public"),
            NamespaceId("benchmark"),
            SnapshotId("snapshot-0001"),
            1,
            limit_small_cache.stored_index,
            0,
            0,
            CENTROID_HEAD_POSTING_CAP,
        ),
        limit_small_cache.stored_task.task,
    )

    var browsecomp_cache = ensure_browsecomp_plus_real_subset_cache()
    append_sweep_for_dataset(
        summaries,
        backend,
        browsecomp_cache.stored_task.dataset_id,
        browsecomp_cache.stored_index.model_name,
        ensure_one_segment_collection_mirror(
            Path(".cache/kayak/browsecomp_plus_real_subset_collection"),
            CollectionId("browsecomp_plus_real_subset"),
            TenantId("public"),
            NamespaceId("benchmark"),
            SnapshotId("snapshot-0001"),
            1,
            browsecomp_cache.stored_index,
            0,
            0,
            CENTROID_HEAD_POSTING_CAP,
        ),
        browsecomp_cache.stored_task.task,
    )

    var browsecomp_gold_cache = ensure_browsecomp_plus_gold_real_subset_cache()
    append_sweep_for_dataset(
        summaries,
        backend,
        browsecomp_gold_cache.stored_task.dataset_id,
        browsecomp_gold_cache.stored_index.model_name,
        ensure_one_segment_collection_mirror(
            Path(".cache/kayak/browsecomp_plus_gold_real_subset_collection"),
            CollectionId("browsecomp_plus_gold_real_subset"),
            TenantId("public"),
            NamespaceId("benchmark"),
            SnapshotId("snapshot-0001"),
            1,
            browsecomp_gold_cache.stored_index,
            0,
            0,
            CENTROID_HEAD_POSTING_CAP,
        ),
        browsecomp_gold_cache.stored_task.task,
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    (output_root / "public_candidate_window_sweep.json").write_text(
        candidate_window_sweep_summaries_json(summaries)
    )
    print("wrote ", String(output_root / "public_candidate_window_sweep.json"))
