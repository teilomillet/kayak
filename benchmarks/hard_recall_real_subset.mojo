from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import (
    best_effort_faithfulness_policy,
    centroid_heads_search_plan,
    centroid_postings_head_auto_search_plan,
    centroid_postings_head_search_plan,
    centroid_postings_imputed_search_plan,
    centroid_postings_search_plan,
    document_proxy_search_plan,
    exact_full_scan_search_plan,
)
from kayak.benchmarks import (
    StageAwareSearchSummary,
    build_stage_aware_search_summary,
    stage_aware_search_summaries_json,
)
from kayak.collections import (
    CollectionId,
    NamespaceId,
    SnapshotId,
    TenantId,
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import (
    StoredJudgedTask,
    StoredPackedIndex,
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_browsecomp_plus_real_subset_cache,
)


comptime CENTROID_HEAD_POSTING_CAP = 16


def capped_candidate_budgets(final_k: Int, document_count: Int) -> List[Int]:
    var budgets = List[Int]()
    var candidate_k = final_k
    if candidate_k > document_count:
        candidate_k = document_count
    budgets.append(candidate_k)

    candidate_k = 100
    if candidate_k > document_count:
        candidate_k = document_count
    if budgets[len(budgets) - 1] != candidate_k:
        budgets.append(candidate_k)

    candidate_k = 1000
    if candidate_k > document_count:
        candidate_k = document_count
    if budgets[len(budgets) - 1] != candidate_k:
        budgets.append(candidate_k)

    return budgets^


def append_hard_recall_summaries(
    mut summaries: List[StageAwareSearchSummary],
    collection_name: String,
    read stored_task: StoredJudgedTask,
    read stored_index: StoredPackedIndex,
) raises:
    var collection_root = ensure_one_segment_collection_mirror(
        Path(".cache/kayak/" + collection_name + "_collection"),
        CollectionId(collection_name),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        stored_index,
        0,
        0,
        CENTROID_HEAD_POSTING_CAP,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )
    var task = stored_task.task.copy()

    summaries.append(
        build_stage_aware_search_summary(
            ExactCpuBackend(),
            stored_task,
            snapshot,
            exact_full_scan_search_plan(task.k, task.k),
        )
    )

    for candidate_k in capped_candidate_budgets(
        task.k,
        snapshot.snapshot.stats.document_count,
    ):
        summaries.append(
            build_stage_aware_search_summary(
                ExactCpuBackend(),
                stored_task,
                snapshot,
                document_proxy_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
            )
        )
        summaries.append(
            build_stage_aware_search_summary(
                ExactCpuBackend(),
                stored_task,
                snapshot,
                centroid_heads_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
            )
        )
        summaries.append(
            build_stage_aware_search_summary(
                ExactCpuBackend(),
                stored_task,
                snapshot,
                centroid_postings_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
            )
        )
        summaries.append(
            build_stage_aware_search_summary(
                ExactCpuBackend(),
                stored_task,
                snapshot,
                centroid_postings_head_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
            )
        )
        summaries.append(
            build_stage_aware_search_summary(
                ExactCpuBackend(),
                stored_task,
                snapshot,
                centroid_postings_head_auto_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
            )
        )
        summaries.append(
            build_stage_aware_search_summary(
                ExactCpuBackend(),
                stored_task,
                snapshot,
                centroid_postings_imputed_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
            )
        )


def main() raises:
    var summaries = List[StageAwareSearchSummary]()
    var browsecomp_cache = ensure_browsecomp_plus_real_subset_cache()
    var browsecomp_gold_cache = ensure_browsecomp_plus_gold_real_subset_cache()

    append_hard_recall_summaries(
        summaries,
        "browsecomp_plus_real_subset",
        browsecomp_cache.stored_task,
        browsecomp_cache.stored_index,
    )
    append_hard_recall_summaries(
        summaries,
        "browsecomp_plus_gold_real_subset",
        browsecomp_gold_cache.stored_task,
        browsecomp_gold_cache.stored_index,
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    (output_root / "hard_recall_stage_aware_search.json").write_text(
        stage_aware_search_summaries_json(summaries)
    )
    print("wrote ", String(output_root / "hard_recall_stage_aware_search.json"))
