import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.pathlib import Path

from kayak import (
    CollectionId,
    JudgedTask,
    NamespaceId,
    SnapshotId,
    StoredPackedIndex,
    TenantId,
    centroid_posting_blockmax_scores_for_segment_profiled,
    ensure_browsecomp_plus_real_subset_cache,
    ensure_one_segment_collection_mirror,
    ensure_scifact_real_subset_cache,
    loaded_segment_stored_centroid_postings_index,
    load_resolved_collection_snapshot,
)
from kayak.collections import ResolvedCollectionSnapshot
from kayak.index import CentroidPostingIndex


comptime CENTROID_HEAD_POSTING_CAP = 16


def append_probe_candidate_k(
    mut budgets: List[Int], requested_k: Int, document_count: Int
):
    var candidate_k = requested_k
    if candidate_k > document_count:
        candidate_k = document_count
    if candidate_k <= 0:
        return
    if len(budgets) != 0 and budgets[len(budgets) - 1] == candidate_k:
        return
    budgets.append(candidate_k)


def probe_candidate_ks(final_k: Int, document_count: Int) -> List[Int]:
    var budgets = List[Int]()
    append_probe_candidate_k(budgets, final_k * 4, document_count)
    append_probe_candidate_k(budgets, final_k * 8, document_count)
    append_probe_candidate_k(budgets, 128, document_count)
    append_probe_candidate_k(budgets, 1000, document_count)
    return budgets^


def load_profile_snapshot(
    collection_name: String,
    collection_root: Path,
    read stored_index: StoredPackedIndex,
) raises -> ResolvedCollectionSnapshot:
    var root = ensure_one_segment_collection_mirror(
        collection_root,
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
    return load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))


def benchmark_blockmax_mean_seconds(
    read task: JudgedTask,
    read index: CentroidPostingIndex,
    candidate_k: Int,
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            centroid_posting_blockmax_scores_for_segment_profiled(
                task.queries[query_index].query.token_vectors,
                index,
                candidate_k,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return report.mean()


def run_dataset_probe(
    dataset_name: String,
    collection_name: String,
    collection_root: Path,
    read task: JudgedTask,
    read stored_index: StoredPackedIndex,
) raises:
    var snapshot = load_profile_snapshot(
        collection_name,
        collection_root,
        stored_index,
    )
    var index = loaded_segment_stored_centroid_postings_index(
        snapshot.segments[0]
    ).index.copy()

    for candidate_k in probe_candidate_ks(
        task.k,
        snapshot.snapshot.stats.document_count,
    ):
        print("dataset= ", dataset_name, "  candidate_k= ", candidate_k)
        print(
            "mean_stage1_seconds= ",
            benchmark_blockmax_mean_seconds(task, index, candidate_k),
        )
        print("")


def main() raises:
    var scifact_cache = ensure_scifact_real_subset_cache()
    run_dataset_probe(
        "SciFact",
        "scifact_real_subset_profile",
        Path(".cache/kayak/scifact_real_subset_stage1_profile"),
        scifact_cache.stored_task.task,
        scifact_cache.stored_index,
    )

    var browsecomp_cache = ensure_browsecomp_plus_real_subset_cache()
    run_dataset_probe(
        "BrowseComp-Plus",
        "browsecomp_plus_real_subset_blockmax_probe",
        Path(".cache/kayak/browsecomp_plus_real_subset_blockmax_probe"),
        browsecomp_cache.stored_task.task,
        browsecomp_cache.stored_index,
    )
