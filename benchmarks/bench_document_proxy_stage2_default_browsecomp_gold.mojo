import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.pathlib import Path

from kayak import (
    CollectionId,
    ExactCpuBackend,
    NamespaceId,
    SnapshotId,
    TenantId,
    best_effort_faithfulness_policy,
    candidate_generation_for_plan,
    document_proxy_search_plan,
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
    stage2_result_for_plan,
)
from kayak.planning.candidate_set import CandidateSet


def main() raises:
    var backend = ExactCpuBackend()
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var candidate_k = task.k * 4
    var collection_root = ensure_one_segment_collection_mirror(
        Path(".cache/kayak/browsecomp_plus_gold_document_proxy_stage2_default"),
        CollectionId("browsecomp_plus_gold_document_proxy_stage2_default"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        cache.stored_index,
        task.nominal_document_vector_count,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root, SnapshotId("snapshot-0001")
    )
    var plan = document_proxy_search_plan(
        task.k,
        candidate_k,
        best_effort_faithfulness_policy(),
    )
    var candidate_sets = List[CandidateSet]()
    for judged_query in task.queries:
        candidate_sets.append(
            candidate_generation_for_plan(
                backend,
                judged_query.query,
                snapshot,
                plan,
            )
        )

    var query_index = 0
    print(
        "stage2_result_for_plan dataset=BrowseComp-Plus Gold slice=",
        task.slice_name,
        " candidate_k=",
        candidate_k,
    )

    def score_once() capturing raises:
        bench_compiler.keep(
            stage2_result_for_plan(
                backend,
                task.queries[query_index].query,
                "",
                snapshot,
                candidate_sets[query_index],
                plan,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    benchmark.run[score_once]().print()
