from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    StageAwareSearchSummary,
    build_stage_aware_search_summary,
    ensure_public_benchmark_dataset_collection_mirror,
    load_public_benchmark_dataset,
    stage_aware_search_summaries_json,
    standard_candidate_window_sizes,
)
from kayak.collections import SnapshotId, load_resolved_collection_snapshot
from kayak.planning import (
    best_effort_faithfulness_policy,
    centroid_postings_flat_search_plan,
    centroid_postings_imputed_flat_search_plan,
)
from kayak.runtime import ExactCpuBackend


comptime CENTROID_HEAD_POSTING_CAP = 16


def wide_window_candidate_budgets(final_k: Int, document_count: Int) -> List[Int]:
    var budgets = List[Int]()
    for candidate_k in standard_candidate_window_sizes(final_k, document_count):
        if candidate_k >= final_k * 8 or candidate_k == document_count:
            budgets.append(candidate_k)
    return budgets^


def append_wide_window_native_frontier_for_dataset(
    mut summaries: List[StageAwareSearchSummary], dataset_key: String
) raises:
    var dataset = load_public_benchmark_dataset(dataset_key)
    var collection_root = ensure_public_benchmark_dataset_collection_mirror(
        dataset,
        "wide_window_native_frontier",
        0,
        0,
        CENTROID_HEAD_POSTING_CAP,
        include_frontier_gem_graph=False,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )
    var task = dataset.stored_task.task.copy()
    var backend = ExactCpuBackend()

    for candidate_k in wide_window_candidate_budgets(
        task.k,
        snapshot.snapshot.stats.document_count,
    ):
        summaries.append(
            build_stage_aware_search_summary(
                backend,
                dataset.stored_task,
                snapshot,
                centroid_postings_flat_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
            )
        )
        summaries.append(
            build_stage_aware_search_summary(
                backend,
                dataset.stored_task,
                snapshot,
                centroid_postings_imputed_flat_search_plan(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
            )
        )


def main() raises:
    var summaries = List[StageAwareSearchSummary]()
    for dataset_key in [
        "fiqa_real_subset",
        "browsecomp_plus_real_subset",
        "browsecomp_plus_gold",
    ]:
        append_wide_window_native_frontier_for_dataset(summaries, dataset_key)

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "public_wide_window_native_frontier.json"
    output_path.write_text(stage_aware_search_summaries_json(summaries))
    print("wrote ", String(output_path))
