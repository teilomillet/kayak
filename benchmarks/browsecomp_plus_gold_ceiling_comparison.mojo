from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    CeilingComparisonSummary,
    build_exact_clause_text_ceiling_summary,
    build_exact_full_scan_ceiling_summary,
    build_stage_aware_ceiling_summary_for_plan,
    ceiling_comparison_summaries_json,
    ensure_public_benchmark_dataset_collection_mirror,
    load_public_benchmark_dataset,
    require_public_benchmark_dataset_loaded_text_corpus,
)
from kayak.collections import (
    SnapshotId,
    load_resolved_collection_snapshot,
)
from kayak.planning import (
    best_effort_faithfulness_policy,
    document_proxy_search_plan,
)
from kayak.runtime import ExactCpuBackend


def main() raises:
    var dataset = load_public_benchmark_dataset(
        "browsecomp_plus_gold",
        load_text_corpus=True,
    )
    var document_texts = require_public_benchmark_dataset_loaded_text_corpus(dataset)
    var backend = ExactCpuBackend()
    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var snapshot_id = SnapshotId("snapshot-0001")
    var collection_root = ensure_public_benchmark_dataset_collection_mirror(
        dataset,
        "ceiling_base",
        dataset.stored_task.task.nominal_document_vector_count,
        0,
        0,
        include_frontier_gem_graph=True,
    )
    var snapshot = load_resolved_collection_snapshot(collection_root, snapshot_id)
    var summaries = List[CeilingComparisonSummary]()

    summaries.append(
        build_exact_full_scan_ceiling_summary(
            backend,
            dataset.stored_task,
            dataset.stored_index,
        )
    )

    for candidate_k in [20, 40]:
        summaries.append(
            build_stage_aware_ceiling_summary_for_plan(
                backend,
                dataset.stored_task,
                snapshot,
                document_proxy_search_plan(
                    dataset.stored_task.task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
            )
        )
        summaries.append(
            build_exact_clause_text_ceiling_summary(
                backend,
                dataset.stored_task,
                dataset.stored_index,
                document_texts,
                candidate_k,
            )
        )
    for summary in summaries:
        print(
            "method=",
            summary.method_kind,
            " generator=",
            summary.candidate_generator_kind,
            " stage2=",
            summary.stage2_kind,
            " candidate_k=",
            summary.candidate_k,
            " ndcg=",
            summary.mean_ndcg_at_k,
            " search_s=",
            summary.mean_search_seconds,
        )
    (output_root / "browsecomp_plus_gold_ceiling_comparison.json").write_text(
        ceiling_comparison_summaries_json(summaries)
    )
    print("wrote ", String(output_root / "browsecomp_plus_gold_ceiling_comparison.json"))
