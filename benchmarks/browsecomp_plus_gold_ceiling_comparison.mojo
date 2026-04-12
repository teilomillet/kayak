from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    CeilingComparisonSummary,
    build_exact_clause_text_ceiling_summary,
    build_exact_full_scan_ceiling_summary,
    build_stage_aware_ceiling_summary_for_plan,
    ceiling_comparison_summaries_json,
)
from kayak.collections import (
    CollectionId,
    NamespaceId,
    SnapshotId,
    TenantId,
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
)
from kayak.interop import load_document_text_corpus_json
from kayak.planning import (
    best_effort_faithfulness_policy,
    document_proxy_search_plan,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import ensure_browsecomp_plus_gold_real_subset_cache


def main() raises:
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var document_texts = load_document_text_corpus_json(
        ".cache/kayak/browsecomp_plus_real_subset/python_task_gold.json"
    )
    var backend = ExactCpuBackend()
    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var tenant_id = TenantId("public")
    var namespace_id = NamespaceId("benchmark")
    var snapshot_id = SnapshotId("snapshot-0001")
    var collection_root = ensure_one_segment_collection_mirror(
        output_root / "browsecomp_plus_gold_ceiling_base",
        CollectionId("browsecomp_plus_gold_ceiling_base"),
        tenant_id,
        namespace_id,
        snapshot_id,
        1,
        cache.stored_index,
        cache.stored_task.task.nominal_document_vector_count,
    )
    var snapshot = load_resolved_collection_snapshot(collection_root, snapshot_id)
    var summaries = List[CeilingComparisonSummary]()

    summaries.append(
        build_exact_full_scan_ceiling_summary(
            backend,
            cache.stored_task,
            cache.stored_index,
        )
    )

    for candidate_k in [20, 40]:
        summaries.append(
            build_stage_aware_ceiling_summary_for_plan(
                backend,
                cache.stored_task,
                snapshot,
                document_proxy_search_plan(
                    cache.stored_task.task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                ),
            )
        )
        summaries.append(
            build_exact_clause_text_ceiling_summary(
                backend,
                cache.stored_task,
                cache.stored_index,
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
            " reranker=",
            summary.reranker_kind,
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
