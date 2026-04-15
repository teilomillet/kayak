from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import (
    best_effort_faithfulness_policy,
    centroid_heads_search_plan,
    centroid_postings_blockmax_search_plan,
    centroid_postings_flat_search_plan,
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
    default_contradiction_hard_recall_profiles,
    make_contradiction_hard_recall_fixture,
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


def contradiction_hard_recall_candidate_budgets(
    document_count: Int
) -> List[Int]:
    var budgets = List[Int]()

    for requested_budget in [2, 4, 8, 16, 32, 64, 128]:
        var candidate_k = requested_budget
        if candidate_k > document_count:
            candidate_k = document_count
        if len(budgets) == 0 or budgets[len(budgets) - 1] != candidate_k:
            budgets.append(candidate_k)

    return budgets^


def print_stage_aware_summary(read summary: StageAwareSearchSummary):
    print(
        "Mean: ",
        summary.mean_search_seconds,
        " family=",
        summary.family,
        " slice=",
        summary.slice_name,
        " generator=",
        summary.plan.candidate_generator.kind,
        " candidate_k=",
        summary.candidate_k,
        " candidate_recall=",
        summary.mean_candidate_recall_at_final_k,
        " primary=",
        summary.primary_value,
    )


def append_contradiction_hard_recall_summaries(
    mut summaries: List[StageAwareSearchSummary]
) raises:
    var backend = ExactCpuBackend()
    var tenant_id = TenantId("public")
    var namespace_id = NamespaceId("benchmark")
    var snapshot_id = SnapshotId("snapshot-0001")

    for profile in default_contradiction_hard_recall_profiles():
        var fixture = make_contradiction_hard_recall_fixture(profile)
        var collection_root = ensure_one_segment_collection_mirror(
            Path(".cache/kayak/contradiction_hard_recall_" + profile.slice_name),
            CollectionId("contradiction-hard-recall-" + profile.slice_name),
            tenant_id,
            namespace_id,
            snapshot_id,
            1,
            fixture.stored_index,
            0,
            0,
            profile.centroid_head_posting_cap,
        )
        var snapshot = load_resolved_collection_snapshot(collection_root, snapshot_id)

        summaries.append(
            build_stage_aware_search_summary(
                backend,
                fixture.stored_task,
                snapshot,
                exact_full_scan_search_plan(
                    profile.final_k,
                    profile.final_k,
                ),
            )
        )
        print_stage_aware_summary(summaries[len(summaries) - 1])

        for candidate_k in contradiction_hard_recall_candidate_budgets(
            snapshot.snapshot.stats.document_count
        ):
            summaries.append(
                build_stage_aware_search_summary(
                    backend,
                    fixture.stored_task,
                    snapshot,
                    document_proxy_search_plan(
                        profile.final_k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                    ),
                )
            )
            print_stage_aware_summary(summaries[len(summaries) - 1])
            summaries.append(
                build_stage_aware_search_summary(
                    backend,
                    fixture.stored_task,
                    snapshot,
                    centroid_postings_search_plan(
                        profile.final_k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                    ),
                )
            )
            print_stage_aware_summary(summaries[len(summaries) - 1])
            summaries.append(
                build_stage_aware_search_summary(
                    backend,
                    fixture.stored_task,
                    snapshot,
                    centroid_postings_flat_search_plan(
                        profile.final_k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                    ),
                )
            )
            print_stage_aware_summary(summaries[len(summaries) - 1])
            summaries.append(
                build_stage_aware_search_summary(
                    backend,
                    fixture.stored_task,
                    snapshot,
                    centroid_heads_search_plan(
                        profile.final_k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                    ),
                )
            )
            print_stage_aware_summary(summaries[len(summaries) - 1])
            summaries.append(
                build_stage_aware_search_summary(
                    backend,
                    fixture.stored_task,
                    snapshot,
                    centroid_postings_head_search_plan(
                        profile.final_k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                    ),
                )
            )
            print_stage_aware_summary(summaries[len(summaries) - 1])
            summaries.append(
                build_stage_aware_search_summary(
                    backend,
                    fixture.stored_task,
                    snapshot,
                    centroid_postings_head_auto_search_plan(
                        profile.final_k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                    ),
                )
            )
            print_stage_aware_summary(summaries[len(summaries) - 1])
            summaries.append(
                build_stage_aware_search_summary(
                    backend,
                    fixture.stored_task,
                    snapshot,
                    centroid_postings_blockmax_search_plan(
                        profile.final_k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                    ),
                )
            )
            print_stage_aware_summary(summaries[len(summaries) - 1])
            summaries.append(
                build_stage_aware_search_summary(
                    backend,
                    fixture.stored_task,
                    snapshot,
                    centroid_postings_imputed_search_plan(
                        profile.final_k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                    ),
                )
            )
            print_stage_aware_summary(summaries[len(summaries) - 1])


def main() raises:
    var summaries = List[StageAwareSearchSummary]()
    append_contradiction_hard_recall_summaries(summaries)

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    (output_root / "contradiction_hard_recall_stage_aware_search.json").write_text(
        stage_aware_search_summaries_json(summaries)
    )
    print(
        "wrote ",
        String(output_root / "contradiction_hard_recall_stage_aware_search.json"),
    )
