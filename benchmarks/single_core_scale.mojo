from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    StageAwareSearchSummary,
    default_single_core_scale_profiles,
    make_single_core_scale_fixture,
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
from kayak.planning import (
    best_effort_faithfulness_policy,
    centroid_heads_search_plan,
    centroid_postings_head_auto_search_plan,
    centroid_postings_search_plan,
    document_proxy_search_plan,
    exact_full_scan_search_plan,
)
from kayak.runtime import ExactCpuBackend
from kayak.scoring import ExactScoringConfig


def single_core_backend() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_parallel_scoring = False
    return ExactCpuBackend(config^)


def print_stage_aware_mean_line(read summary: StageAwareSearchSummary):
    print(
        "Mean: ",
        summary.mean_search_seconds,
        " family=",
        summary.family,
        " slice=",
        summary.slice_name,
        " generator=",
        summary.candidate_generator_kind,
        " docs=",
        summary.document_count,
        " vectors=",
        summary.vector_count,
        " candidate_vectors=",
        summary.candidate_stage_vector_count,
    )


def append_scale_summaries(
    mut summaries: List[StageAwareSearchSummary]
) raises:
    var backend = single_core_backend()
    var tenant_id = TenantId("public")
    var namespace_id = NamespaceId("benchmark")
    var snapshot_id = SnapshotId("snapshot-0001")

    for profile in default_single_core_scale_profiles():
        var fixture = make_single_core_scale_fixture(profile)
        var collection_root = ensure_one_segment_collection_mirror(
            Path(".cache/kayak/single_core_scale_" + profile.slice_name),
            CollectionId("single-core-scale-" + profile.slice_name),
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
        var candidate_k = profile.candidate_k
        if candidate_k > snapshot.snapshot.stats.document_count:
            candidate_k = snapshot.snapshot.stats.document_count

        summaries.append(
            build_stage_aware_search_summary(
                backend,
                fixture.stored_task,
                snapshot,
                exact_full_scan_search_plan(profile.final_k, profile.final_k),
            )
        )
        print_stage_aware_mean_line(summaries[len(summaries) - 1])
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
        print_stage_aware_mean_line(summaries[len(summaries) - 1])
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
        print_stage_aware_mean_line(summaries[len(summaries) - 1])
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
        print_stage_aware_mean_line(summaries[len(summaries) - 1])
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
        print_stage_aware_mean_line(summaries[len(summaries) - 1])


def main() raises:
    var summaries = List[StageAwareSearchSummary]()
    append_scale_summaries(summaries)

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    (output_root / "single_core_scale.json").write_text(
        stage_aware_search_summaries_json(summaries)
    )
    print("wrote ", String(output_root / "single_core_scale.json"))
