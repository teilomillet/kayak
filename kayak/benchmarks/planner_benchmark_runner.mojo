from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.collections import (
    CollectionId,
    NamespaceId,
    SnapshotId,
    TenantId,
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
    load_snapshot_search_artifact_availability,
)
from kayak.eval import JudgedTask
from kayak.planning import (
    SEARCH_PLANNING_GOAL_BALANCED,
    SEARCH_PLANNING_GOAL_EXACT_ONLY,
    SEARCH_PLANNING_GOAL_LATENCY_FIRST,
    SEARCH_PLANNING_GOAL_NATIVE_MULTI_VECTOR,
    SearchPlanSelectionRequest,
    best_effort_faithfulness_policy,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import (
    StoredJudgedTask,
    StoredPackedIndex,
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_browsecomp_plus_real_subset_cache,
    ensure_fiqa_real_subset_cache,
    ensure_limit_small_real_subset_cache,
    ensure_scifact_real_subset_cache,
)

from .candidate_window_json import standard_candidate_window_sizes
from .gem_frontier_config import frontier_gem_graph_build_config
from .planner_benchmark_json import (
    PlannerBenchmarkSummary,
    build_planner_benchmark_summary,
    planner_benchmark_summaries_json,
)


comptime CENTROID_HEAD_POSTING_CAP = 16


struct PlannerBenchmarkRunOptions(Copyable):
    var output_file_name: String
    var collection_suffix: String
    var full_candidate_window_sweep: Bool
    var candidate_window_limit: Int
    var include_scifact: Bool
    var include_fiqa: Bool
    var include_limit_small: Bool
    var include_browsecomp_plus: Bool
    var include_browsecomp_plus_gold: Bool

    def __init__(
        out self,
        var output_file_name: String,
        var collection_suffix: String,
        full_candidate_window_sweep: Bool,
        candidate_window_limit: Int,
        include_scifact: Bool,
        include_fiqa: Bool,
        include_limit_small: Bool,
        include_browsecomp_plus: Bool,
        include_browsecomp_plus_gold: Bool,
    ):
        self.output_file_name = output_file_name^
        self.collection_suffix = collection_suffix^
        self.full_candidate_window_sweep = full_candidate_window_sweep
        self.candidate_window_limit = candidate_window_limit
        self.include_scifact = include_scifact
        self.include_fiqa = include_fiqa
        self.include_limit_small = include_limit_small
        self.include_browsecomp_plus = include_browsecomp_plus
        self.include_browsecomp_plus_gold = include_browsecomp_plus_gold


def default_planner_benchmark_run_options() -> PlannerBenchmarkRunOptions:
    return PlannerBenchmarkRunOptions(
        "public_planner_benchmark.json",
        "planner_collection",
        True,
        0,
        True,
        True,
        True,
        True,
        True,
    )


def smoke_planner_benchmark_run_options() -> PlannerBenchmarkRunOptions:
    return PlannerBenchmarkRunOptions(
        "public_planner_benchmark_smoke.json",
        "planner_smoke_collection",
        False,
        1,
        True,
        False,
        False,
        False,
        False,
    )


def max_query_vector_budget(read task: JudgedTask) -> Int:
    var max_budget = 0
    for judged_query in task.queries:
        if judged_query.query.vector_count > max_budget:
            max_budget = judged_query.query.vector_count

    if max_budget <= 0:
        return 1
    return max_budget


def append_unique_int(mut values: List[Int], value: Int):
    for existing in values:
        if existing == value:
            return
    values.append(value)


def planner_candidate_window_sizes(
    final_k: Int,
    document_count: Int,
    full_sweep: Bool,
    candidate_window_limit: Int,
) -> List[Int]:
    var all_sizes = standard_candidate_window_sizes(final_k, document_count)
    if full_sweep or len(all_sizes) <= 1:
        if candidate_window_limit <= 0 or candidate_window_limit >= len(all_sizes):
            return all_sizes^

        var limited_sizes = List[Int]()
        for index in range(candidate_window_limit):
            limited_sizes.append(all_sizes[index])
        return limited_sizes^

    var smoke_sizes = List[Int]()
    append_unique_int(smoke_sizes, final_k)
    append_unique_int(smoke_sizes, all_sizes[0])
    if len(all_sizes) > 1:
        append_unique_int(smoke_sizes, all_sizes[len(all_sizes) - 1])
    if candidate_window_limit > 0 and candidate_window_limit < len(smoke_sizes):
        var limited_smoke_sizes = List[Int]()
        for index in range(candidate_window_limit):
            limited_smoke_sizes.append(smoke_sizes[index])
        return limited_smoke_sizes^
    return smoke_sizes^


def append_planner_summaries_for_dataset(
    mut summaries: List[PlannerBenchmarkSummary],
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    collection_root: Path,
    full_candidate_window_sweep: Bool,
    candidate_window_limit: Int,
) raises:
    var snapshot_id = SnapshotId("snapshot-0001")
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        snapshot_id,
    )
    var availability = load_snapshot_search_artifact_availability(
        collection_root,
        snapshot_id,
    )
    var task = stored_task.task.copy()
    var query_budget = max_query_vector_budget(task)

    for candidate_k in planner_candidate_window_sizes(
        task.k,
        snapshot.snapshot.stats.document_count,
        full_candidate_window_sweep,
        candidate_window_limit,
    ):
        summaries.append(
            build_planner_benchmark_summary(
                backend,
                stored_task,
                snapshot,
                availability,
                SearchPlanSelectionRequest(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                    goal=SEARCH_PLANNING_GOAL_BALANCED,
                ),
                query_budget,
                0,
                CENTROID_HEAD_POSTING_CAP,
            )
        )
        summaries.append(
            build_planner_benchmark_summary(
                backend,
                stored_task,
                snapshot,
                availability,
                SearchPlanSelectionRequest(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                    goal=SEARCH_PLANNING_GOAL_LATENCY_FIRST,
                ),
                query_budget,
                0,
                CENTROID_HEAD_POSTING_CAP,
            )
        )
        summaries.append(
            build_planner_benchmark_summary(
                backend,
                stored_task,
                snapshot,
                availability,
                SearchPlanSelectionRequest(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                    goal=SEARCH_PLANNING_GOAL_NATIVE_MULTI_VECTOR,
                ),
                query_budget,
                0,
                CENTROID_HEAD_POSTING_CAP,
            )
        )
        summaries.append(
            build_planner_benchmark_summary(
                backend,
                stored_task,
                snapshot,
                availability,
                SearchPlanSelectionRequest(
                    task.k,
                    candidate_k,
                    best_effort_faithfulness_policy(),
                    goal=SEARCH_PLANNING_GOAL_EXACT_ONLY,
                ),
                query_budget,
                0,
                CENTROID_HEAD_POSTING_CAP,
            )
        )


def append_planner_summaries_for_cache(
    mut summaries: List[PlannerBenchmarkSummary],
    read backend: ExactCpuBackend,
    collection_root: Path,
    read stored_task: StoredJudgedTask,
    read stored_index: StoredPackedIndex,
    full_candidate_window_sweep: Bool,
    candidate_window_limit: Int,
) raises:
    var gem_config = frontier_gem_graph_build_config(
        stored_index,
        max_query_vector_budget(stored_task.task),
    )
    append_planner_summaries_for_dataset(
        summaries,
        backend,
        stored_task,
        ensure_one_segment_collection_mirror(
            collection_root,
            CollectionId(stored_task.dataset_id),
            TenantId("public"),
            NamespaceId("benchmark"),
            SnapshotId("snapshot-0001"),
            1,
            stored_index,
            0,
            0,
            CENTROID_HEAD_POSTING_CAP,
            gem_config.fine_cluster_count,
            gem_config.coarse_cluster_count,
            gem_config.cluster_cutoff,
        ),
        full_candidate_window_sweep,
        candidate_window_limit,
    )


def planner_collection_root_for_dataset(
    dataset_stem: String, collection_suffix: String
) -> Path:
    return Path(".cache/kayak/" + dataset_stem + "_" + collection_suffix)


def planner_benchmark_summaries_for_options(
    read options: PlannerBenchmarkRunOptions
) raises -> List[PlannerBenchmarkSummary]:
    var backend = ExactCpuBackend()
    var summaries = List[PlannerBenchmarkSummary]()

    if options.include_scifact:
        var scifact_cache = ensure_scifact_real_subset_cache()
        append_planner_summaries_for_cache(
            summaries,
            backend,
            planner_collection_root_for_dataset(
                "scifact_real_subset", options.collection_suffix
            ),
            scifact_cache.stored_task,
            scifact_cache.stored_index,
            options.full_candidate_window_sweep,
            options.candidate_window_limit,
        )

    if options.include_fiqa:
        var fiqa_cache = ensure_fiqa_real_subset_cache()
        append_planner_summaries_for_cache(
            summaries,
            backend,
            planner_collection_root_for_dataset(
                "fiqa_real_subset", options.collection_suffix
            ),
            fiqa_cache.stored_task,
            fiqa_cache.stored_index,
            options.full_candidate_window_sweep,
            options.candidate_window_limit,
        )

    if options.include_limit_small:
        var limit_cache = ensure_limit_small_real_subset_cache()
        append_planner_summaries_for_cache(
            summaries,
            backend,
            planner_collection_root_for_dataset(
                "limit_small", options.collection_suffix
            ),
            limit_cache.stored_task,
            limit_cache.stored_index,
            options.full_candidate_window_sweep,
            options.candidate_window_limit,
        )

    if options.include_browsecomp_plus:
        var browsecomp_cache = ensure_browsecomp_plus_real_subset_cache()
        append_planner_summaries_for_cache(
            summaries,
            backend,
            planner_collection_root_for_dataset(
                "browsecomp_plus_real_subset", options.collection_suffix
            ),
            browsecomp_cache.stored_task,
            browsecomp_cache.stored_index,
            options.full_candidate_window_sweep,
            options.candidate_window_limit,
        )

    if options.include_browsecomp_plus_gold:
        var browsecomp_gold_cache = ensure_browsecomp_plus_gold_real_subset_cache()
        append_planner_summaries_for_cache(
            summaries,
            backend,
            planner_collection_root_for_dataset(
                "browsecomp_plus_gold", options.collection_suffix
            ),
            browsecomp_gold_cache.stored_task,
            browsecomp_gold_cache.stored_index,
            options.full_candidate_window_sweep,
            options.candidate_window_limit,
        )

    return summaries^


def write_public_planner_benchmark(
    read options: PlannerBenchmarkRunOptions
) raises -> Path:
    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)

    var output_path = output_root / options.output_file_name
    output_path.write_text(
        planner_benchmark_summaries_json(
            planner_benchmark_summaries_for_options(options)
        )
    )
    return output_path
