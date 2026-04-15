from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.collections import (
    SnapshotId,
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
    clause_text_stage3_verifier_operator,
    exact_late_interaction_stage2_reference_operator,
    none_stage3_verifier_operator,
    Stage2ReferenceOperator,
    Stage3VerifierOperator,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import StoredJudgedTask

from .candidate_window_json import standard_candidate_window_sizes
from .planner_benchmark_json import (
    PlannerBenchmarkSummary,
    build_planner_benchmark_summary,
    planner_benchmark_summaries_json,
)
from .public_benchmark_dataset import (
    PublicBenchmarkDataset,
    ensure_public_benchmark_dataset_collection_mirror,
    load_public_benchmark_dataset,
    public_benchmark_dataset_has_loaded_text_corpus,
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
    var include_bright_stackoverflow: Bool
    var include_legal_rag_bench: Bool
    var include_lemb_narrativeqa: Bool
    var include_r2med_biology: Bool
    var include_browsecomp_plus: Bool
    var include_browsecomp_plus_gold: Bool
    var include_clause_text_stage3_when_text_available: Bool

    def __init__(
        out self,
        var output_file_name: String,
        var collection_suffix: String,
        full_candidate_window_sweep: Bool,
        candidate_window_limit: Int,
        include_scifact: Bool,
        include_fiqa: Bool,
        include_limit_small: Bool,
        include_bright_stackoverflow: Bool,
        include_legal_rag_bench: Bool,
        include_lemb_narrativeqa: Bool,
        include_r2med_biology: Bool,
        include_browsecomp_plus: Bool,
        include_browsecomp_plus_gold: Bool,
        include_clause_text_stage3_when_text_available: Bool = False,
    ):
        self.output_file_name = output_file_name^
        self.collection_suffix = collection_suffix^
        self.full_candidate_window_sweep = full_candidate_window_sweep
        self.candidate_window_limit = candidate_window_limit
        self.include_scifact = include_scifact
        self.include_fiqa = include_fiqa
        self.include_limit_small = include_limit_small
        self.include_bright_stackoverflow = include_bright_stackoverflow
        self.include_legal_rag_bench = include_legal_rag_bench
        self.include_lemb_narrativeqa = include_lemb_narrativeqa
        self.include_r2med_biology = include_r2med_biology
        self.include_browsecomp_plus = include_browsecomp_plus
        self.include_browsecomp_plus_gold = include_browsecomp_plus_gold
        self.include_clause_text_stage3_when_text_available = (
            include_clause_text_stage3_when_text_available
        )


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
        False,
        True,
        False,
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


struct PlannerBenchmarkRefinementSetup(Copyable):
    var stage2_reference_operator: Stage2ReferenceOperator
    var stage3_verifier: Stage3VerifierOperator

    def __init__(
        out self,
        read stage2_reference_operator: Stage2ReferenceOperator,
        read stage3_verifier: Stage3VerifierOperator,
    ):
        self.stage2_reference_operator = stage2_reference_operator.copy()
        self.stage3_verifier = stage3_verifier.copy()


def planner_benchmark_refinement_setups(
    include_clause_text_stage3_when_text_available: Bool,
    has_text_corpus: Bool,
) raises -> List[PlannerBenchmarkRefinementSetup]:
    var refinement_setups = List[PlannerBenchmarkRefinementSetup]()
    refinement_setups.append(
        PlannerBenchmarkRefinementSetup(
            exact_late_interaction_stage2_reference_operator(),
            none_stage3_verifier_operator(),
        )
    )
    if include_clause_text_stage3_when_text_available and has_text_corpus:
        refinement_setups.append(
            PlannerBenchmarkRefinementSetup(
                exact_late_interaction_stage2_reference_operator(),
                clause_text_stage3_verifier_operator(),
            )
        )
    return refinement_setups^


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
    read refinement_setups: List[PlannerBenchmarkRefinementSetup],
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
        for refinement_setup in refinement_setups:
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
                    refinement_setup.stage2_reference_operator,
                    refinement_setup.stage3_verifier,
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
                    refinement_setup.stage2_reference_operator,
                    refinement_setup.stage3_verifier,
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
                    refinement_setup.stage2_reference_operator,
                    refinement_setup.stage3_verifier,
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
                    refinement_setup.stage2_reference_operator,
                    refinement_setup.stage3_verifier,
                    query_budget,
                    0,
                    CENTROID_HEAD_POSTING_CAP,
                )
            )


def append_planner_summaries_for_public_dataset(
    mut summaries: List[PlannerBenchmarkSummary],
    read backend: ExactCpuBackend,
    read dataset: PublicBenchmarkDataset,
    read options: PlannerBenchmarkRunOptions,
) raises:
    append_planner_summaries_for_dataset(
        summaries,
        backend,
        dataset.stored_task,
        ensure_public_benchmark_dataset_collection_mirror(
            dataset,
            options.collection_suffix,
            0,
            0,
            CENTROID_HEAD_POSTING_CAP,
            include_frontier_gem_graph=True,
        ),
        options.full_candidate_window_sweep,
        options.candidate_window_limit,
        planner_benchmark_refinement_setups(
            options.include_clause_text_stage3_when_text_available,
            public_benchmark_dataset_has_loaded_text_corpus(dataset),
        ),
    )


def selected_public_benchmark_dataset_keys(
    read options: PlannerBenchmarkRunOptions
) -> List[String]:
    var dataset_keys = List[String]()
    if options.include_scifact:
        dataset_keys.append("scifact_real_subset")
    if options.include_fiqa:
        dataset_keys.append("fiqa_real_subset")
    if options.include_limit_small:
        dataset_keys.append("limit_small")
    if options.include_bright_stackoverflow:
        dataset_keys.append("bright_stackoverflow_real_subset")
    if options.include_legal_rag_bench:
        dataset_keys.append("legal_rag_bench_real_subset")
    if options.include_lemb_narrativeqa:
        dataset_keys.append("lemb_narrativeqa_real_subset")
    if options.include_r2med_biology:
        dataset_keys.append("r2med_biology_real_subset")
    if options.include_browsecomp_plus:
        dataset_keys.append("browsecomp_plus_real_subset")
    if options.include_browsecomp_plus_gold:
        dataset_keys.append("browsecomp_plus_gold")
    return dataset_keys^


def planner_benchmark_summaries_for_options(
    read options: PlannerBenchmarkRunOptions
) raises -> List[PlannerBenchmarkSummary]:
    var backend = ExactCpuBackend()
    var summaries = List[PlannerBenchmarkSummary]()
    for dataset_key in selected_public_benchmark_dataset_keys(options):
        append_planner_summaries_for_public_dataset(
            summaries,
            backend,
            load_public_benchmark_dataset(
                dataset_key,
                load_text_corpus=(
                    options.include_clause_text_stage3_when_text_available
                ),
            ),
            options,
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
