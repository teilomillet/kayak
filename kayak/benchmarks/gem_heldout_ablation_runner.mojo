from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.runtime import ExactCpuBackend
from kayak.scoring import ExactScoringConfig

from .candidate_window_json import standard_candidate_window_sizes
from .gem_frontier_config import frontier_gem_graph_build_config
from .gem_heldout_ablation_json import (
    GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_RELEVANT_COVERAGE,
    GemHeldoutAblationSummary,
    build_gem_heldout_ablation_summary,
    gem_heldout_ablation_variant_spec,
    build_gem_heldout_query_split,
    default_gem_heldout_training_query_count,
    gem_heldout_ablation_summaries_json,
    standard_gem_heldout_ablation_variant_specs,
)
from .synthetic_hard_recall_fixture import (
    SyntheticHardRecallProfile,
    default_synthetic_hard_recall_profiles,
    make_synthetic_hard_recall_fixture,
    smoke_synthetic_hard_recall_profile,
)


struct GemHeldoutAblationRunOptions(Copyable):
    var output_filename: String
    var collection_root_stem: String
    var variant_kinds: List[String]
    var use_smoke_profile: Bool
    var profile_limit: Int
    var full_candidate_window_sweep: Bool
    var candidate_window_limit: Int
    var emit_progress_lines: Bool

    def __init__(
        out self,
        var output_filename: String,
        var collection_root_stem: String,
        var variant_kinds: List[String],
        use_smoke_profile: Bool,
        profile_limit: Int,
        full_candidate_window_sweep: Bool,
        candidate_window_limit: Int,
        emit_progress_lines: Bool,
    ) raises:
        if profile_limit < 0:
            raise Error("gem held-out ablation profile_limit must be non-negative")
        if candidate_window_limit < 0:
            raise Error(
                "gem held-out ablation candidate_window_limit must be non-negative"
            )

        self.output_filename = output_filename^
        self.collection_root_stem = collection_root_stem^
        self.variant_kinds = variant_kinds^
        self.use_smoke_profile = use_smoke_profile
        self.profile_limit = profile_limit
        self.full_candidate_window_sweep = full_candidate_window_sweep
        self.candidate_window_limit = candidate_window_limit
        self.emit_progress_lines = emit_progress_lines


def default_gem_heldout_ablation_run_options(
) raises -> GemHeldoutAblationRunOptions:
    return GemHeldoutAblationRunOptions(
        "synthetic_hard_recall_gem_heldout_ablation.json",
        "synthetic_hard_recall_gem_heldout",
        List[String](),
        False,
        0,
        True,
        0,
        True,
    )


def smoke_gem_heldout_ablation_run_options(
) raises -> GemHeldoutAblationRunOptions:
    return GemHeldoutAblationRunOptions(
        "synthetic_hard_recall_gem_heldout_ablation_smoke_small.json",
        "synthetic_hard_recall_gem_heldout_smoke_small",
        List[String](),
        True,
        1,
        True,
        2,
        True,
    )


def focused_gem_heldout_ablation_run_options(
) raises -> GemHeldoutAblationRunOptions:
    return GemHeldoutAblationRunOptions(
        "synthetic_hard_recall_gem_heldout_ablation_focused.json",
        "synthetic_hard_recall_gem_heldout_focused",
        List[String](),
        False,
        1,
        True,
        2,
        True,
    )


def adaptive_probe_gem_heldout_ablation_run_options(
) raises -> GemHeldoutAblationRunOptions:
    return GemHeldoutAblationRunOptions(
        "synthetic_hard_recall_gem_heldout_ablation_adaptive_probe.json",
        "synthetic_hard_recall_gem_heldout_adaptive_probe",
        [
            "baseline",
            "adaptive_cutoff",
            "adaptive_cutoff_relevant_coverage",
        ],
        False,
        1,
        True,
        2,
        True,
    )


def smoke_adaptive_probe_gem_heldout_ablation_run_options(
) raises -> GemHeldoutAblationRunOptions:
    return GemHeldoutAblationRunOptions(
        "synthetic_hard_recall_gem_heldout_ablation_adaptive_probe_smoke.json",
        "synthetic_hard_recall_gem_heldout_adaptive_probe_smoke",
        [
            "baseline",
            "adaptive_cutoff",
            "adaptive_cutoff_relevant_coverage",
        ],
        True,
        1,
        True,
        2,
        True,
    )


def single_core_backend() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_parallel_scoring = False
    return ExactCpuBackend(config^)


def documents_per_evaluation_query_group(
    document_count: Int, evaluation_query_count: Int
) -> Int:
    if evaluation_query_count <= 0:
        return document_count
    return document_count // evaluation_query_count


def max_frontier_candidate_k(
    document_count: Int, evaluation_query_count: Int, baseline_k: Int
) -> Int:
    var max_candidate_k = documents_per_evaluation_query_group(
        document_count,
        evaluation_query_count,
    )
    if baseline_k > max_candidate_k:
        max_candidate_k = baseline_k
    if max_candidate_k > document_count:
        max_candidate_k = document_count
    return max_candidate_k


def append_unique_int(mut values: List[Int], value: Int):
    for existing in values:
        if existing == value:
            return
    values.append(value)


def gem_heldout_candidate_window_sizes(
    final_k: Int,
    max_candidate_k: Int,
    full_sweep: Bool,
    candidate_window_limit: Int,
) -> List[Int]:
    var all_sizes = standard_candidate_window_sizes(final_k, max_candidate_k)
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


def print_summary_line(read summary: GemHeldoutAblationSummary):
    print(
        "Mean: ",
        summary.mean_search_seconds,
        " profile=",
        summary.slice_name,
        " variant=",
        summary.variant_kind,
        " candidate_k=",
        summary.candidate_k,
        " heldout_queries=",
        summary.evaluation_query_count,
        " recall=",
        summary.mean_candidate_recall_at_final_k,
        " shortcuts=",
        summary.artifact_shortcut_edge_count,
    )


def selected_synthetic_hard_recall_profile_count(
    read options: GemHeldoutAblationRunOptions
) raises -> Int:
    if options.use_smoke_profile:
        return 1

    var profile_count = len(default_synthetic_hard_recall_profiles())
    if options.profile_limit > 0 and options.profile_limit < profile_count:
        return options.profile_limit
    return profile_count


def selected_synthetic_hard_recall_profiles(
    read options: GemHeldoutAblationRunOptions
) raises -> List[SyntheticHardRecallProfile]:
    if options.use_smoke_profile:
        return [smoke_synthetic_hard_recall_profile()]
    return default_synthetic_hard_recall_profiles()


def gem_heldout_collection_root(
    read options: GemHeldoutAblationRunOptions,
    slice_name: String,
    variant_kind: String,
) -> Path:
    return Path(
        ".cache/kayak/"
        + options.collection_root_stem
        + "_"
        + slice_name
        + "_"
        + variant_kind
    )


def gem_heldout_collection_id(
    read options: GemHeldoutAblationRunOptions,
    slice_name: String,
    variant_kind: String,
) -> String:
    return options.collection_root_stem + "_" + slice_name + "_" + variant_kind


def gem_heldout_variant_enabled(
    read options: GemHeldoutAblationRunOptions, variant_kind: String
) -> Bool:
    if len(options.variant_kinds) == 0:
        return True

    for selected_variant_kind in options.variant_kinds:
        if selected_variant_kind == variant_kind:
            return True
    return False


def synthetic_hard_recall_gem_heldout_ablation_summaries_for_options(
    read options: GemHeldoutAblationRunOptions
) raises -> List[GemHeldoutAblationSummary]:
    var summaries = List[GemHeldoutAblationSummary]()
    var backend = single_core_backend()
    var profile_count = selected_synthetic_hard_recall_profile_count(options)
    var profiles = selected_synthetic_hard_recall_profiles(options)

    for profile_index in range(profile_count):
        var profile = profiles[profile_index].copy()
        var fixture = make_synthetic_hard_recall_fixture(profile)
        var training_query_count = default_gem_heldout_training_query_count(
            fixture.stored_task
        )
        var split = build_gem_heldout_query_split(
            fixture.stored_task,
            training_query_count,
        )
        var variant_specs = standard_gem_heldout_ablation_variant_specs(
            fixture.stored_index,
            fixture.stored_task.task.nominal_query_vector_count,
            split,
        )
        if gem_heldout_variant_enabled(
            options,
            GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_RELEVANT_COVERAGE,
        ):
            variant_specs.append(
                gem_heldout_ablation_variant_spec(
                    GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_RELEVANT_COVERAGE,
                    frontier_gem_graph_build_config(
                        fixture.stored_index,
                        fixture.stored_task.task.nominal_query_vector_count,
                    ),
                    split.training_pairs,
                )
            )
        var max_candidate_k = max_frontier_candidate_k(
            fixture.stored_index.index.document_count,
            len(split.evaluation_task.task.queries),
            split.evaluation_task.task.k,
        )

        for variant in variant_specs:
            if not gem_heldout_variant_enabled(options, variant.variant_kind):
                continue
            for candidate_k in gem_heldout_candidate_window_sizes(
                split.evaluation_task.task.k,
                max_candidate_k,
                options.full_candidate_window_sweep,
                options.candidate_window_limit,
            ):
                var summary = build_gem_heldout_ablation_summary(
                    backend,
                    fixture.stored_index,
                    split,
                    gem_heldout_collection_root(
                        options,
                        profile.slice_name,
                        variant.variant_kind,
                    ),
                    gem_heldout_collection_id(
                        options,
                        profile.slice_name,
                        variant.variant_kind,
                    ),
                    variant,
                    candidate_k,
                )
                summaries.append(summary.copy())
                if options.emit_progress_lines:
                    print_summary_line(summary)
    return summaries^


def write_synthetic_hard_recall_gem_heldout_ablation(
    read options: GemHeldoutAblationRunOptions
) raises:
    var summaries = synthetic_hard_recall_gem_heldout_ablation_summaries_for_options(
        options
    )
    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / options.output_filename
    output_path.write_text(gem_heldout_ablation_summaries_json(summaries))
    print("wrote ", String(output_path))
