# Narrow shortcut probe for held-out GEM variants on the first real synthetic
# hard-recall profile.
#
# Owns:
# - a fixed-profile comparison between baseline and shortcut-enabled variants
# - persistent JSON output over fixed search settings
#
# Does not own:
# - generic held-out ablation surfaces
# - scalar graph-density sweeps

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.runtime import ExactCpuBackend
from kayak.scoring import ExactScoringConfig

from .gem_frontier_config import frontier_gem_graph_build_config
from .gem_heldout_ablation_json import (
    GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF,
    GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_SHORTCUTS,
    GEM_HELDOUT_VARIANT_BASELINE,
    GEM_HELDOUT_VARIANT_SHORTCUTS,
    build_gem_heldout_query_split,
    default_gem_heldout_training_query_count,
    gem_heldout_ablation_variant_spec,
)
from .gem_heldout_querytime_probe import (
    GemHeldoutQuerytimeProbeSummary,
    build_gem_heldout_querytime_probe_summary,
    gem_heldout_querytime_probe_summaries_json,
)
from .synthetic_hard_recall_fixture import (
    default_synthetic_hard_recall_profiles,
    make_synthetic_hard_recall_fixture,
)


def gem_heldout_shortcut_probe_backend() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_parallel_scoring = False
    return ExactCpuBackend(config^)


def gem_heldout_shortcut_probe_candidate_ks() -> List[Int]:
    return [32, 64]


def gem_heldout_shortcut_probe_search_cluster_top_k() -> Int:
    return 2


def gem_heldout_shortcut_probe_beam_width() -> Int:
    return 32


def gem_heldout_shortcut_probe_collection_root(
    slice_name: String, variant_kind: String
) -> Path:
    return Path(
        ".cache/kayak/synthetic_hard_recall_gem_heldout_shortcut_probe/"
        + slice_name
        + "/"
        + variant_kind
    )


def gem_heldout_shortcut_probe_collection_id(
    slice_name: String, variant_kind: String
) -> String:
    return (
        "synthetic-hard-recall-gem-heldout-shortcut-probe-"
        + slice_name
        + "-"
        + variant_kind
    )


def print_gem_heldout_shortcut_probe_summary_line(
    read summary: GemHeldoutQuerytimeProbeSummary
):
    print(
        "shortcut_probe profile=",
        summary.slice_name,
        " variant=",
        summary.variant_kind,
        " shortcut_edges=",
        summary.artifact_shortcut_edge_count,
        " candidate_k=",
        summary.candidate_k,
        " recall=",
        summary.mean_recall_at_k,
        " reachable=",
        summary.evaluation_positive_reachable_rate,
    )


def synthetic_hard_recall_gem_heldout_shortcut_probe_summaries(
) raises -> List[GemHeldoutQuerytimeProbeSummary]:
    var profiles = default_synthetic_hard_recall_profiles()
    if len(profiles) == 0:
        raise Error(
            "held-out GEM shortcut probe requires at least one synthetic profile"
        )

    var fixture = make_synthetic_hard_recall_fixture(profiles[0])
    var split = build_gem_heldout_query_split(
        fixture.stored_task,
        default_gem_heldout_training_query_count(fixture.stored_task),
    )
    var base_config = frontier_gem_graph_build_config(
        fixture.stored_index,
        fixture.stored_task.task.nominal_query_vector_count,
    )
    var variants = [
        gem_heldout_ablation_variant_spec(
            GEM_HELDOUT_VARIANT_BASELINE,
            base_config,
            split.training_pairs,
        ),
        gem_heldout_ablation_variant_spec(
            GEM_HELDOUT_VARIANT_SHORTCUTS,
            base_config,
            split.training_pairs,
        ),
        gem_heldout_ablation_variant_spec(
            GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF,
            base_config,
            split.training_pairs,
        ),
        gem_heldout_ablation_variant_spec(
            GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_SHORTCUTS,
            base_config,
            split.training_pairs,
        ),
    ]

    var summaries = List[GemHeldoutQuerytimeProbeSummary]()
    var backend = gem_heldout_shortcut_probe_backend()
    for variant in variants:
        for candidate_k in gem_heldout_shortcut_probe_candidate_ks():
            var summary = build_gem_heldout_querytime_probe_summary(
                backend,
                fixture.stored_index,
                split,
                gem_heldout_shortcut_probe_collection_root(
                    fixture.stored_task.task.slice_name,
                    variant.variant_kind,
                ),
                gem_heldout_shortcut_probe_collection_id(
                    fixture.stored_task.task.slice_name,
                    variant.variant_kind,
                ),
                variant.copy(),
                candidate_k,
                gem_heldout_shortcut_probe_search_cluster_top_k(),
                gem_heldout_shortcut_probe_beam_width(),
            )
            print_gem_heldout_shortcut_probe_summary_line(summary)
            summaries.append(summary^)
    return summaries^


def write_synthetic_hard_recall_gem_heldout_shortcut_probe() raises:
    var output_dir = Path(".cache/kayak")
    makedirs(output_dir, exist_ok=True)
    (output_dir / "synthetic_hard_recall_gem_heldout_shortcut_probe.json").write_text(
        gem_heldout_querytime_probe_summaries_json(
            synthetic_hard_recall_gem_heldout_shortcut_probe_summaries()
        )
    )
