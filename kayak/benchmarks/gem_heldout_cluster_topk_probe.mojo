# Narrow search-cluster-top-k probe for held-out GEM ablations on the first
# real synthetic hard-recall profile.
#
# Owns:
# - a fixed-profile query-time sweep over cluster_top_k_per_query_token
# - persistent JSON output over a fixed beam width and candidate windows
#
# Does not own:
# - generic held-out ablation surfaces
# - graph-construction benchmarking

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.runtime import ExactCpuBackend
from kayak.scoring import ExactScoringConfig

from .gem_frontier_config import frontier_gem_graph_build_config
from .gem_heldout_ablation_json import (
    GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF,
    GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_RELEVANT_COVERAGE,
    GEM_HELDOUT_VARIANT_BASELINE,
    build_gem_heldout_query_split,
    default_gem_heldout_training_query_count,
    gem_heldout_ablation_variant_spec,
)
from .gem_heldout_querytime_probe import (
    GemHeldoutQuerytimeProbeSummary,
    build_gem_heldout_querytime_probe_summary,
    gem_heldout_querytime_probe_collection_id,
    gem_heldout_querytime_probe_collection_root,
    gem_heldout_querytime_probe_summaries_json,
)
from .synthetic_hard_recall_fixture import (
    default_synthetic_hard_recall_profiles,
    make_synthetic_hard_recall_fixture,
)


def gem_heldout_cluster_topk_probe_backend() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_parallel_scoring = False
    return ExactCpuBackend(config^)


def gem_heldout_cluster_topk_probe_candidate_ks() -> List[Int]:
    return [32, 64]


def gem_heldout_cluster_topk_probe_search_cluster_top_ks() -> List[Int]:
    return [1, 2, 4, 8]


def gem_heldout_cluster_topk_probe_beam_width() -> Int:
    return 32


def print_gem_heldout_cluster_topk_probe_summary_line(
    read summary: GemHeldoutQuerytimeProbeSummary
):
    print(
        "cluster_topk_probe profile=",
        summary.slice_name,
        " variant=",
        summary.variant_kind,
        " candidate_k=",
        summary.candidate_k,
        " cluster_top_k=",
        summary.search_cluster_top_k_per_query_token,
        " beam=",
        summary.search_beam_width,
        " recall=",
        summary.mean_recall_at_k,
        " reachable=",
        summary.evaluation_positive_reachable_rate,
    )


def synthetic_hard_recall_gem_heldout_cluster_topk_probe_summaries(
) raises -> List[GemHeldoutQuerytimeProbeSummary]:
    var profiles = default_synthetic_hard_recall_profiles()
    if len(profiles) == 0:
        raise Error(
            "held-out GEM cluster_top_k probe requires at least one synthetic profile"
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
            GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF,
            base_config,
            split.training_pairs,
        ),
        gem_heldout_ablation_variant_spec(
            GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_RELEVANT_COVERAGE,
            base_config,
            split.training_pairs,
        ),
    ]

    var summaries = List[GemHeldoutQuerytimeProbeSummary]()
    var backend = gem_heldout_cluster_topk_probe_backend()
    var search_beam_width = gem_heldout_cluster_topk_probe_beam_width()
    for variant in variants:
        for candidate_k in gem_heldout_cluster_topk_probe_candidate_ks():
            for search_cluster_top_k in (
                gem_heldout_cluster_topk_probe_search_cluster_top_ks()
            ):
                var summary = build_gem_heldout_querytime_probe_summary(
                    backend,
                    fixture.stored_index,
                    split,
                    gem_heldout_querytime_probe_collection_root(
                        fixture.stored_task.task.slice_name,
                        variant.variant_kind,
                    ),
                    gem_heldout_querytime_probe_collection_id(
                        fixture.stored_task.task.slice_name,
                        variant.variant_kind,
                    ),
                    variant.copy(),
                    candidate_k,
                    search_cluster_top_k,
                    search_beam_width,
                )
                print_gem_heldout_cluster_topk_probe_summary_line(summary)
                summaries.append(summary^)
    return summaries^


def write_synthetic_hard_recall_gem_heldout_cluster_topk_probe() raises:
    var output_dir = Path(".cache/kayak")
    makedirs(output_dir, exist_ok=True)
    (output_dir / "synthetic_hard_recall_gem_heldout_cluster_topk_probe.json").write_text(
        gem_heldout_querytime_probe_summaries_json(
            synthetic_hard_recall_gem_heldout_cluster_topk_probe_summaries()
        )
    )
