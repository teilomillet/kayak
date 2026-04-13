from std.collections import List

from kayak.collections import (
    SearchArtifactBuildPolicy,
    search_artifact_build_policy_supports_required_families,
)
from kayak.collections.validation import require_non_empty_string


comptime STAGE1_INTERACTION_SEMANTICS_NONE = "none"
comptime STAGE1_INTERACTION_SEMANTICS_APPROXIMATE_LATE_INTERACTION = (
    "approximate_late_interaction"
)
comptime STAGE1_INTERACTION_SEMANTICS_EXACT_LATE_INTERACTION = (
    "exact_late_interaction"
)

comptime STAGE1_ALIGNMENT_GRANULARITY_DOCUMENT = "document"
comptime STAGE1_ALIGNMENT_GRANULARITY_DOCUMENT_TOKENS = "document_tokens"
comptime STAGE1_ALIGNMENT_GRANULARITY_CENTROID = "centroid"
comptime STAGE1_ALIGNMENT_GRANULARITY_GRAPH_NODE = "graph_node"

comptime STAGE1_SCORE_KIND_PROXY_SCORE = "proxy_score"
comptime STAGE1_SCORE_KIND_APPROXIMATE_INTERACTION_SCORE = (
    "approximate_interaction_score"
)
comptime STAGE1_SCORE_KIND_EXACT_SCORE = "exact_score"


struct Stage1Capabilities(Copyable):
    var generator_kind: String
    var generator_family: String
    var interaction_semantics: String
    var alignment_granularity: String
    var score_kind: String
    var required_search_artifact_families: List[String]
    var stage1_is_exact: Bool
    var supports_match_all_filter: Bool
    var supports_structured_filter: Bool

    def __init__(
        out self,
        var generator_kind: String,
        var generator_family: String,
        var interaction_semantics: String,
        var alignment_granularity: String,
        var score_kind: String,
        read required_search_artifact_families: List[String],
        stage1_is_exact: Bool,
        supports_match_all_filter: Bool,
        supports_structured_filter: Bool,
    ) raises:
        self.generator_kind = require_non_empty_string(
            generator_kind, "generator_kind"
        )
        self.generator_family = require_non_empty_string(
            generator_family, "generator_family"
        )
        self.interaction_semantics = require_non_empty_string(
            interaction_semantics, "interaction_semantics"
        )
        self.alignment_granularity = require_non_empty_string(
            alignment_granularity, "alignment_granularity"
        )
        self.score_kind = require_non_empty_string(score_kind, "score_kind")
        var normalized_required_families = List[String]()
        for family in required_search_artifact_families:
            var normalized = require_non_empty_string(
                family, "required_search_artifact family"
            )
            var already_present = False
            for existing in normalized_required_families:
                if existing == normalized:
                    already_present = True
                    break

            if not already_present:
                normalized_required_families.append(normalized)

        self.required_search_artifact_families = normalized_required_families^
        self.stage1_is_exact = stage1_is_exact
        self.supports_match_all_filter = supports_match_all_filter
        self.supports_structured_filter = supports_structured_filter
        if self.stage1_is_exact and not self.supports_structured_filter:
            raise Error("exact stage-1 must support structured filters")
        if (
            self.stage1_is_exact
            and self.interaction_semantics
                != STAGE1_INTERACTION_SEMANTICS_EXACT_LATE_INTERACTION
        ):
            raise Error(
                "exact stage-1 must declare exact_late_interaction semantics"
            )

    def requires_search_artifact_family(self, family: String) -> Bool:
        for required_family in self.required_search_artifact_families:
            if required_family == family:
                return True

        return False

    def single_required_search_artifact_family(self) raises -> String:
        if len(self.required_search_artifact_families) == 0:
            return ""

        if len(self.required_search_artifact_families) == 1:
            return self.required_search_artifact_families[0].copy()

        raise Error(
            "stage-1 capabilities for "
            + self.generator_kind
            + " require multiple search artifact families"
        )


def exact_stage1_capabilities() raises -> Stage1Capabilities:
    return Stage1Capabilities(
        "exact_full_scan",
        "exact",
        STAGE1_INTERACTION_SEMANTICS_EXACT_LATE_INTERACTION,
        STAGE1_ALIGNMENT_GRANULARITY_DOCUMENT_TOKENS,
        STAGE1_SCORE_KIND_EXACT_SCORE,
        [],
        True,
        True,
        True,
    )


def document_proxy_stage1_capabilities() raises -> Stage1Capabilities:
    return Stage1Capabilities(
        "document_proxy",
        "proxy",
        STAGE1_INTERACTION_SEMANTICS_NONE,
        STAGE1_ALIGNMENT_GRANULARITY_DOCUMENT,
        STAGE1_SCORE_KIND_PROXY_SCORE,
        ["document_proxy"],
        False,
        True,
        False,
    )


def centroid_heads_stage1_capabilities() raises -> Stage1Capabilities:
    return Stage1Capabilities(
        "centroid_heads",
        "centroid",
        STAGE1_INTERACTION_SEMANTICS_APPROXIMATE_LATE_INTERACTION,
        STAGE1_ALIGNMENT_GRANULARITY_CENTROID,
        STAGE1_SCORE_KIND_APPROXIMATE_INTERACTION_SCORE,
        ["centroid_heads"],
        False,
        True,
        False,
    )


def centroid_postings_stage1_capabilities(kind: String) raises -> Stage1Capabilities:
    return Stage1Capabilities(
        kind,
        "centroid",
        STAGE1_INTERACTION_SEMANTICS_APPROXIMATE_LATE_INTERACTION,
        STAGE1_ALIGNMENT_GRANULARITY_CENTROID,
        STAGE1_SCORE_KIND_APPROXIMATE_INTERACTION_SCORE,
        ["centroid_postings"],
        False,
        True,
        False,
    )


def gem_graph_stage1_capabilities() raises -> Stage1Capabilities:
    return Stage1Capabilities(
        "gem_graph",
        "graph",
        STAGE1_INTERACTION_SEMANTICS_APPROXIMATE_LATE_INTERACTION,
        STAGE1_ALIGNMENT_GRANULARITY_GRAPH_NODE,
        STAGE1_SCORE_KIND_APPROXIMATE_INTERACTION_SCORE,
        ["gem_graph"],
        False,
        True,
        False,
    )


def stage1_capabilities_for_candidate_generator_kind(
    kind: String
) raises -> Stage1Capabilities:
    if kind == "exact_full_scan":
        return exact_stage1_capabilities()
    if kind == "document_proxy":
        return document_proxy_stage1_capabilities()
    if kind == "centroid_heads":
        return centroid_heads_stage1_capabilities()
    if (
        kind == "centroid_postings"
        or kind == "centroid_postings_flat"
        or kind == "centroid_postings_head"
        or kind == "centroid_postings_head_auto"
        or kind == "centroid_postings_blockmax"
        or kind == "centroid_postings_imputed"
        or kind == "centroid_postings_imputed_flat"
    ):
        return centroid_postings_stage1_capabilities(kind)
    if kind == "gem_graph":
        return gem_graph_stage1_capabilities()

    raise Error("unknown candidate generator kind: " + kind)


def stage1_required_search_artifact_families(
    candidate_generator_kind: String
) raises -> List[String]:
    return stage1_capabilities_for_candidate_generator_kind(
        candidate_generator_kind
    ).required_search_artifact_families.copy()


def stage1_requires_search_artifact_family(
    candidate_generator_kind: String,
    family: String,
) raises -> Bool:
    return stage1_capabilities_for_candidate_generator_kind(
        candidate_generator_kind
    ).requires_search_artifact_family(family)


def stage1_single_required_search_artifact_family(
    candidate_generator_kind: String
) raises -> String:
    return stage1_capabilities_for_candidate_generator_kind(
        candidate_generator_kind
    ).single_required_search_artifact_family()


def stage1_generator_supports_match_all_filter(
    candidate_generator_kind: String
) raises -> Bool:
    return stage1_capabilities_for_candidate_generator_kind(
        candidate_generator_kind
    ).supports_match_all_filter


def stage1_generator_supports_structured_filter(
    candidate_generator_kind: String
) raises -> Bool:
    return stage1_capabilities_for_candidate_generator_kind(
        candidate_generator_kind
    ).supports_structured_filter


def stage1_generator_is_exact(candidate_generator_kind: String) raises -> Bool:
    return stage1_capabilities_for_candidate_generator_kind(
        candidate_generator_kind
    ).stage1_is_exact


def stage1_supported_by_search_artifact_build_policy(
    read policy: SearchArtifactBuildPolicy,
    candidate_generator_kind: String,
) raises -> Bool:
    return search_artifact_build_policy_supports_required_families(
        policy,
        stage1_required_search_artifact_families(candidate_generator_kind),
    )
