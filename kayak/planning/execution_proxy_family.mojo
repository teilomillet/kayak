from std.collections import List

from kayak.collections import (
    SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY,
    SEARCH_ARTIFACT_FAMILY_LATENT_PROXY,
    ResolvedCollectionSnapshot,
    loaded_search_artifact_stored_document_proxy_index,
    loaded_search_artifact_stored_latent_proxy_index,
    loaded_segment_has_search_artifact,
    loaded_segment_search_artifact,
)
from kayak.contracts import EncodedQuery
from kayak.filters import (
    FilterExpression,
    filter_expression_requires_document_metadata,
    match_all_filter,
)
from kayak.index import build_query_proxy_vector
from kayak.runtime import ExactScoringBackend
from kayak.scoring.dot import dot_product

from .candidate_set import CandidateSet
from .collection_hit import CollectionHit
from .filter_allowlist import (
    document_filter_allowlist_artifact_byte_size_for_segment,
    document_filter_allowlist_for_segment,
)
from .filter_application_profile import (
    filter_application_profile_for_effective_filter,
)
from .filter_scope import (
    effective_filter_expression_for_collection,
    effective_filter_expression_for_segment,
)
from .latent_proxy_primitives import (
    project_query_with_latent_proxy,
    segment_hits_for_projected_latent_query,
)
from .search_plan import SearchPlan
from .topk import insert_descending_collection_hit


def candidate_generation_for_proxy_family[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    read filter_expression: FilterExpression = match_all_filter(),
) raises -> CandidateSet:
    _ = backend

    var hits = List[CollectionHit]()
    var vector_count = 0
    var byte_size = 0
    var effective_collection_filter = effective_filter_expression_for_collection(
        snapshot.collection,
        filter_expression,
    )
    var filter_input_document_count = 0
    var filter_matching_document_count = 0
    var filter_artifact_byte_size = 0
    var uses_document_filter_index = False

    for segment_index in range(len(snapshot.segments)):
        var required_family = SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY
        var missing_family_label = "document_proxy"
        if plan.candidate_generator.kind == "latent_proxy":
            required_family = SEARCH_ARTIFACT_FAMILY_LATENT_PROXY
            missing_family_label = "latent_proxy"
        if not loaded_segment_has_search_artifact(
            snapshot.segments[segment_index],
            required_family,
        ):
            raise Error(
                missing_family_label
                + " stage-1 requires a matching sidecar for every segment"
            )
        var allowed_flags = List[Int]()
        var effective_filter = effective_filter_expression_for_segment(
            snapshot.collection,
            snapshot.segments[segment_index],
            filter_expression,
        )
        if plan.candidate_generator.kind == "latent_proxy":
            var stored_proxy = loaded_search_artifact_stored_latent_proxy_index(
                loaded_segment_search_artifact(
                    snapshot.segments[segment_index],
                    SEARCH_ARTIFACT_FAMILY_LATENT_PROXY,
                )
            )
            vector_count += stored_proxy.index.document_count
            filter_input_document_count += stored_proxy.index.document_count
            byte_size += stored_proxy.artifact_byte_size
            var matching_document_count = stored_proxy.index.document_count
            if not effective_filter.is_match_all():
                var filter_artifact_bytes = (
                    document_filter_allowlist_artifact_byte_size_for_segment(
                        snapshot.segments[segment_index],
                        effective_filter,
                    )
                )
                filter_artifact_byte_size += filter_artifact_bytes
                byte_size += filter_artifact_bytes
                uses_document_filter_index = (
                    uses_document_filter_index
                    or filter_expression_requires_document_metadata(
                        effective_filter
                    )
                )
                var allowlist = document_filter_allowlist_for_segment(
                    snapshot.segments[segment_index],
                    effective_filter,
                )
                matching_document_count = allowlist.matching_document_count
                allowed_flags = allowlist.flags.copy()
            filter_matching_document_count += matching_document_count
            if matching_document_count == 0:
                continue
            var query_proxy = project_query_with_latent_proxy(
                query, stored_proxy.query_projection
            )
            var segment_hits = segment_hits_for_projected_latent_query(
                query_proxy,
                snapshot.segments[segment_index].manifest.segment_id.value.copy(),
                segment_index,
                stored_proxy.index,
                plan.candidate_budget.candidate_k,
                allowed_flags,
            )
            for hit in segment_hits:
                insert_descending_collection_hit(
                    hits,
                    hit.copy(),
                    plan.candidate_budget.candidate_k,
                )
            continue

        var stored_proxy = loaded_search_artifact_stored_document_proxy_index(
            loaded_segment_search_artifact(
                snapshot.segments[segment_index],
                SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY,
            )
        )
        vector_count += (
            stored_proxy.index.document_count
            * stored_proxy.proxy_vector_count_per_document
        )
        filter_input_document_count += stored_proxy.index.document_count
        byte_size += stored_proxy.artifact_byte_size
        var matching_document_count = stored_proxy.index.document_count
        if not effective_filter.is_match_all():
            var filter_artifact_bytes = (
                document_filter_allowlist_artifact_byte_size_for_segment(
                    snapshot.segments[segment_index],
                    effective_filter,
                )
            )
            filter_artifact_byte_size += filter_artifact_bytes
            byte_size += filter_artifact_bytes
            uses_document_filter_index = (
                uses_document_filter_index
                or filter_expression_requires_document_metadata(
                    effective_filter
                )
            )
            var allowlist = document_filter_allowlist_for_segment(
                snapshot.segments[segment_index],
                effective_filter,
            )
            matching_document_count = allowlist.matching_document_count
            allowed_flags = allowlist.flags.copy()

        filter_matching_document_count += matching_document_count
        if matching_document_count == 0:
            continue

        var query_proxy = build_query_proxy_vector(query, 0)
        for document_index in range(stored_proxy.index.document_count):
            if len(allowed_flags) != 0 and allowed_flags[document_index] == 0:
                continue
            var doc_id = stored_proxy.index.doc_ids[document_index]
            insert_descending_collection_hit(
                hits,
                CollectionHit(
                    snapshot.segments[segment_index].manifest.segment_id.value.copy(),
                    doc_id.copy(),
                    dot_product(
                        query_proxy,
                        stored_proxy.index.proxy_vectors[document_index],
                    ),
                    segment_index,
                    document_index,
                ),
                plan.candidate_budget.candidate_k,
            )

    var candidate_set = CandidateSet(
        plan.candidate_generator,
        hits^,
        snapshot.snapshot.stats.segment_count,
        snapshot.snapshot.stats.document_count,
        0,
        vector_count,
        byte_size,
    )
    candidate_set.filter_application_profile = (
        filter_application_profile_for_effective_filter(
            filter_expression,
            effective_collection_filter,
            filter_input_document_count,
            filter_matching_document_count,
            filter_artifact_byte_size,
            uses_document_filter_index,
        )
    )
    return candidate_set^
