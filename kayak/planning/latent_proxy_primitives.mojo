from std.collections import List

from kayak.contracts import EncodedQuery
from kayak.index import (
    LatentProxyIndex,
    LatentQueryProjection,
    build_query_latent_proxy_vector,
)
from kayak.index.latent_proxy import (
    build_query_latent_proxy_vector_multi_block,
    build_query_latent_proxy_vector_single_block,
)
from kayak.numeric import ScoreScalar, VectorScalar, zero_score_scalar
from kayak.scoring.dot import dot_product

from .collection_hit import CollectionHit
from .topk import insert_descending_collection_hit


# Owns latent-proxy stage-1 primitives. It owns query projection into one
# document-aligned proxy vector and full-segment proxy scans. It does not own
# segment loading, filter policy, or exact reranking.


struct ProjectedLatentQuery(Copyable):
    var proxy_vector: List[VectorScalar]
    var vector_count: Int
    var vector_dim: Int

    def __init__(out self, var proxy_vector: List[VectorScalar]) raises:
        if len(proxy_vector) == 0:
            raise Error("projected latent query proxy_vector must not be empty")
        self.proxy_vector = proxy_vector^
        self.vector_count = 1
        self.vector_dim = len(self.proxy_vector)


def require_projected_latent_query_matches_index(
    read projected_query: ProjectedLatentQuery,
    read proxy_index: LatentProxyIndex,
) raises:
    if projected_query.vector_dim != proxy_index.vector_dim:
        raise Error(
            "projected latent query vector_dim must match latent proxy index vector_dim"
        )


def require_proxy_scan_flags_match_index(
    read allowed_flags: List[Int], read proxy_index: LatentProxyIndex
) raises:
    if len(allowed_flags) != 0 and len(allowed_flags) != proxy_index.document_count:
        raise Error(
            "latent proxy allowed_flags length must match index document_count"
        )


def project_query_with_latent_proxy(
    read query: EncodedQuery,
    read projection: LatentQueryProjection,
) raises -> ProjectedLatentQuery:
    return ProjectedLatentQuery(
        build_query_latent_proxy_vector(query, projection)
    )


def project_query_with_latent_proxy_generic(
    read query: EncodedQuery,
    read projection: LatentQueryProjection,
) raises -> ProjectedLatentQuery:
    return ProjectedLatentQuery(
        build_query_latent_proxy_vector_multi_block(query, projection)
    )


def project_query_with_latent_proxy_single_block(
    read query: EncodedQuery,
    read projection: LatentQueryProjection,
) raises -> ProjectedLatentQuery:
    return ProjectedLatentQuery(
        build_query_latent_proxy_vector_single_block(query, projection)
    )


def score_projected_latent_query_against_document(
    read projected_query: ProjectedLatentQuery,
    read proxy_index: LatentProxyIndex,
    document_index: Int,
) raises -> ScoreScalar:
    require_projected_latent_query_matches_index(projected_query, proxy_index)
    if document_index < 0 or document_index >= proxy_index.document_count:
        raise Error("latent proxy document_index is out of range")
    return dot_product(
        projected_query.proxy_vector,
        proxy_index.proxy_vectors[document_index],
    )


def sum_projected_latent_query_scores_against_index(
    read projected_query: ProjectedLatentQuery,
    read proxy_index: LatentProxyIndex,
) raises -> ScoreScalar:
    require_projected_latent_query_matches_index(projected_query, proxy_index)
    var total = zero_score_scalar()
    for document_index in range(proxy_index.document_count):
        total += score_projected_latent_query_against_document(
            projected_query,
            proxy_index,
            document_index,
        )
    return total


def segment_hits_for_projected_latent_query(
    read projected_query: ProjectedLatentQuery,
    segment_id: String,
    segment_index: Int,
    read proxy_index: LatentProxyIndex,
    candidate_k: Int,
    read allowed_flags: List[Int] = [],
) raises -> List[CollectionHit]:
    if candidate_k <= 0:
        return List[CollectionHit]()
    require_projected_latent_query_matches_index(projected_query, proxy_index)
    require_proxy_scan_flags_match_index(allowed_flags, proxy_index)

    var hits = List[CollectionHit]()
    for document_index in range(proxy_index.document_count):
        if len(allowed_flags) != 0 and allowed_flags[document_index] == 0:
            continue
        insert_descending_collection_hit(
            hits,
            CollectionHit(
                segment_id.copy(),
                proxy_index.doc_ids[document_index].copy(),
                score_projected_latent_query_against_document(
                    projected_query,
                    proxy_index,
                    document_index,
                ),
                segment_index,
                document_index,
            ),
            candidate_k,
        )
    return hits^


def segment_hits_for_latent_proxy(
    read query: EncodedQuery,
    segment_id: String,
    segment_index: Int,
    read projection: LatentQueryProjection,
    read proxy_index: LatentProxyIndex,
    candidate_k: Int,
    read allowed_flags: List[Int] = [],
) raises -> List[CollectionHit]:
    return segment_hits_for_projected_latent_query(
        project_query_with_latent_proxy(query, projection),
        segment_id,
        segment_index,
        proxy_index,
        candidate_k,
        allowed_flags,
    )
