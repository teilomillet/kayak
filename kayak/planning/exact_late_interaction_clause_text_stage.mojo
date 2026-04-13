# Hybrid stage-2 refinement over exact late interaction plus clause text.

from std.collections import List

from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import EncodedQuery
from kayak.runtime import ExactScoringBackend
from kayak.search import SearchHit
from kayak.verifier import (
    default_clause_text_rerank_config,
    rerank_hits_clause_text,
)

from .clause_text_stage import candidate_text_corpus_for_hits
from .collection_hit import CollectionHit
from .exact_stage import materialize_candidate_index
from .stage_artifact_materialization import StageArtifactMaterialization
from .stage2_operator import (
    STAGE2_REQUIRED_ARTIFACT_DOCUMENT_TEXT,
    STAGE2_REQUIRED_ARTIFACT_LATE_INTERACTION,
)
from .stage2_result import Stage2Result


def exact_late_interaction_clause_text_rerank_candidates_for_plan[
    Backend: ExactScoringBackend
](
    read backend: Backend,
    read query: EncodedQuery,
    query_text: String,
    read snapshot: ResolvedCollectionSnapshot,
    read hits: List[CollectionHit],
    final_k: Int,
) raises -> Stage2Result:
    if len(hits) == 0:
        return Stage2Result([], 0, 0, 0, 0, 0)

    if query_text.byte_length() == 0:
        raise Error(
            "exact_late_interaction_clause_text stage-2 requires non-empty query_text"
        )

    var materialized = materialize_candidate_index(snapshot, hits)
    var candidate_texts = candidate_text_corpus_for_hits(snapshot, hits)
    var exact_scores = backend.score_all(query, materialized.index)
    var exact_hits = List[SearchHit]()

    for document_index in range(len(exact_scores)):
        exact_hits.append(
            SearchHit(
                materialized.index.doc_ids[document_index].copy(),
                exact_scores[document_index],
            )
        )

    var reranked_hits = rerank_hits_clause_text(
        query_text,
        exact_hits^,
        candidate_texts.corpus,
        final_k,
        default_clause_text_rerank_config(),
    )
    var final_hits = List[CollectionHit]()

    for reranked_hit in reranked_hits:
        var matched = False
        for hit in hits:
            if hit.doc_id != reranked_hit.doc_id:
                continue

            final_hits.append(
                CollectionHit(
                    hit.segment_id.copy(),
                    hit.doc_id.copy(),
                    reranked_hit.score,
                )
            )
            matched = True
            break

        if not matched:
            raise Error(
                "hybrid exact clause-text doc_id not found in candidate window: "
                + reranked_hit.doc_id
            )

    var total_token_count = (
        materialized.token_count + candidate_texts.token_count
    )
    var total_byte_size = materialized.byte_size + candidate_texts.byte_size

    return Stage2Result(
        final_hits^,
        [
            StageArtifactMaterialization(
                STAGE2_REQUIRED_ARTIFACT_LATE_INTERACTION,
                materialized.segment_count,
                materialized.index.document_count,
                materialized.token_count,
                materialized.vector_count,
                materialized.byte_size,
            ),
            StageArtifactMaterialization(
                STAGE2_REQUIRED_ARTIFACT_DOCUMENT_TEXT,
                materialized.segment_count,
                materialized.index.document_count,
                candidate_texts.token_count,
                0,
                candidate_texts.byte_size,
            ),
        ],
        materialized.segment_count,
        materialized.index.document_count,
        total_token_count,
        materialized.vector_count,
        total_byte_size,
    )
