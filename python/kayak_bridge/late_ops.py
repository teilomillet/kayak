"""Exposes the Python-facing late-interaction constructors and local ops."""

from __future__ import annotations

from .api_types import (
    DocIdsInput,
    DocumentMatricesInput,
    DocumentTokenIdsInput,
    DocOffsetsInput,
    DocTextsInput,
    QueryBatchInput,
    TokenIdValuesInput,
    TokenMatrixInput,
    TokenValuesInput,
)
from .backend_info import BackendInfo, available_backends, backend_info
from .candidate_generator import (
    CandidateGenerator,
    document_proxy_candidate_generator,
    exact_full_scan_candidate_generator,
)
from .candidate_stage import CandidateStageResult
from .reference_scoring_semantics import (
    ReferenceScoringSemantics,
    exact_late_interaction_reference_scoring_semantics,
)
from .stage2_reference_operator import (
    Stage2ReferenceOperator,
    exact_late_interaction_stage2_reference_operator,
    noop_topk_stage2_reference_operator,
)
from .stage3_verifier_operator import (
    Stage3VerifierOperator,
    clause_text_stage3_verifier_operator,
    none_stage3_verifier_operator,
)
from .dtypes import FLAT_DIM128_VECTOR_DIM
from .late_query_batch import LateQueryBatch
from .late_documents import LateDocuments
from .late_index import LateIndex
from .late_query import LateQuery
from .late_scores import LateScores, SearchHit
from .batch_dispatch import maxsim_scores_batch
from .layouts import MOJO_EXACT_CPU_BACKEND, NUMPY_REFERENCE_BACKEND
from .backend_dispatch import maxsim_scores
from .mojo_exact_cpu import load_module as load_mojo_exact_cpu_module
from .prepared_index_cache import prepared_packed_index_object
from .mojo_payloads import query_payload
from .plaid_approx import (
    PlaidApproxConfig,
    PlaidApproxIndex,
)
from .planned_search import SearchPlanResult
from .search_plan import (
    SearchPlan,
    document_proxy_search_plan,
    exact_full_scan_clause_text_search_plan,
    exact_full_scan_search_plan,
)
from .search_stage_profile import SearchStageProfile


def query(
    token_vectors: TokenMatrixInput,
    *,
    text: str | None = None,
) -> LateQuery:
    """Build one ``LateQuery`` from a 2D token-vector matrix."""
    return LateQuery.from_vectors(token_vectors, text=text)


def query_batch(token_vectors: QueryBatchInput) -> LateQueryBatch:
    """Build one ``LateQueryBatch`` from a sequence of query matrices."""
    return LateQueryBatch.from_inputs(token_vectors)


def flat_query_dim128(
    token_values: TokenValuesInput,
    *,
    text: str | None = None,
) -> LateQuery:
    """Build one flat 128-dimensional query layout directly from values."""
    return LateQuery.from_flat_values(
        token_values,
        vector_dim=FLAT_DIM128_VECTOR_DIM,
        text=text,
    )


def documents(
    doc_ids: DocIdsInput,
    token_vectors: DocumentMatricesInput,
    *,
    texts: DocTextsInput | None = None,
    token_ids: DocumentTokenIdsInput | None = None,
) -> LateDocuments:
    """Build ``LateDocuments`` from document ids and token-level vectors."""
    return LateDocuments.from_inputs(
        doc_ids,
        token_vectors,
        texts=texts,
        token_ids=token_ids,
    )


def packed_index(
    doc_ids: DocIdsInput,
    doc_offsets: DocOffsetsInput,
    token_vectors: TokenMatrixInput,
    *,
    doc_texts: DocTextsInput | None = None,
    token_ids: TokenIdValuesInput | None = None,
) -> LateIndex:
    """Build one packed ``LateIndex`` directly from packed layout fields."""
    return LateIndex.from_packed(
        doc_ids,
        doc_offsets,
        token_vectors,
        doc_texts=doc_texts,
        token_ids=token_ids,
    )


def hybrid_flat_dim128_index(
    doc_ids: DocIdsInput,
    doc_offsets: DocOffsetsInput,
    token_values: TokenValuesInput,
    *,
    doc_texts: DocTextsInput | None = None,
    token_ids: TokenIdValuesInput | None = None,
) -> LateIndex:
    """Build one ``hybrid_flat_dim128`` index directly from flat values."""
    return LateIndex.from_hybrid_flat_dim128(
        doc_ids,
        doc_offsets,
        token_values,
        doc_texts=doc_texts,
        token_ids=token_ids,
    )


def maxsim(
    late_query: LateQuery,
    late_index: LateIndex,
    *,
    backend: str = NUMPY_REFERENCE_BACKEND,
) -> LateScores:
    """Return exact scores for every document in one index."""
    return maxsim_scores(late_query, late_index, backend=backend)


def maxsim_batch(
    late_query_batch: LateQueryBatch,
    late_index: LateIndex,
    *,
    backend: str = NUMPY_REFERENCE_BACKEND,
) -> tuple[LateScores, ...]:
    """Return exact score vectors for every query in one batch."""
    return maxsim_scores_batch(late_query_batch, late_index, backend=backend)


def search(
    late_query: LateQuery,
    late_index: LateIndex,
    *,
    k: int,
    backend: str = NUMPY_REFERENCE_BACKEND,
    approximation: PlaidApproxConfig | None = None,
) -> tuple[SearchHit, ...]:
    """Return top-k hits for one query against one index.

    Passing ``approximation=PlaidApproxConfig(...)`` opts into the Mojo
    sampled-centroid approximation lane with exact rerank.
    """
    if approximation is not None:
        prepared_index = prepare_plaid_approx_index(
            late_index,
            config=approximation,
            final_k=k,
        )
        return prepared_index.search(late_query, final_k=k)

    if (
        backend == MOJO_EXACT_CPU_BACKEND
        and late_index.layout == "packed"
        and late_query.layout == "nested"
    ):
        module = load_mojo_exact_cpu_module()
        prepared_index = prepared_packed_index_object(late_index, module=module)
        raw_hits = module.search_prepared_packed(
            query_payload(late_query),
            k,
            prepared_index,
        )
        return tuple(
            SearchHit(doc_id=str(raw_hit[0]), score=float(raw_hit[1]))
            for raw_hit in raw_hits
        )

    return maxsim(late_query, late_index, backend=backend).topk(k)


def search_batch(
    late_query_batch: LateQueryBatch,
    late_index: LateIndex,
    *,
    k: int,
    backend: str = NUMPY_REFERENCE_BACKEND,
    approximation: PlaidApproxConfig | None = None,
) -> tuple[tuple[SearchHit, ...], ...]:
    """Return top-k hits for every query in one batch.

    Passing ``approximation=PlaidApproxConfig(...)`` opts into the Mojo
    sampled-centroid approximation lane with exact rerank.
    """
    if approximation is not None:
        prepared_index = prepare_plaid_approx_index(
            late_index,
            config=approximation,
            final_k=k,
        )
        return prepared_index.search_batch(late_query_batch, final_k=k)

    if (
        backend == MOJO_EXACT_CPU_BACKEND
        and late_index.layout == "packed"
        and all(query.layout == "nested" for query in late_query_batch.queries)
    ):
        module = load_mojo_exact_cpu_module()
        prepared_index = prepared_packed_index_object(
            late_index, module=module
        )
        raw_hits_by_query = module.search_prepared_packed_batch(
            [query_payload(query) for query in late_query_batch.queries],
            k,
            prepared_index,
        )
        return tuple(
            tuple(
                SearchHit(doc_id=str(raw_hit[0]), score=float(raw_hit[1]))
                for raw_hit in raw_hits
            )
            for raw_hits in raw_hits_by_query
        )

    return tuple(
        scores.topk(k)
        for scores in maxsim_batch(late_query_batch, late_index, backend=backend)
    )


def prepare_plaid_approx_index(
    late_index: LateIndex,
    *,
    config: PlaidApproxConfig,
    final_k: int,
) -> PlaidApproxIndex:
    """Prepare one index for explicit Mojo PLAID-style approximate search."""
    return PlaidApproxIndex.from_late_index(
        late_index,
        config=config,
        final_k=final_k,
    )


def generate_candidates(
    late_query: LateQuery,
    late_index: LateIndex,
    generator: CandidateGenerator,
    *,
    k: int,
    backend: str = NUMPY_REFERENCE_BACKEND,
) -> CandidateStageResult:
    """Run the explicit stage-1 candidate generator for one query and index."""
    return late_index.generate_candidates(
        late_query,
        generator,
        k=k,
        backend=backend,
    )


def search_with_plan(
    late_query: LateQuery,
    late_index: LateIndex,
    plan: SearchPlan,
    *,
    backend: str = NUMPY_REFERENCE_BACKEND,
) -> SearchPlanResult:
    """Run one explicit staged retrieval plan against one index."""
    return late_index.search_with_plan(
        late_query,
        plan=plan,
        backend=backend,
    )
