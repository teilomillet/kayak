from .contracts import (
    EncodedDocument,
    EncodedQuery,
    FlatQueryDim128,
    build_flat_query_dim128,
)
from .eval import (
    JudgedQuery,
    JudgedTask,
    QueryEvaluation,
    TaskEvaluation,
    evaluate_query_hits,
    evaluate_task,
    evaluate_task_with_verifier,
)
from .index import (
    HybridFlatDim128Index,
    PackedIndex,
    build_hybrid_flat_dim128_index,
    pack_documents,
)
from .interop import (
    load_browsecomp_plus_gold_real_subset,
    load_browsecomp_plus_real_subset,
    load_fiqa_real_subset,
    load_limit_small_real_subset,
    load_mock_python_task,
    load_scifact_real_subset,
)
from .numeric import (
    METRIC_SCALAR_NAME,
    SCORE_SCALAR_NAME,
    STORAGE_FORMAT_VERSION,
    VECTOR_SCALAR_NAME,
    MetricScalar,
    ScoreScalar,
    VectorScalar,
)
from .runtime import ExactCpuBackend
from .scoring import ExactScoringConfig
from .scoring import (
    exact_score_for_hybrid_flat_document_dim128,
    exact_score_for_hybrid_flat_document_dim128_with_flat_query,
    exact_scores_for_hybrid_flat_index_dim128,
    exact_scores_for_hybrid_flat_index_dim128_with_flat_query,
)
from .search import (
    SearchHit,
    search_exact,
    search_exact_all,
    search_exact_hybrid_flat_dim128,
    search_exact_hybrid_flat_dim128_with_flat_query,
)
from .storage import (
    BrowsecompPlusGoldRealSubsetCache,
    BrowsecompPlusRealSubsetCache,
    FiqaRealSubsetCache,
    HybridFlatDim128CacheEntry,
    LimitSmallRealSubsetCache,
    ScifactRealSubsetCache,
    StoredHybridFlatDim128Index,
    StoredJudgedTask,
    StoredPackedIndex,
    build_stored_hybrid_flat_dim128_index,
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_browsecomp_plus_real_subset_cache,
    ensure_fiqa_real_subset_cache,
    ensure_limit_small_real_subset_cache,
    ensure_scifact_real_subset_cache,
    ensure_stored_hybrid_flat_dim128_index,
    hybrid_flat_dim128_index_exists,
    load_stored_judged_task,
    load_stored_hybrid_flat_dim128_index,
    load_stored_packed_index,
    save_stored_judged_task,
    save_stored_hybrid_flat_dim128_index,
    save_stored_packed_index,
)
from .verifier import (
    VerifierReranker,
    effective_candidate_k,
    exact_late_interaction_verifier,
    no_verifier,
    rerank_hits_with_verifier,
    search_exact_with_verifier,
)
