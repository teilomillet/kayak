from .contracts import EncodedDocument, EncodedQuery
from .eval import (
    JudgedQuery,
    JudgedTask,
    TaskEvaluation,
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
    load_fiqa_real_subset,
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
    exact_scores_for_hybrid_flat_index_dim128,
)
from .search import SearchHit, search_exact, search_exact_hybrid_flat_dim128
from .storage import (
    FiqaRealSubsetCache,
    HybridFlatDim128CacheEntry,
    ScifactRealSubsetCache,
    StoredHybridFlatDim128Index,
    StoredJudgedTask,
    StoredPackedIndex,
    build_stored_hybrid_flat_dim128_index,
    ensure_fiqa_real_subset_cache,
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
