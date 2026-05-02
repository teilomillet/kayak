from std.collections import List

from kayak.numeric import ScoreScalar


# Result types for benchmark-only TAC candidate-generation profiling.
# Production candidate generation stays in `tachiom_tac_dim128.mojo`.
struct TachiomTacCandidateAccumulation(Movable):
    var document_scores: List[ScoreScalar]
    var posting_visit_count: Int
    var selected_centroid_count: Int
    var touched_document_count: Int
    var seen_document_count: Int
    var score_checksum: ScoreScalar

    def __init__(
        out self,
        var document_scores: List[ScoreScalar],
        posting_visit_count: Int,
        selected_centroid_count: Int,
        touched_document_count: Int,
        seen_document_count: Int,
        score_checksum: ScoreScalar,
    ):
        self.document_scores = document_scores^
        self.posting_visit_count = posting_visit_count
        self.selected_centroid_count = selected_centroid_count
        self.touched_document_count = touched_document_count
        self.seen_document_count = seen_document_count
        self.score_checksum = score_checksum


struct TachiomTacCandidateGenerationProfile(Movable):
    var full_candidate_mean_seconds: Float64
    var full_search_mean_seconds: Float64
    var centroid_scoring_mean_seconds: Float64
    var centroid_selection_mean_seconds: Float64
    var posting_accumulation_mean_seconds: Float64
    var final_topk_mean_seconds: Float64
    var exact_rerank_mean_seconds: Float64
    var query_vector_count: Int
    var document_count: Int
    var document_vector_count: Int
    var total_document_vector_count: Int
    var centroid_count: Int
    var centroids_per_query_vector: Int
    var candidate_k: Int
    var final_k: Int
    var selected_centroid_count: Int
    var posting_visit_count: Int
    var touched_document_count: Int
    var seen_document_count: Int
    var output_candidate_count: Int
    var output_final_count: Int
    var measurement_iterations: Int
    var sink_value: Float64

    def __init__(
        out self,
        full_candidate_mean_seconds: Float64,
        full_search_mean_seconds: Float64,
        centroid_scoring_mean_seconds: Float64,
        centroid_selection_mean_seconds: Float64,
        posting_accumulation_mean_seconds: Float64,
        final_topk_mean_seconds: Float64,
        exact_rerank_mean_seconds: Float64,
        query_vector_count: Int,
        document_count: Int,
        document_vector_count: Int,
        total_document_vector_count: Int,
        centroid_count: Int,
        centroids_per_query_vector: Int,
        candidate_k: Int,
        final_k: Int,
        selected_centroid_count: Int,
        posting_visit_count: Int,
        touched_document_count: Int,
        seen_document_count: Int,
        output_candidate_count: Int,
        output_final_count: Int,
        measurement_iterations: Int,
        sink_value: Float64,
    ):
        self.full_candidate_mean_seconds = full_candidate_mean_seconds
        self.full_search_mean_seconds = full_search_mean_seconds
        self.centroid_scoring_mean_seconds = centroid_scoring_mean_seconds
        self.centroid_selection_mean_seconds = centroid_selection_mean_seconds
        self.posting_accumulation_mean_seconds = (
            posting_accumulation_mean_seconds
        )
        self.final_topk_mean_seconds = final_topk_mean_seconds
        self.exact_rerank_mean_seconds = exact_rerank_mean_seconds
        self.query_vector_count = query_vector_count
        self.document_count = document_count
        self.document_vector_count = document_vector_count
        self.total_document_vector_count = total_document_vector_count
        self.centroid_count = centroid_count
        self.centroids_per_query_vector = centroids_per_query_vector
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.selected_centroid_count = selected_centroid_count
        self.posting_visit_count = posting_visit_count
        self.touched_document_count = touched_document_count
        self.seen_document_count = seen_document_count
        self.output_candidate_count = output_candidate_count
        self.output_final_count = output_final_count
        self.measurement_iterations = measurement_iterations
        self.sink_value = sink_value


struct TachiomTacHnswPqQueryProfile(Movable):
    var full_search_mean_seconds: Float64
    var candidate_generation_mean_seconds: Float64
    var hnsw_traversal_mean_seconds: Float64
    var candidate_score_accumulation_mean_seconds: Float64
    var candidate_topk_mean_seconds: Float64
    var candidate_pruning_mean_seconds: Float64
    var residual_score_table_mean_seconds: Float64
    var rerank_scoring_mean_seconds: Float64
    var rerank_topk_mean_seconds: Float64
    var query_vector_count: Int
    var document_count: Int
    var document_vector_count: Int
    var total_document_vector_count: Int
    var centroid_count: Int
    var centroids_per_query_vector: Int
    var candidate_k: Int
    var final_k: Int
    var ef_search: Int
    var candidate_pruning_alpha: Float64
    var selected_centroid_count: Int
    var posting_visit_count: Int
    var touched_document_count: Int
    var seen_document_count: Int
    var ranked_candidate_count: Int
    var output_candidate_count: Int
    var output_final_count: Int
    var measurement_iterations: Int
    var sink_value: Float64

    def __init__(
        out self,
        full_search_mean_seconds: Float64,
        candidate_generation_mean_seconds: Float64,
        hnsw_traversal_mean_seconds: Float64,
        candidate_score_accumulation_mean_seconds: Float64,
        candidate_topk_mean_seconds: Float64,
        candidate_pruning_mean_seconds: Float64,
        residual_score_table_mean_seconds: Float64,
        rerank_scoring_mean_seconds: Float64,
        rerank_topk_mean_seconds: Float64,
        query_vector_count: Int,
        document_count: Int,
        document_vector_count: Int,
        total_document_vector_count: Int,
        centroid_count: Int,
        centroids_per_query_vector: Int,
        candidate_k: Int,
        final_k: Int,
        ef_search: Int,
        candidate_pruning_alpha: Float64,
        selected_centroid_count: Int,
        posting_visit_count: Int,
        touched_document_count: Int,
        seen_document_count: Int,
        ranked_candidate_count: Int,
        output_candidate_count: Int,
        output_final_count: Int,
        measurement_iterations: Int,
        sink_value: Float64,
    ):
        self.full_search_mean_seconds = full_search_mean_seconds
        self.candidate_generation_mean_seconds = (
            candidate_generation_mean_seconds
        )
        self.hnsw_traversal_mean_seconds = hnsw_traversal_mean_seconds
        self.candidate_score_accumulation_mean_seconds = (
            candidate_score_accumulation_mean_seconds
        )
        self.candidate_topk_mean_seconds = candidate_topk_mean_seconds
        self.candidate_pruning_mean_seconds = candidate_pruning_mean_seconds
        self.residual_score_table_mean_seconds = (
            residual_score_table_mean_seconds
        )
        self.rerank_scoring_mean_seconds = rerank_scoring_mean_seconds
        self.rerank_topk_mean_seconds = rerank_topk_mean_seconds
        self.query_vector_count = query_vector_count
        self.document_count = document_count
        self.document_vector_count = document_vector_count
        self.total_document_vector_count = total_document_vector_count
        self.centroid_count = centroid_count
        self.centroids_per_query_vector = centroids_per_query_vector
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.ef_search = ef_search
        self.candidate_pruning_alpha = candidate_pruning_alpha
        self.selected_centroid_count = selected_centroid_count
        self.posting_visit_count = posting_visit_count
        self.touched_document_count = touched_document_count
        self.seen_document_count = seen_document_count
        self.ranked_candidate_count = ranked_candidate_count
        self.output_candidate_count = output_candidate_count
        self.output_final_count = output_final_count
        self.measurement_iterations = measurement_iterations
        self.sink_value = sink_value
