from std.collections import List

from kayak.numeric import ScoreScalar


# Result types for benchmark-only PLAID i8 candidate-generation profiling.
# This file owns data contracts only; candidate-generation logic stays in the
# profiling module so measurement fields remain easy to audit.
struct PlaidI8CandidateAccumulation(Movable):
    var document_scores: List[ScoreScalar]
    var posting_visit_count: Int
    var selected_centroid_count: Int
    var touched_document_count: Int
    var score_checksum: ScoreScalar

    def __init__(
        out self,
        var document_scores: List[ScoreScalar],
        posting_visit_count: Int,
        selected_centroid_count: Int,
        touched_document_count: Int,
        score_checksum: ScoreScalar,
    ):
        self.document_scores = document_scores^
        self.posting_visit_count = posting_visit_count
        self.selected_centroid_count = selected_centroid_count
        self.touched_document_count = touched_document_count
        self.score_checksum = score_checksum


struct PlaidI8CandidateGenerationProfile(Movable):
    var full_candidate_mean_seconds: Float64
    var workspace_full_candidate_mean_seconds: Float64
    var workspace_candidate_position_agreement: Float64
    var unordered_candidate_mean_seconds: Float64
    var unordered_candidate_set_agreement: Float64
    var centroid_scoring_mean_seconds: Float64
    var centroid_selection_mean_seconds: Float64
    var posting_accumulation_mean_seconds: Float64
    var final_topk_mean_seconds: Float64
    var unordered_final_topk_mean_seconds: Float64
    var query_vector_count: Int
    var document_count: Int
    var document_vector_count: Int
    var total_document_vector_count: Int
    var centroid_count: Int
    var centroids_per_query_vector: Int
    var candidate_k: Int
    var selected_centroid_count: Int
    var posting_visit_count: Int
    var touched_document_count: Int
    var output_candidate_count: Int
    var measurement_iterations: Int
    var sink_value: Float64

    def __init__(
        out self,
        full_candidate_mean_seconds: Float64,
        workspace_full_candidate_mean_seconds: Float64,
        workspace_candidate_position_agreement: Float64,
        unordered_candidate_mean_seconds: Float64,
        unordered_candidate_set_agreement: Float64,
        centroid_scoring_mean_seconds: Float64,
        centroid_selection_mean_seconds: Float64,
        posting_accumulation_mean_seconds: Float64,
        final_topk_mean_seconds: Float64,
        unordered_final_topk_mean_seconds: Float64,
        query_vector_count: Int,
        document_count: Int,
        document_vector_count: Int,
        total_document_vector_count: Int,
        centroid_count: Int,
        centroids_per_query_vector: Int,
        candidate_k: Int,
        selected_centroid_count: Int,
        posting_visit_count: Int,
        touched_document_count: Int,
        output_candidate_count: Int,
        measurement_iterations: Int,
        sink_value: Float64,
    ):
        self.full_candidate_mean_seconds = full_candidate_mean_seconds
        self.workspace_full_candidate_mean_seconds = (
            workspace_full_candidate_mean_seconds
        )
        self.workspace_candidate_position_agreement = (
            workspace_candidate_position_agreement
        )
        self.unordered_candidate_mean_seconds = unordered_candidate_mean_seconds
        self.unordered_candidate_set_agreement = (
            unordered_candidate_set_agreement
        )
        self.centroid_scoring_mean_seconds = centroid_scoring_mean_seconds
        self.centroid_selection_mean_seconds = centroid_selection_mean_seconds
        self.posting_accumulation_mean_seconds = (
            posting_accumulation_mean_seconds
        )
        self.final_topk_mean_seconds = final_topk_mean_seconds
        self.unordered_final_topk_mean_seconds = (
            unordered_final_topk_mean_seconds
        )
        self.query_vector_count = query_vector_count
        self.document_count = document_count
        self.document_vector_count = document_vector_count
        self.total_document_vector_count = total_document_vector_count
        self.centroid_count = centroid_count
        self.centroids_per_query_vector = centroids_per_query_vector
        self.candidate_k = candidate_k
        self.selected_centroid_count = selected_centroid_count
        self.posting_visit_count = posting_visit_count
        self.touched_document_count = touched_document_count
        self.output_candidate_count = output_candidate_count
        self.measurement_iterations = measurement_iterations
        self.sink_value = sink_value
