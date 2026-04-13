# Counts that make search-stage behavior inspectable.

from .graph_search_counters import GraphSearchCounters
from .score_histogram import ScoreHistogram


struct SearchStageProfile(Copyable):
    var stage_name: String
    var input_hit_count: Int
    var output_hit_count: Int
    var segment_count: Int
    var document_count: Int
    var token_count: Int
    var vector_count: Int
    var byte_size: Int
    var graph_search_counters: GraphSearchCounters
    var score_histogram: ScoreHistogram

    def __init__(
        out self,
        var stage_name: String,
        input_hit_count: Int,
        output_hit_count: Int,
        segment_count: Int,
        document_count: Int,
        token_count: Int,
        vector_count: Int,
        byte_size: Int,
        score_histogram: ScoreHistogram,
    ) raises:
        if input_hit_count < 0:
            raise Error("search stage input_hit_count must be non-negative")

        if output_hit_count < 0:
            raise Error("search stage output_hit_count must be non-negative")

        if segment_count < 0:
            raise Error("search stage segment_count must be non-negative")

        if document_count < 0:
            raise Error("search stage document_count must be non-negative")

        if token_count < 0:
            raise Error("search stage token_count must be non-negative")

        if vector_count < 0:
            raise Error("search stage vector_count must be non-negative")

        if byte_size < 0:
            raise Error("search stage byte_size must be non-negative")

        self.stage_name = stage_name^
        self.input_hit_count = input_hit_count
        self.output_hit_count = output_hit_count
        self.segment_count = segment_count
        self.document_count = document_count
        self.token_count = token_count
        self.vector_count = vector_count
        self.byte_size = byte_size
        self.graph_search_counters = GraphSearchCounters()
        self.score_histogram = score_histogram.copy()

    def __init__(
        out self,
        var stage_name: String,
        input_hit_count: Int,
        output_hit_count: Int,
        segment_count: Int,
        document_count: Int,
        token_count: Int,
        vector_count: Int,
        byte_size: Int,
        graph_search_counters: GraphSearchCounters,
        score_histogram: ScoreHistogram,
    ) raises:
        self = SearchStageProfile(
            stage_name^,
            input_hit_count,
            output_hit_count,
            segment_count,
            document_count,
            token_count,
            vector_count,
            byte_size,
            score_histogram,
        )
        self.graph_search_counters = graph_search_counters.copy()
