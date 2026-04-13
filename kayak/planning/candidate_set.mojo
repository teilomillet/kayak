# Candidate artifact emitted by the stage-1 generator.

from std.collections import List

from .candidate_generator import CandidateGenerator
from .collection_hit import CollectionHit
from .filter_application_profile import (
    FilterApplicationProfile,
    identity_filter_application_profile,
)
from .graph_search_counters import GraphSearchCounters
from .stage1_capabilities import (
    stage1_capabilities_for_candidate_generator_kind,
)


struct CandidateSet(Copyable):
    var generator_kind: String
    var generator_family: String
    var interaction_semantics: String
    var alignment_granularity: String
    var score_kind: String
    var hits: List[CollectionHit]
    var segment_count: Int
    var document_count: Int
    var token_count: Int
    var vector_count: Int
    var byte_size: Int
    var filter_application_profile: FilterApplicationProfile
    var tracks_graph_search: Bool
    var graph_search_counters: GraphSearchCounters

    def __init__(
        out self,
        read generator: CandidateGenerator,
        var hits: List[CollectionHit],
        segment_count: Int,
        document_count: Int,
        token_count: Int,
        vector_count: Int,
        byte_size: Int,
    ) raises:
        self = CandidateSet(
            generator.kind.copy(),
            generator.family.copy(),
            generator.interaction_semantics.copy(),
            generator.alignment_granularity.copy(),
            generator.score_kind.copy(),
            hits^,
            segment_count,
            document_count,
            token_count,
            vector_count,
            byte_size,
        )

    def __init__(
        out self,
        var generator_kind: String,
        var hits: List[CollectionHit],
        segment_count: Int,
        document_count: Int,
        token_count: Int,
        vector_count: Int,
        byte_size: Int,
    ) raises:
        var capabilities = stage1_capabilities_for_candidate_generator_kind(
            generator_kind
        )
        self = CandidateSet(
            generator_kind^,
            capabilities.generator_family.copy(),
            capabilities.interaction_semantics.copy(),
            capabilities.alignment_granularity.copy(),
            capabilities.score_kind.copy(),
            hits^,
            segment_count,
            document_count,
            token_count,
            vector_count,
            byte_size,
        )

    def __init__(
        out self,
        var generator_kind: String,
        var generator_family: String,
        var interaction_semantics: String,
        var alignment_granularity: String,
        var score_kind: String,
        var hits: List[CollectionHit],
        segment_count: Int,
        document_count: Int,
        token_count: Int,
        vector_count: Int,
        byte_size: Int,
    ) raises:
        if segment_count < 0:
            raise Error("candidate set segment_count must be non-negative")

        if document_count < 0:
            raise Error("candidate set document_count must be non-negative")

        if token_count < 0:
            raise Error("candidate set token_count must be non-negative")

        if vector_count < 0:
            raise Error("candidate set vector_count must be non-negative")

        if byte_size < 0:
            raise Error("candidate set byte_size must be non-negative")

        self.generator_kind = generator_kind^
        self.generator_family = generator_family^
        self.interaction_semantics = interaction_semantics^
        self.alignment_granularity = alignment_granularity^
        self.score_kind = score_kind^
        self.hits = hits^
        self.segment_count = segment_count
        self.document_count = document_count
        self.token_count = token_count
        self.vector_count = vector_count
        self.byte_size = byte_size
        self.filter_application_profile = identity_filter_application_profile(
            document_count
        )
        self.tracks_graph_search = False
        self.graph_search_counters = GraphSearchCounters()

    def __init__(
        out self,
        read generator: CandidateGenerator,
        var hits: List[CollectionHit],
        segment_count: Int,
        document_count: Int,
        token_count: Int,
        vector_count: Int,
        byte_size: Int,
        graph_search_counters: GraphSearchCounters,
    ) raises:
        self = CandidateSet(
            generator,
            hits^,
            segment_count,
            document_count,
            token_count,
            vector_count,
            byte_size,
        )
        self.tracks_graph_search = True
        self.graph_search_counters = graph_search_counters.copy()

    def __init__(
        out self,
        var generator_kind: String,
        var hits: List[CollectionHit],
        segment_count: Int,
        document_count: Int,
        token_count: Int,
        vector_count: Int,
        byte_size: Int,
        graph_search_counters: GraphSearchCounters,
    ) raises:
        self = CandidateSet(
            generator_kind^,
            hits^,
            segment_count,
            document_count,
            token_count,
            vector_count,
            byte_size,
        )
        self.tracks_graph_search = True
        self.graph_search_counters = graph_search_counters.copy()
