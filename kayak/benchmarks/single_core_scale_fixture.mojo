from std.collections import List

from kayak.contracts import EncodedDocument, EncodedQuery
from kayak.eval import JudgedQuery, JudgedTask
from kayak.index import pack_documents
from kayak.numeric import VECTOR_SCALAR_NAME
from kayak.storage import StoredJudgedTask, StoredPackedIndex

from .proxy_vectors import concept_vectors


struct SingleCoreScaleProfile(Copyable):
    var family: String
    var slice_name: String
    var why: String
    var document_count: Int
    var query_count: Int
    var query_vector_count: Int
    var document_vector_count: Int
    var vector_dim: Int
    var final_k: Int
    var candidate_k: Int
    var centroid_head_posting_cap: Int

    def __init__(
        out self,
        var family: String,
        var slice_name: String,
        var why: String,
        document_count: Int,
        query_count: Int,
        query_vector_count: Int,
        document_vector_count: Int,
        vector_dim: Int,
        final_k: Int,
        candidate_k: Int,
        centroid_head_posting_cap: Int,
    ) raises:
        if document_count <= 0:
            raise Error("single-core scale profile document_count must be positive")

        if query_count <= 0:
            raise Error("single-core scale profile query_count must be positive")

        if document_count % query_count != 0:
            raise Error(
                "single-core scale profile document_count must divide evenly across queries"
            )

        if query_vector_count <= 0:
            raise Error(
                "single-core scale profile query_vector_count must be positive"
            )

        if document_vector_count < query_vector_count:
            raise Error(
                "single-core scale profile document_vector_count must cover all query vectors"
            )

        if vector_dim < query_count * document_vector_count:
            raise Error(
                "single-core scale profile vector_dim must cover grouped concept slots"
            )

        if final_k <= 0:
            raise Error("single-core scale profile final_k must be positive")

        if candidate_k < final_k:
            raise Error(
                "single-core scale profile candidate_k must be at least final_k"
            )

        if centroid_head_posting_cap < 0:
            raise Error(
                "single-core scale profile centroid_head_posting_cap must be non-negative"
            )

        self.family = family^
        self.slice_name = slice_name^
        self.why = why^
        self.document_count = document_count
        self.query_count = query_count
        self.query_vector_count = query_vector_count
        self.document_vector_count = document_vector_count
        self.vector_dim = vector_dim
        self.final_k = final_k
        self.candidate_k = candidate_k
        self.centroid_head_posting_cap = centroid_head_posting_cap


struct SingleCoreScaleFixture(Copyable):
    var stored_task: StoredJudgedTask
    var stored_index: StoredPackedIndex

    def __init__(
        out self,
        stored_task: StoredJudgedTask,
        stored_index: StoredPackedIndex,
    ):
        self.stored_task = stored_task.copy()
        self.stored_index = stored_index.copy()


comptime RELEVANT_DOCS_PER_QUERY = 2


def default_single_core_scale_profiles() raises -> List[SingleCoreScaleProfile]:
    var profiles = List[SingleCoreScaleProfile]()
    profiles.append(
        SingleCoreScaleProfile(
            "synthetic_scale",
            "docs_64",
            "Single-core control with fixed query/document vector counts and 64 documents.",
            64,
            8,
            4,
            8,
            64,
            2,
            16,
            16,
        )
    )
    profiles.append(
        SingleCoreScaleProfile(
            "synthetic_scale",
            "docs_256",
            "Single-core sweep that increases only corpus size to 256 documents.",
            256,
            8,
            4,
            8,
            64,
            2,
            16,
            16,
        )
    )
    profiles.append(
        SingleCoreScaleProfile(
            "synthetic_scale",
            "docs_1024",
            "Single-core sweep that increases only corpus size to 1024 documents.",
            1024,
            8,
            4,
            8,
            64,
            2,
            16,
            16,
        )
    )
    profiles.append(
        SingleCoreScaleProfile(
            "synthetic_scale",
            "docs_4096",
            "Single-core sweep that increases only corpus size to 4096 documents.",
            4096,
            8,
            4,
            8,
            64,
            2,
            16,
            16,
        )
    )
    return profiles^


def documents_per_query_group(read profile: SingleCoreScaleProfile) -> Int:
    return profile.document_count // profile.query_count


def document_id_for_group(query_index: Int, document_index: Int) -> String:
    return "doc-q" + String(query_index) + "-" + String(document_index)


def group_concept_offset(read profile: SingleCoreScaleProfile, query_index: Int) -> Int:
    return query_index * profile.document_vector_count


def core_concepts_for_group(
    read profile: SingleCoreScaleProfile, query_index: Int
) -> List[Int]:
    var concepts = List[Int]()
    var offset = group_concept_offset(profile, query_index)

    for concept_index in range(profile.query_vector_count):
        concepts.append(offset + concept_index)

    return concepts^


def filler_concepts_for_group(
    read profile: SingleCoreScaleProfile, query_index: Int
) -> List[Int]:
    var concepts = List[Int]()
    var offset = group_concept_offset(profile, query_index)

    for concept_index in range(
        profile.query_vector_count, profile.document_vector_count
    ):
        concepts.append(offset + concept_index)

    return concepts^


def outsider_concept_for_group(
    read profile: SingleCoreScaleProfile, query_index: Int, document_index: Int
) -> Int:
    var next_group = query_index + 1
    if next_group == profile.query_count:
        next_group = 0

    return (
        group_concept_offset(profile, next_group)
        + (document_index % profile.document_vector_count)
    )


def document_is_relevant(document_index: Int) -> Bool:
    return document_index < RELEVANT_DOCS_PER_QUERY


def document_concepts_for_group(
    read profile: SingleCoreScaleProfile, query_index: Int, document_index: Int
) -> List[Int]:
    var concepts = List[Int]()
    var core_concepts = core_concepts_for_group(profile, query_index)
    var filler_concepts = filler_concepts_for_group(profile, query_index)

    if document_is_relevant(document_index):
        for concept in core_concepts:
            concepts.append(concept)
    else:
        var omitted_core_index = (
            document_index - RELEVANT_DOCS_PER_QUERY
        ) % profile.query_vector_count
        for concept_index in range(len(core_concepts)):
            if concept_index == omitted_core_index:
                continue
            concepts.append(core_concepts[concept_index])
        concepts.append(
            outsider_concept_for_group(profile, query_index, document_index)
        )

    for concept in filler_concepts:
        concepts.append(concept)

    return concepts^


def relevant_doc_ids_for_group(
    read profile: SingleCoreScaleProfile, query_index: Int
) raises -> List[String]:
    if documents_per_query_group(profile) < RELEVANT_DOCS_PER_QUERY:
        raise Error(
            "single-core scale profile must allocate enough documents for the relevant set"
        )

    var doc_ids = List[String]()
    for document_index in range(RELEVANT_DOCS_PER_QUERY):
        doc_ids.append(document_id_for_group(query_index, document_index))

    return doc_ids^


def make_single_core_scale_fixture(
    read profile: SingleCoreScaleProfile
) raises -> SingleCoreScaleFixture:
    if documents_per_query_group(profile) < RELEVANT_DOCS_PER_QUERY:
        raise Error(
            "single-core scale profile requires at least two documents per query group"
        )

    var documents = List[EncodedDocument]()
    var queries = List[JudgedQuery]()

    for query_index in range(profile.query_count):
        for document_index in range(documents_per_query_group(profile)):
            documents.append(
                EncodedDocument(
                    document_id_for_group(query_index, document_index),
                    concept_vectors(
                        profile.vector_dim,
                        document_concepts_for_group(
                            profile, query_index, document_index
                        ),
                    ),
                )
            )

        queries.append(
            JudgedQuery(
                "query-" + String(query_index),
                "Synthetic fixed-shape scale query for grouped late-interaction benchmarking.",
                EncodedQuery(
                    concept_vectors(
                        profile.vector_dim,
                        core_concepts_for_group(profile, query_index),
                    )
                ),
                relevant_doc_ids_for_group(profile, query_index),
            )
        )

    var dataset_id = "synthetic://single-core-scale/" + profile.slice_name
    var model_name = "synthetic-grouped-concepts"
    var packed_index = pack_documents(documents)

    return SingleCoreScaleFixture(
        StoredJudgedTask(
            dataset_id.copy(),
            model_name.copy(),
            VECTOR_SCALAR_NAME,
            JudgedTask(
                profile.family.copy(),
                profile.slice_name.copy(),
                profile.why.copy(),
                "success",
                profile.final_k,
                profile.query_vector_count,
                profile.document_vector_count,
                profile.vector_dim,
                documents^,
                queries^,
            ),
        ),
        StoredPackedIndex(
            dataset_id,
            model_name,
            VECTOR_SCALAR_NAME,
            packed_index^,
        ),
    )
