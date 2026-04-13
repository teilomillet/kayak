from std.collections import List
from std.pathlib import Path

from kayak.collections import (
    CollectionId,
    NamespaceId,
    SnapshotId,
    TenantId,
    ensure_one_segment_collection_mirror,
)
from kayak.interop import (
    load_browsecomp_plus_gold_real_subset_document_text_corpus,
    load_browsecomp_plus_real_subset_document_text_corpus,
)
from kayak.storage import (
    StoredJudgedTask,
    StoredPackedIndex,
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_browsecomp_plus_real_subset_cache,
    ensure_fiqa_real_subset_cache,
    ensure_limit_small_real_subset_cache,
    ensure_scifact_real_subset_cache,
)
from kayak.text import DocumentTextCorpus

from .gem_frontier_config import frontier_gem_graph_build_config


struct PublicBenchmarkDataset(Copyable):
    var dataset_key: String
    var collection_root_stem: String
    var has_text_sidecars: Bool
    var loaded_task_from_storage: Bool
    var loaded_index_from_storage: Bool
    var stored_task: StoredJudgedTask
    var stored_index: StoredPackedIndex
    var document_text_corpus: DocumentTextCorpus

    def __init__(
        out self,
        var dataset_key: String,
        var collection_root_stem: String,
        has_text_sidecars: Bool,
        loaded_task_from_storage: Bool,
        loaded_index_from_storage: Bool,
        stored_task: StoredJudgedTask,
        stored_index: StoredPackedIndex,
        document_text_corpus: DocumentTextCorpus,
    ):
        self.dataset_key = dataset_key^
        self.collection_root_stem = collection_root_stem^
        self.has_text_sidecars = has_text_sidecars
        self.loaded_task_from_storage = loaded_task_from_storage
        self.loaded_index_from_storage = loaded_index_from_storage
        self.stored_task = stored_task.copy()
        self.stored_index = stored_index.copy()
        self.document_text_corpus = document_text_corpus.copy()


def empty_document_text_corpus() raises -> DocumentTextCorpus:
    return DocumentTextCorpus(List[String](), List[String]())


def public_benchmark_dataset_has_text_sidecars(
    read dataset: PublicBenchmarkDataset
) -> Bool:
    return dataset.has_text_sidecars


def public_benchmark_dataset_has_loaded_text_corpus(
    read dataset: PublicBenchmarkDataset
) -> Bool:
    return len(dataset.document_text_corpus.doc_ids) != 0


def public_benchmark_dataset_collection_root(
    read dataset: PublicBenchmarkDataset, collection_suffix: String
) -> Path:
    return Path(
        ".cache/kayak/" + dataset.collection_root_stem + "_" + collection_suffix
    )


def require_public_benchmark_dataset_loaded_text_corpus(
    read dataset: PublicBenchmarkDataset
) raises -> DocumentTextCorpus:
    if public_benchmark_dataset_has_loaded_text_corpus(dataset):
        return dataset.document_text_corpus.copy()

    if public_benchmark_dataset_has_text_sidecars(dataset):
        raise Error(
            "public benchmark dataset "
            + dataset.dataset_key
            + " supports text sidecars, but no text corpus was loaded"
        )

    raise Error(
        "public benchmark dataset "
        + dataset.dataset_key
        + " does not provide text sidecars"
    )


def require_public_benchmark_text_corpus_matches_index(
    dataset_key: String,
    read stored_index: StoredPackedIndex,
    read document_text_corpus: DocumentTextCorpus,
) raises:
    if len(document_text_corpus.doc_ids) == 0:
        return

    if len(document_text_corpus.doc_ids) != stored_index.index.document_count:
        raise Error(
            "text corpus for "
            + dataset_key
            + " has "
            + String(len(document_text_corpus.doc_ids))
            + " docs, but index has "
            + String(stored_index.index.document_count)
        )

    for index in range(stored_index.index.document_count):
        if document_text_corpus.doc_ids[index] != stored_index.index.doc_ids[index]:
            raise Error(
                "text corpus doc order mismatch for "
                + dataset_key
                + " at index "
                + String(index)
            )


def max_query_vector_budget_for_public_benchmark_dataset(
    read dataset: PublicBenchmarkDataset
) -> Int:
    var max_budget = 0
    for judged_query in dataset.stored_task.task.queries:
        if judged_query.query.vector_count > max_budget:
            max_budget = judged_query.query.vector_count

    if max_budget <= 0:
        return 1
    return max_budget


def ensure_public_benchmark_dataset_collection_mirror(
    read dataset: PublicBenchmarkDataset,
    collection_suffix: String,
    document_proxy_vector_budget: Int = 0,
    centroid_postings_vector_budget: Int = 0,
    centroid_head_posting_cap: Int = 0,
    include_frontier_gem_graph: Bool = False,
) raises -> Path:
    var fine_cluster_count = 0
    var coarse_cluster_count = 0
    var cluster_cutoff = 0
    if include_frontier_gem_graph:
        var gem_config = frontier_gem_graph_build_config(
            dataset.stored_index,
            max_query_vector_budget_for_public_benchmark_dataset(dataset),
        )
        fine_cluster_count = gem_config.fine_cluster_count
        coarse_cluster_count = gem_config.coarse_cluster_count
        cluster_cutoff = gem_config.cluster_cutoff

    return ensure_one_segment_collection_mirror(
        public_benchmark_dataset_collection_root(dataset, collection_suffix),
        CollectionId(dataset.stored_task.dataset_id),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        dataset.stored_index,
        dataset.document_text_corpus,
        document_proxy_vector_budget,
        centroid_postings_vector_budget,
        centroid_head_posting_cap,
        fine_cluster_count,
        coarse_cluster_count,
        cluster_cutoff,
    )


def load_public_benchmark_dataset(
    dataset_key: String,
    load_text_corpus: Bool = False,
) raises -> PublicBenchmarkDataset:
    if dataset_key == "scifact_real_subset":
        var cache = ensure_scifact_real_subset_cache()
        return PublicBenchmarkDataset(
            "scifact_real_subset",
            "scifact_real_subset",
            False,
            cache.loaded_task_from_storage,
            cache.loaded_index_from_storage,
            cache.stored_task,
            cache.stored_index,
            empty_document_text_corpus(),
        )

    if dataset_key == "fiqa_real_subset":
        var cache = ensure_fiqa_real_subset_cache()
        return PublicBenchmarkDataset(
            "fiqa_real_subset",
            "fiqa_real_subset",
            False,
            cache.loaded_task_from_storage,
            cache.loaded_index_from_storage,
            cache.stored_task,
            cache.stored_index,
            empty_document_text_corpus(),
        )

    if dataset_key == "limit_small":
        var cache = ensure_limit_small_real_subset_cache()
        return PublicBenchmarkDataset(
            "limit_small",
            "limit_small",
            False,
            cache.loaded_task_from_storage,
            cache.loaded_index_from_storage,
            cache.stored_task,
            cache.stored_index,
            empty_document_text_corpus(),
        )

    if dataset_key == "browsecomp_plus_real_subset":
        var cache = ensure_browsecomp_plus_real_subset_cache()
        var document_text_corpus = empty_document_text_corpus()
        if load_text_corpus:
            document_text_corpus = (
                load_browsecomp_plus_real_subset_document_text_corpus()
            )
            require_public_benchmark_text_corpus_matches_index(
                dataset_key,
                cache.stored_index,
                document_text_corpus,
            )
        return PublicBenchmarkDataset(
            "browsecomp_plus_real_subset",
            "browsecomp_plus_real_subset",
            True,
            cache.loaded_task_from_storage,
            cache.loaded_index_from_storage,
            cache.stored_task,
            cache.stored_index,
            document_text_corpus,
        )

    if dataset_key == "browsecomp_plus_gold":
        var cache = ensure_browsecomp_plus_gold_real_subset_cache()
        var document_text_corpus = empty_document_text_corpus()
        if load_text_corpus:
            document_text_corpus = (
                load_browsecomp_plus_gold_real_subset_document_text_corpus()
            )
            require_public_benchmark_text_corpus_matches_index(
                dataset_key,
                cache.stored_index,
                document_text_corpus,
            )
        return PublicBenchmarkDataset(
            "browsecomp_plus_gold",
            "browsecomp_plus_gold",
            True,
            cache.loaded_task_from_storage,
            cache.loaded_index_from_storage,
            cache.stored_task,
            cache.stored_index,
            document_text_corpus,
        )

    raise Error("unknown public benchmark dataset key: " + dataset_key)


def default_public_benchmark_dataset_keys() -> List[String]:
    var dataset_keys = List[String]()
    dataset_keys.append("scifact_real_subset")
    dataset_keys.append("fiqa_real_subset")
    dataset_keys.append("limit_small")
    dataset_keys.append("browsecomp_plus_real_subset")
    dataset_keys.append("browsecomp_plus_gold")
    return dataset_keys^


def load_default_public_benchmark_datasets(
    load_text_corpus: Bool = False,
) raises -> List[PublicBenchmarkDataset]:
    var datasets = List[PublicBenchmarkDataset]()
    for dataset_key in default_public_benchmark_dataset_keys():
        datasets.append(
            load_public_benchmark_dataset(
                dataset_key,
                load_text_corpus=load_text_corpus,
            )
        )
    return datasets^
