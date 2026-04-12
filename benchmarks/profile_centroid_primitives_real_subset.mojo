import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import (
    CollectionId,
    JudgedTask,
    NamespaceId,
    ScoredCentroidSelection,
    SnapshotId,
    StoredPackedIndex,
    TenantId,
    accumulate_selected_centroid_scores,
    ensure_fiqa_real_subset_cache,
    ensure_limit_small_real_subset_cache,
    ensure_one_segment_collection_mirror,
    ensure_scifact_real_subset_cache,
    load_resolved_collection_snapshot,
)
from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import EncodedQuery, FlatQueryDim128, build_flat_query_dim128
from kayak.numeric import ScoreScalar
from kayak.planning.centroid_postings_flat_stage import (
    top_centroid_selection_for_flat_query_token_dim128,
)
from kayak.planning.centroid_postings_imputed_flat_stage import (
    centroid_selection_for_flat_query_token_dim128,
)
from kayak.planning.centroid_postings_imputed_stage import (
    centroid_selection_for_query_token,
)
from kayak.planning.centroid_postings_stage import top_centroid_selection_for_query_token


comptime CENTROID_HEAD_POSTING_CAP = 16


struct PrimitiveMeasurement(Copyable):
    var dataset_name: String
    var slice_name: String
    var primitive_kind: String
    var mean_seconds: Float64

    def __init__(
        out self,
        var dataset_name: String,
        var slice_name: String,
        var primitive_kind: String,
        mean_seconds: Float64,
    ):
        self.dataset_name = dataset_name^
        self.slice_name = slice_name^
        self.primitive_kind = primitive_kind^
        self.mean_seconds = mean_seconds


def append_measurement_tsv_line(mut out: String, read measurement: PrimitiveMeasurement):
    out += measurement.dataset_name
    out += "\t"
    out += measurement.slice_name
    out += "\t"
    out += measurement.primitive_kind
    out += "\t"
    out += String(measurement.mean_seconds)
    out += "\n"


def write_measurements_tsv(path: Path, measurements: List[PrimitiveMeasurement]) raises:
    var lines = String()
    lines += "dataset_name\tslice_name\tprimitive_kind\tmean_seconds\n"

    for measurement in measurements:
        append_measurement_tsv_line(lines, measurement)

    path.write_text(lines)


def load_profile_snapshot(
    collection_name: String,
    collection_root: Path,
    read stored_index: StoredPackedIndex,
) raises -> ResolvedCollectionSnapshot:
    var root = ensure_one_segment_collection_mirror(
        collection_root,
        CollectionId(collection_name),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        stored_index,
        0,
        0,
        CENTROID_HEAD_POSTING_CAP,
    )
    return load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))


def exact_selections_for_query(
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
) -> List[ScoredCentroidSelection]:
    var selections = List[ScoredCentroidSelection]()
    var index = snapshot.segments[0].stored_centroid_postings_index.index.copy()

    for token in query.token_vectors:
        selections.append(top_centroid_selection_for_query_token(token, index))

    return selections^


def exact_flat_selections_for_query(
    read query: FlatQueryDim128,
    read snapshot: ResolvedCollectionSnapshot,
) -> List[ScoredCentroidSelection]:
    var selections = List[ScoredCentroidSelection]()
    var index = snapshot.segments[0].stored_centroid_postings_index.index.copy()

    for query_index in range(query.vector_count):
        selections.append(
            top_centroid_selection_for_flat_query_token_dim128(
                query,
                query_index,
                index,
            )
        )

    return selections^


def imputed_selections_for_query(
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    final_k: Int,
) -> List[ScoredCentroidSelection]:
    var selections = List[ScoredCentroidSelection]()
    var index = snapshot.segments[0].stored_centroid_postings_index.index.copy()

    for token in query.token_vectors:
        selections.append(centroid_selection_for_query_token(token, index, final_k))

    return selections^


def imputed_flat_selections_for_query(
    read query: FlatQueryDim128,
    read snapshot: ResolvedCollectionSnapshot,
    final_k: Int,
) -> List[ScoredCentroidSelection]:
    var selections = List[ScoredCentroidSelection]()
    var index = snapshot.segments[0].stored_centroid_postings_index.index.copy()

    for query_index in range(query.vector_count):
        selections.append(
            centroid_selection_for_flat_query_token_dim128(
                query,
                query_index,
                index,
                final_k,
            )
        )

    return selections^


def baseline_score(read selections: List[ScoredCentroidSelection]) -> ScoreScalar:
    var total = ScoreScalar(0.0)
    for selection in selections:
        total += selection.baseline_correction
    return total


def accumulate_precomputed_selections(
    read selections: List[ScoredCentroidSelection],
    read snapshot: ResolvedCollectionSnapshot,
) -> List[ScoreScalar]:
    var index = snapshot.segments[0].stored_centroid_postings_index.index.copy()
    var scores = List[ScoreScalar]()
    var active_flags = List[Int]()
    var active_doc_indices = List[Int]()
    var base = baseline_score(selections)

    for _ in range(index.document_count):
        scores.append(base)
        active_flags.append(0)

    for selection in selections:
        accumulate_selected_centroid_scores(
            selection,
            index,
            scores,
            active_doc_indices,
            active_flags,
        )

    return scores^


def append_measurements_for_dataset(
    mut measurements: List[PrimitiveMeasurement],
    dataset_name: String,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
) raises:
    print("dataset=", dataset_name, " slice=", task.slice_name)
    print("")

    var query_index = 0
    print("== exact_selection_nested ==")

    def exact_selection_nested_once() capturing:
        bench_compiler.keep(
            exact_selections_for_query(
                task.queries[query_index].query,
                snapshot,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[exact_selection_nested_once]()
    report.print()
    print("")
    measurements.append(
        PrimitiveMeasurement(
            dataset_name.copy(),
            task.slice_name.copy(),
            "exact_selection_nested",
            report.mean(),
        )
    )

    var flat_queries = List[FlatQueryDim128]()
    for judged_query in task.queries:
        flat_queries.append(build_flat_query_dim128(judged_query.query))

    query_index = 0
    print("== exact_selection_flat ==")

    def exact_selection_flat_once() capturing:
        bench_compiler.keep(
            exact_flat_selections_for_query(
                flat_queries[query_index],
                snapshot,
            )
        )
        query_index += 1
        if query_index == len(flat_queries):
            query_index = 0

    report = benchmark.run[exact_selection_flat_once]()
    report.print()
    print("")
    measurements.append(
        PrimitiveMeasurement(
            dataset_name.copy(),
            task.slice_name.copy(),
            "exact_selection_flat",
            report.mean(),
        )
    )

    query_index = 0
    print("== imputed_selection_nested ==")

    def imputed_selection_nested_once() capturing:
        bench_compiler.keep(
            imputed_selections_for_query(
                task.queries[query_index].query,
                snapshot,
                task.k,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    report = benchmark.run[imputed_selection_nested_once]()
    report.print()
    print("")
    measurements.append(
        PrimitiveMeasurement(
            dataset_name.copy(),
            task.slice_name.copy(),
            "imputed_selection_nested",
            report.mean(),
        )
    )

    query_index = 0
    print("== imputed_selection_flat ==")

    def imputed_selection_flat_once() capturing:
        bench_compiler.keep(
            imputed_flat_selections_for_query(
                flat_queries[query_index],
                snapshot,
                task.k,
            )
        )
        query_index += 1
        if query_index == len(flat_queries):
            query_index = 0

    report = benchmark.run[imputed_selection_flat_once]()
    report.print()
    print("")
    measurements.append(
        PrimitiveMeasurement(
            dataset_name.copy(),
            task.slice_name.copy(),
            "imputed_selection_flat",
            report.mean(),
        )
    )

    var exact_selection_cache = List[List[ScoredCentroidSelection]]()
    var imputed_selection_cache = List[List[ScoredCentroidSelection]]()
    for judged_query in task.queries:
        exact_selection_cache.append(
            exact_selections_for_query(judged_query.query, snapshot)
        )
        imputed_selection_cache.append(
            imputed_selections_for_query(judged_query.query, snapshot, task.k)
        )

    query_index = 0
    print("== exact_accumulation ==")

    def exact_accumulation_once() capturing:
        bench_compiler.keep(
            accumulate_precomputed_selections(
                exact_selection_cache[query_index],
                snapshot,
            )
        )
        query_index += 1
        if query_index == len(exact_selection_cache):
            query_index = 0

    report = benchmark.run[exact_accumulation_once]()
    report.print()
    print("")
    measurements.append(
        PrimitiveMeasurement(
            dataset_name.copy(),
            task.slice_name.copy(),
            "exact_accumulation",
            report.mean(),
        )
    )

    query_index = 0
    print("== imputed_accumulation ==")

    def imputed_accumulation_once() capturing:
        bench_compiler.keep(
            accumulate_precomputed_selections(
                imputed_selection_cache[query_index],
                snapshot,
            )
        )
        query_index += 1
        if query_index == len(imputed_selection_cache):
            query_index = 0

    report = benchmark.run[imputed_accumulation_once]()
    report.print()
    print("")
    measurements.append(
        PrimitiveMeasurement(
            dataset_name.copy(),
            task.slice_name.copy(),
            "imputed_accumulation",
            report.mean(),
        )
    )


def main() raises:
    var measurements = List[PrimitiveMeasurement]()

    var scifact_cache = ensure_scifact_real_subset_cache()
    var scifact_snapshot = load_profile_snapshot(
        "scifact_real_subset_primitives",
        Path(".cache/kayak/scifact_real_subset_primitives"),
        scifact_cache.stored_index,
    )
    append_measurements_for_dataset(
        measurements,
        "SciFact",
        scifact_cache.stored_task.task,
        scifact_snapshot,
    )

    var fiqa_cache = ensure_fiqa_real_subset_cache()
    var fiqa_snapshot = load_profile_snapshot(
        "fiqa_real_subset_primitives",
        Path(".cache/kayak/fiqa_real_subset_primitives"),
        fiqa_cache.stored_index,
    )
    append_measurements_for_dataset(
        measurements,
        "FIQA",
        fiqa_cache.stored_task.task,
        fiqa_snapshot,
    )

    var limit_small_cache = ensure_limit_small_real_subset_cache()
    var limit_small_snapshot = load_profile_snapshot(
        "limit_small_real_subset_primitives",
        Path(".cache/kayak/limit_small_real_subset_primitives"),
        limit_small_cache.stored_index,
    )
    append_measurements_for_dataset(
        measurements,
        "LIMIT-small",
        limit_small_cache.stored_task.task,
        limit_small_snapshot,
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "profile_centroid_primitives_real_subset.tsv"
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
