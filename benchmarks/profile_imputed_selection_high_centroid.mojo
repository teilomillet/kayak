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
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
    loaded_segment_stored_centroid_postings_index,
)
from kayak.benchmarks import (
    high_centroid_synthetic_hard_recall_profile,
    make_synthetic_hard_recall_fixture,
)
from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import EncodedQuery
from kayak.numeric import VectorScalar
from kayak.planning.centroid_postings_imputed_flat_stage import (
    centroid_selection_for_flat_query_token_generic,
)
from kayak.planning.centroid_postings_imputed_stage import (
    centroid_selection_for_query_token,
)


comptime CENTROID_HEAD_POSTING_CAP = 16


struct ImputedSelectionMeasurement(Copyable):
    var slice_name: String
    var primitive_kind: String
    var vector_dim: Int
    var centroid_count: Int
    var mean_seconds: Float64

    def __init__(
        out self,
        var slice_name: String,
        var primitive_kind: String,
        vector_dim: Int,
        centroid_count: Int,
        mean_seconds: Float64,
    ):
        self.slice_name = slice_name^
        self.primitive_kind = primitive_kind^
        self.vector_dim = vector_dim
        self.centroid_count = centroid_count
        self.mean_seconds = mean_seconds


def append_measurement_tsv_line(
    mut out: String, read measurement: ImputedSelectionMeasurement
):
    out += measurement.slice_name
    out += "\t"
    out += measurement.primitive_kind
    out += "\t"
    out += String(measurement.vector_dim)
    out += "\t"
    out += String(measurement.centroid_count)
    out += "\t"
    out += String(measurement.mean_seconds)
    out += "\n"


def write_measurements_tsv(
    path: Path, measurements: List[ImputedSelectionMeasurement]
) raises:
    var lines = String()
    lines += "slice_name\tprimitive_kind\tvector_dim\tcentroid_count\tmean_seconds\n"

    for measurement in measurements:
        append_measurement_tsv_line(lines, measurement)

    path.write_text(lines)


def load_profile_snapshot(
    collection_name: String,
    collection_root: Path,
    read stored_index: StoredPackedIndex,
    centroid_budget: Int,
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
        centroid_budget,
        CENTROID_HEAD_POSTING_CAP,
    )
    return load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))


def flatten_query_values(read query: EncodedQuery) -> List[VectorScalar]:
    var values = List[VectorScalar]()

    for token_vector in query.token_vectors:
        for value in token_vector:
            values.append(value)

    return values^


def imputed_nested_selections_for_query(
    read query: EncodedQuery,
    final_k: Int,
    centroid_count: Int,
    read snapshot: ResolvedCollectionSnapshot,
) raises -> List[ScoredCentroidSelection]:
    var selections = List[ScoredCentroidSelection]()
    var index = loaded_segment_stored_centroid_postings_index(
        snapshot.segments[0]
    ).index.copy()

    if index.centroid_count != centroid_count:
        raise Error("imputed nested probe requires stable centroid_count")

    for token in query.token_vectors:
        selections.append(centroid_selection_for_query_token(token, index, final_k))

    return selections^


def imputed_flat_selections_for_query(
    read query: EncodedQuery,
    final_k: Int,
    centroid_count: Int,
    read snapshot: ResolvedCollectionSnapshot,
) raises -> List[ScoredCentroidSelection]:
    var selections = List[ScoredCentroidSelection]()
    var index = loaded_segment_stored_centroid_postings_index(
        snapshot.segments[0]
    ).index.copy()

    if index.centroid_count != centroid_count:
        raise Error("imputed flat probe requires stable centroid_count")

    var flat_query_values = flatten_query_values(query)
    for query_index in range(query.vector_count):
        selections.append(
            centroid_selection_for_flat_query_token_generic(
                flat_query_values,
                query_index * query.vector_dim,
                index,
                final_k,
            )
        )

    return selections^


def append_measurements_for_snapshot(
    mut measurements: List[ImputedSelectionMeasurement],
    read snapshot: ResolvedCollectionSnapshot,
    read task: JudgedTask,
) raises:
    var index = loaded_segment_stored_centroid_postings_index(
        snapshot.segments[0]
    ).index.copy()
    if index.centroid_count <= 128:
        raise Error(
            "high-centroid imputed selection probe requires centroid_count above 128"
        )

    print(
        "slice= ",
        task.slice_name,
        " vector_dim= ",
        index.vector_dim,
        " centroid_count= ",
        index.centroid_count,
    )
    print("")

    var query_index = 0
    print("== imputed_selection_nested ==")

    def nested_once() capturing raises:
        bench_compiler.keep(
            imputed_nested_selections_for_query(
                task.queries[query_index].query,
                task.k,
                index.centroid_count,
                snapshot,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var nested_report = benchmark.run[nested_once]()
    nested_report.print()
    print("")
    measurements.append(
        ImputedSelectionMeasurement(
            task.slice_name.copy(),
            "imputed_selection_nested",
            index.vector_dim,
            index.centroid_count,
            nested_report.mean(),
        )
    )

    query_index = 0
    print("== imputed_selection_flat ==")

    def flat_once() capturing raises:
        bench_compiler.keep(
            imputed_flat_selections_for_query(
                task.queries[query_index].query,
                task.k,
                index.centroid_count,
                snapshot,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var flat_report = benchmark.run[flat_once]()
    flat_report.print()
    print("")
    measurements.append(
        ImputedSelectionMeasurement(
            task.slice_name.copy(),
            "imputed_selection_flat",
            index.vector_dim,
            index.centroid_count,
            flat_report.mean(),
        )
    )


def main() raises:
    var profile = high_centroid_synthetic_hard_recall_profile()
    var fixture = make_synthetic_hard_recall_fixture(profile)
    var snapshot = load_profile_snapshot(
        "synthetic_high_centroid_imputed_probe",
        Path(".cache/kayak/synthetic_high_centroid_imputed_probe"),
        fixture.stored_index,
        fixture.stored_index.index.vector_dim,
    )

    var measurements = List[ImputedSelectionMeasurement]()
    append_measurements_for_snapshot(
        measurements,
        snapshot,
        fixture.stored_task.task,
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "profile_imputed_selection_high_centroid.tsv"
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
