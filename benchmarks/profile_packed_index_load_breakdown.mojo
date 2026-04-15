import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import StoredPackedIndex
from kayak.storage import (
    VECTOR_PAYLOAD_ENCODING_BINARY_LE,
    ensure_bright_stackoverflow_real_subset_cache,
    ensure_lemb_narrativeqa_real_subset_cache,
    ensure_r2med_biology_real_subset_cache,
    load_stored_packed_index,
    save_stored_packed_index,
)
from kayak.storage.binary_int_codec import (
    read_non_negative_int_payload_with_encoding,
)
from kayak.storage.binary_vector_codec import (
    decode_binary_vector_payload_with_encoding,
    read_binary_vector_payload_with_encoding,
)
from kayak.storage.manifest import (
    load_optional_manifest_value,
    read_manifest,
    require_manifest_value,
)
from kayak.storage.text_codec import parse_int, read_non_empty_lines

comptime LOAD_MAX_ITERS = 256


struct PackedIndexLoadBreakdownMeasurement(Copyable):
    var dataset_name: String
    var operation_name: String
    var document_count: Int
    var total_vector_count: Int
    var vector_dim: Int
    var mean_seconds: Float64

    def __init__(
        out self,
        var dataset_name: String,
        var operation_name: String,
        document_count: Int,
        total_vector_count: Int,
        vector_dim: Int,
        mean_seconds: Float64,
    ):
        self.dataset_name = dataset_name^
        self.operation_name = operation_name^
        self.document_count = document_count
        self.total_vector_count = total_vector_count
        self.vector_dim = vector_dim
        self.mean_seconds = mean_seconds

def append_measurement_tsv_line(
    mut out: String, read measurement: PackedIndexLoadBreakdownMeasurement
):
    out += measurement.dataset_name
    out += "\t"
    out += measurement.operation_name
    out += "\t"
    out += String(measurement.document_count)
    out += "\t"
    out += String(measurement.total_vector_count)
    out += "\t"
    out += String(measurement.vector_dim)
    out += "\t"
    out += String(measurement.mean_seconds)
    out += "\n"


def write_measurements_tsv(
    path: Path, measurements: List[PackedIndexLoadBreakdownMeasurement]
) raises:
    var lines = String()
    lines += "dataset_name\toperation_name\tdocument_count\ttotal_vector_count\tvector_dim\tmean_seconds\n"
    for measurement in measurements:
        append_measurement_tsv_line(lines, measurement)
    path.write_text(lines)


def benchmark_dataset(
    mut measurements: List[PackedIndexLoadBreakdownMeasurement],
    dataset_name: String,
    root: Path,
    read stored_index: StoredPackedIndex,
) raises:
    save_stored_packed_index(root, stored_index.copy())

    var manifest_path = root / "manifest.tsv"
    var doc_ids_path = root / "doc_ids.tsv"
    var doc_offsets_bin_path = root / "doc_offsets.bin"
    var doc_offsets_tsv_path = root / "doc_offsets.tsv"
    var token_vectors_path = root / "token_vectors.bin"
    var vector_dim = stored_index.index.vector_dim

    print("dataset: ", dataset_name)
    print("== read_manifest ==")
    def benchmark_manifest() capturing raises:
        bench_compiler.keep(read_manifest(manifest_path))
    var manifest_report = benchmark.run[benchmark_manifest](max_iters=LOAD_MAX_ITERS)
    manifest_report.print()
    print("")
    var manifest_mean = manifest_report.mean()

    var manifest = read_manifest(manifest_path)
    var vector_payload_encoding = require_manifest_value(
        manifest, "vector_payload_encoding"
    )
    var doc_offsets_encoding = load_optional_manifest_value(
        manifest, "doc_offsets_encoding"
    )

    print("== read_doc_ids ==")
    def benchmark_doc_ids() capturing raises:
        bench_compiler.keep(read_non_empty_lines(doc_ids_path))
    var doc_ids_report = benchmark.run[benchmark_doc_ids](max_iters=LOAD_MAX_ITERS)
    doc_ids_report.print()
    print("")
    var doc_ids_mean = doc_ids_report.mean()

    print("== read_doc_offsets ==")
    def benchmark_doc_offsets() capturing raises:
        if doc_offsets_encoding.byte_length() != 0 and doc_offsets_bin_path.exists():
            bench_compiler.keep(
                read_non_negative_int_payload_with_encoding(
                    doc_offsets_bin_path, doc_offsets_encoding
                )
            )
            return

        var doc_offsets = List[Int]()
        for line in read_non_empty_lines(doc_offsets_tsv_path):
            doc_offsets.append(parse_int(line, "doc offset"))
        bench_compiler.keep(doc_offsets)
    var doc_offsets_report = benchmark.run[benchmark_doc_offsets](max_iters=LOAD_MAX_ITERS)
    doc_offsets_report.print()
    print("")
    var doc_offsets_mean = doc_offsets_report.mean()

    print("== read_token_vectors ==")
    def benchmark_token_vectors() capturing raises:
        bench_compiler.keep(
            read_binary_vector_payload_with_encoding(
                token_vectors_path,
                vector_dim,
                vector_payload_encoding,
            )
        )
    var token_vectors_report = benchmark.run[benchmark_token_vectors](max_iters=LOAD_MAX_ITERS)
    token_vectors_report.print()
    print("")
    var token_vectors_mean = token_vectors_report.mean()

    print("== read_token_vector_bytes ==")
    def benchmark_token_vector_bytes() capturing raises:
        bench_compiler.keep(token_vectors_path.read_bytes())
    var token_vector_bytes_report = benchmark.run[benchmark_token_vector_bytes](
        max_iters=LOAD_MAX_ITERS
    )
    token_vector_bytes_report.print()
    print("")
    var token_vector_bytes_mean = token_vector_bytes_report.mean()

    var token_vector_bytes = token_vectors_path.read_bytes()
    print("== decode_token_vectors_from_bytes ==")
    def benchmark_decode_token_vectors() capturing raises:
        bench_compiler.keep(
            decode_binary_vector_payload_with_encoding(
                token_vector_bytes,
                vector_dim,
                vector_payload_encoding,
            )
        )
    var decode_token_vectors_report = benchmark.run[benchmark_decode_token_vectors](
        max_iters=LOAD_MAX_ITERS
    )
    decode_token_vectors_report.print()
    print("")
    var decode_token_vectors_mean = decode_token_vectors_report.mean()

    print("== load_stored_packed_index ==")
    def benchmark_full_load() capturing raises:
        bench_compiler.keep(load_stored_packed_index(root))
    var full_load_report = benchmark.run[benchmark_full_load](max_iters=LOAD_MAX_ITERS)
    full_load_report.print()
    print("")
    var full_load_mean = full_load_report.mean()

    measurements.append(
        PackedIndexLoadBreakdownMeasurement(
            dataset_name.copy(),
            "read_manifest",
            stored_index.index.document_count,
            stored_index.index.total_vector_count,
            vector_dim,
            manifest_mean,
        )
    )
    measurements.append(
        PackedIndexLoadBreakdownMeasurement(
            dataset_name.copy(),
            "read_doc_ids",
            stored_index.index.document_count,
            stored_index.index.total_vector_count,
            vector_dim,
            doc_ids_mean,
        )
    )
    measurements.append(
        PackedIndexLoadBreakdownMeasurement(
            dataset_name.copy(),
            "read_doc_offsets",
            stored_index.index.document_count,
            stored_index.index.total_vector_count,
            vector_dim,
            doc_offsets_mean,
        )
    )
    measurements.append(
        PackedIndexLoadBreakdownMeasurement(
            dataset_name.copy(),
            "read_token_vectors",
            stored_index.index.document_count,
            stored_index.index.total_vector_count,
            vector_dim,
            token_vectors_mean,
        )
    )
    measurements.append(
        PackedIndexLoadBreakdownMeasurement(
            dataset_name.copy(),
            "read_token_vector_bytes",
            stored_index.index.document_count,
            stored_index.index.total_vector_count,
            vector_dim,
            token_vector_bytes_mean,
        )
    )
    measurements.append(
        PackedIndexLoadBreakdownMeasurement(
            dataset_name.copy(),
            "decode_token_vectors_from_bytes",
            stored_index.index.document_count,
            stored_index.index.total_vector_count,
            vector_dim,
            decode_token_vectors_mean,
        )
    )
    measurements.append(
        PackedIndexLoadBreakdownMeasurement(
            dataset_name.copy(),
            "load_stored_packed_index",
            stored_index.index.document_count,
            stored_index.index.total_vector_count,
            vector_dim,
            full_load_mean,
        )
    )


def main() raises:
    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var measurements = List[PackedIndexLoadBreakdownMeasurement]()

    benchmark_dataset(
        measurements,
        "BRIGHT",
        output_root / "profile_packed_index_load_breakdown" / "bright",
        ensure_bright_stackoverflow_real_subset_cache().stored_index,
    )
    benchmark_dataset(
        measurements,
        "LEMB",
        output_root / "profile_packed_index_load_breakdown" / "lemb",
        ensure_lemb_narrativeqa_real_subset_cache().stored_index,
    )
    benchmark_dataset(
        measurements,
        "R2MED",
        output_root / "profile_packed_index_load_breakdown" / "r2med",
        ensure_r2med_biology_real_subset_cache().stored_index,
    )

    var output_path = output_root / "profile_packed_index_load_breakdown.tsv"
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
