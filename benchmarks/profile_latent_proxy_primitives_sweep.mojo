import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks.latent_proxy_profile_fixtures import (
    DEFAULT_LATENT_PROXY_INPUT_VECTOR_DIM,
    LatentProxyPrimitiveSweepProfile,
    default_latent_proxy_primitive_profiles,
    make_projection,
    make_proxy_index,
    make_query,
)
from kayak.planning import (
    ProjectedLatentQuery,
    project_query_with_latent_proxy,
    project_query_with_latent_proxy_generic,
    project_query_with_latent_proxy_single_block,
    segment_hits_for_projected_latent_query,
    sum_projected_latent_query_scores_against_index,
)


comptime LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS = 0.05
comptime LATENT_PROXY_PROFILE_WARMUP_ITERS = 4
comptime LATENT_PROXY_PROFILE_MAX_ITERS = 32
comptime LATENT_PROXY_SEGMENT_ID = "segment-0001"


struct LatentProxyPrimitiveSweepMeasurement(Copyable):
    var profile_name: String
    var document_count: Int
    var query_vector_count: Int
    var vector_dim: Int
    var latent_dim: Int
    var projection_block_count: Int
    var candidate_k: Int
    var mean_projection_generic_seconds: Float64
    var mean_projection_single_block_seconds: Float64
    var mean_projection_seconds: Float64
    var mean_scan_sum_seconds: Float64
    var mean_segment_hits_seconds: Float64
    var mean_projection_plus_segment_hits_seconds: Float64

    def __init__(
        out self,
        var profile_name: String,
        document_count: Int,
        query_vector_count: Int,
        vector_dim: Int,
        latent_dim: Int,
        projection_block_count: Int,
        candidate_k: Int,
        mean_projection_generic_seconds: Float64,
        mean_projection_single_block_seconds: Float64,
        mean_projection_seconds: Float64,
        mean_scan_sum_seconds: Float64,
        mean_segment_hits_seconds: Float64,
        mean_projection_plus_segment_hits_seconds: Float64,
    ):
        self.profile_name = profile_name^
        self.document_count = document_count
        self.query_vector_count = query_vector_count
        self.vector_dim = vector_dim
        self.latent_dim = latent_dim
        self.projection_block_count = projection_block_count
        self.candidate_k = candidate_k
        self.mean_projection_generic_seconds = mean_projection_generic_seconds
        self.mean_projection_single_block_seconds = (
            mean_projection_single_block_seconds
        )
        self.mean_projection_seconds = mean_projection_seconds
        self.mean_scan_sum_seconds = mean_scan_sum_seconds
        self.mean_segment_hits_seconds = mean_segment_hits_seconds
        self.mean_projection_plus_segment_hits_seconds = (
            mean_projection_plus_segment_hits_seconds
        )


def append_measurement_tsv_line(
    mut out: String, read measurement: LatentProxyPrimitiveSweepMeasurement
):
    out += measurement.profile_name
    out += "\t"
    out += String(measurement.document_count)
    out += "\t"
    out += String(measurement.query_vector_count)
    out += "\t"
    out += String(measurement.vector_dim)
    out += "\t"
    out += String(measurement.latent_dim)
    out += "\t"
    out += String(measurement.projection_block_count)
    out += "\t"
    out += String(measurement.candidate_k)
    out += "\t"
    out += String(measurement.mean_projection_generic_seconds)
    out += "\t"
    out += String(measurement.mean_projection_single_block_seconds)
    out += "\t"
    out += String(measurement.mean_projection_seconds)
    out += "\t"
    out += String(measurement.mean_scan_sum_seconds)
    out += "\t"
    out += String(measurement.mean_segment_hits_seconds)
    out += "\t"
    out += String(measurement.mean_projection_plus_segment_hits_seconds)
    out += "\n"


def write_measurements_tsv(
    path: Path, measurements: List[LatentProxyPrimitiveSweepMeasurement]
) raises:
    var lines = String()
    lines += (
        "profile_name\tdocument_count\tquery_vector_count\tvector_dim\tlatent_dim\tprojection_block_count\tcandidate_k\t"
    )
    lines += (
        "mean_projection_generic_seconds\tmean_projection_single_block_seconds\tmean_projection_seconds\t"
    )
    lines += (
        "mean_scan_sum_seconds\tmean_segment_hits_seconds\tmean_projection_plus_segment_hits_seconds\n"
    )

    for measurement in measurements:
        append_measurement_tsv_line(lines, measurement)

    path.write_text(lines)

def append_measurements_for_profile(
    mut measurements: List[LatentProxyPrimitiveSweepMeasurement],
    read profile: LatentProxyPrimitiveSweepProfile,
) raises:
    var query = make_query(profile.query_vector_count, profile.vector_dim)
    var projection = make_projection(profile)
    var index = make_proxy_index(profile.document_count, profile.latent_dim)

    print("profile=", profile.profile_name)
    print(
        "shape: docs=",
        profile.document_count,
        " query_vecs=",
        profile.query_vector_count,
        " dim=",
        profile.vector_dim,
        " latent=",
        profile.latent_dim,
        " blocks=",
        profile.projection_block_count,
        " top_k=",
        profile.candidate_k,
    )

    def projection_once() capturing raises:
        bench_compiler.keep(project_query_with_latent_proxy(query, projection))

    var projection_report = benchmark.run[projection_once](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=LATENT_PROXY_PROFILE_MAX_ITERS,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )

    def projection_generic_once() capturing raises:
        bench_compiler.keep(project_query_with_latent_proxy_generic(query, projection))

    var projection_generic_report = benchmark.run[projection_generic_once](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=LATENT_PROXY_PROFILE_MAX_ITERS,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )

    var projection_single_block_mean = Float64(-1.0)
    if profile.projection_block_count == 1:
        def projection_single_block_once() capturing raises:
            bench_compiler.keep(
                project_query_with_latent_proxy_single_block(query, projection)
            )

        var projection_single_block_report = benchmark.run[
            projection_single_block_once
        ](
            num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
            max_iters=LATENT_PROXY_PROFILE_MAX_ITERS,
            min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
            max_batch_size=1,
        )
        projection_single_block_mean = Float64(
            projection_single_block_report.mean()
        )

    var projected = project_query_with_latent_proxy(query, projection)

    def scan_sum_once() capturing raises:
        bench_compiler.keep(
            sum_projected_latent_query_scores_against_index(projected, index)
        )

    var scan_sum_report = benchmark.run[scan_sum_once](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=LATENT_PROXY_PROFILE_MAX_ITERS,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )

    def segment_hits_once() capturing raises:
        bench_compiler.keep(
            segment_hits_for_projected_latent_query(
                projected,
                LATENT_PROXY_SEGMENT_ID,
                0,
                index,
                profile.candidate_k,
            )
        )

    var segment_hits_report = benchmark.run[segment_hits_once](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=LATENT_PROXY_PROFILE_MAX_ITERS,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )

    def projection_plus_segment_hits_once() capturing raises:
        bench_compiler.keep(
            segment_hits_for_projected_latent_query(
                project_query_with_latent_proxy(query, projection),
                LATENT_PROXY_SEGMENT_ID,
                0,
                index,
                profile.candidate_k,
            )
        )

    var projection_plus_segment_hits_report = benchmark.run[
        projection_plus_segment_hits_once
    ](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=LATENT_PROXY_PROFILE_MAX_ITERS,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )

    print("mean projection=", projection_report.mean())
    print("mean segment hits=", segment_hits_report.mean())
    print("")

    measurements.append(
        LatentProxyPrimitiveSweepMeasurement(
            profile.profile_name.copy(),
            profile.document_count,
            profile.query_vector_count,
            profile.vector_dim,
            profile.latent_dim,
            profile.projection_block_count,
            profile.candidate_k,
            Float64(projection_generic_report.mean()),
            projection_single_block_mean,
            Float64(projection_report.mean()),
            Float64(scan_sum_report.mean()),
            Float64(segment_hits_report.mean()),
            Float64(projection_plus_segment_hits_report.mean()),
        )
    )


def main() raises:
    var measurements = List[LatentProxyPrimitiveSweepMeasurement]()
    for profile in default_latent_proxy_primitive_profiles():
        append_measurements_for_profile(measurements, profile)

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "profile_latent_proxy_primitives_sweep.tsv"
    write_measurements_tsv(output_path, measurements)
    print("wrote ", output_path)
