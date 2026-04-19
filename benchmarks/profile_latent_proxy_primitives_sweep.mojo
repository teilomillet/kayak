import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import EncodedQuery
from kayak.index import (
    LATENT_PROXY_ACTIVATION_GELU,
    LATENT_PROXY_BLOCK_ORDER_LINEAR_NORM_ACTIVATION,
    LatentProxyIndex,
    LatentQueryProjection,
    LatentQueryProjectionBlock,
)
from kayak.numeric import VectorScalar
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
comptime LATENT_PROXY_INPUT_VECTOR_DIM = 128
comptime LATENT_PROXY_SEGMENT_ID = "segment-0001"
comptime LATENT_PROXY_CANDIDATE_K = 10


struct LatentProxyPrimitiveSweepProfile(Copyable):
    var profile_name: String
    var document_count: Int
    var query_vector_count: Int
    var vector_dim: Int
    var latent_dim: Int
    var projection_block_count: Int
    var candidate_k: Int

    def __init__(
        out self,
        var profile_name: String,
        document_count: Int,
        query_vector_count: Int,
        vector_dim: Int,
        latent_dim: Int,
        projection_block_count: Int,
        candidate_k: Int,
    ):
        self.profile_name = profile_name^
        self.document_count = document_count
        self.query_vector_count = query_vector_count
        self.vector_dim = vector_dim
        self.latent_dim = latent_dim
        self.projection_block_count = projection_block_count
        self.candidate_k = candidate_k


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


def deterministic_scalar(
    seed_a: Int, seed_b: Int, dim_index: Int
) -> VectorScalar:
    var numerator = ((seed_a + 1) * (seed_b + 3) * (dim_index + 5)) % 29
    return VectorScalar((Float64(numerator) - Float64(14.0)) / Float64(29.0))


def make_vector(seed_a: Int, seed_b: Int, vector_dim: Int) -> List[VectorScalar]:
    var values = List[VectorScalar]()
    for dim_index in range(vector_dim):
        values.append(deterministic_scalar(seed_a, seed_b, dim_index))
    return values^


def make_query(
    query_vector_count: Int, vector_dim: Int
) raises -> EncodedQuery:
    var query_vectors = List[List[VectorScalar]]()
    for vector_index in range(query_vector_count):
        query_vectors.append(make_vector(1000, vector_index, vector_dim))
    return EncodedQuery(query_vectors^)


def make_affine_weight(output_dim: Int, seed: Int) -> List[VectorScalar]:
    var values = List[VectorScalar]()
    for dim_index in range(output_dim):
        var offset = Float64(((seed + dim_index) % 7)) / Float64(100.0)
        values.append(VectorScalar(Float64(1.0) + offset))
    return values^


def make_affine_bias(output_dim: Int, seed: Int) -> List[VectorScalar]:
    var values = List[VectorScalar]()
    for dim_index in range(output_dim):
        values.append(
            VectorScalar(Float64(((seed + dim_index) % 5) - 2) / Float64(100.0))
        )
    return values^


def make_projection_block(
    input_dim: Int, output_dim: Int, seed: Int
) raises -> LatentQueryProjectionBlock:
    var linear_rows = List[List[VectorScalar]]()
    var linear_bias = List[VectorScalar]()
    for row_index in range(output_dim):
        linear_rows.append(make_vector(seed + row_index, seed + 17, input_dim))
        linear_bias.append(
            VectorScalar(
                Float64(((seed + row_index) % 11) - 5) / Float64(50.0)
            )
        )

    return LatentQueryProjectionBlock(
        LATENT_PROXY_BLOCK_ORDER_LINEAR_NORM_ACTIVATION,
        LATENT_PROXY_ACTIVATION_GELU,
        input_dim,
        output_dim,
        linear_rows^,
        linear_bias^,
        VectorScalar(1.0),
        True,
        VectorScalar(0.00001),
        make_affine_weight(output_dim, seed),
        make_affine_bias(output_dim, seed),
    )


def make_projection(
    read profile: LatentProxyPrimitiveSweepProfile
) raises -> LatentQueryProjection:
    var blocks = List[LatentQueryProjectionBlock]()
    if profile.projection_block_count == 1:
        blocks.append(
            make_projection_block(profile.vector_dim, profile.latent_dim, 0)
        )
    elif profile.projection_block_count == 2:
        var hidden_dim = profile.latent_dim // 2
        if hidden_dim < profile.vector_dim:
            hidden_dim = profile.vector_dim
        blocks.append(make_projection_block(profile.vector_dim, hidden_dim, 0))
        blocks.append(make_projection_block(hidden_dim, profile.latent_dim, 1))
    else:
        raise Error(
            "unsupported latent proxy projection_block_count: "
            + String(profile.projection_block_count)
        )

    return LatentQueryProjection(
        profile.vector_dim,
        profile.latent_dim,
        VectorScalar(profile.query_vector_count),
        blocks^,
    )


def make_proxy_index(
    document_count: Int, latent_dim: Int
) raises -> LatentProxyIndex:
    var doc_ids = List[String]()
    var proxy_vectors = List[List[VectorScalar]]()
    for document_index in range(document_count):
        doc_ids.append("doc-" + String(document_index))
        proxy_vectors.append(make_vector(document_index, 2000, latent_dim))
    return LatentProxyIndex(doc_ids^, proxy_vectors^, latent_dim)


def default_latent_proxy_primitive_profiles(
) -> List[LatentProxyPrimitiveSweepProfile]:
    return [
        LatentProxyPrimitiveSweepProfile(
            "q8_lat256_b1_docs128",
            128,
            8,
            LATENT_PROXY_INPUT_VECTOR_DIM,
            256,
            1,
            LATENT_PROXY_CANDIDATE_K,
        ),
        LatentProxyPrimitiveSweepProfile(
            "q32_lat256_b1_docs128",
            128,
            32,
            LATENT_PROXY_INPUT_VECTOR_DIM,
            256,
            1,
            LATENT_PROXY_CANDIDATE_K,
        ),
        LatentProxyPrimitiveSweepProfile(
            "q32_lat1024_b1_docs128",
            128,
            32,
            LATENT_PROXY_INPUT_VECTOR_DIM,
            1024,
            1,
            LATENT_PROXY_CANDIDATE_K,
        ),
        LatentProxyPrimitiveSweepProfile(
            "q32_lat2048_b1_docs128",
            128,
            32,
            LATENT_PROXY_INPUT_VECTOR_DIM,
            2048,
            1,
            LATENT_PROXY_CANDIDATE_K,
        ),
        LatentProxyPrimitiveSweepProfile(
            "q32_lat2048_b2_docs128",
            128,
            32,
            LATENT_PROXY_INPUT_VECTOR_DIM,
            2048,
            2,
            LATENT_PROXY_CANDIDATE_K,
        ),
        LatentProxyPrimitiveSweepProfile(
            "q64_lat2048_b1_docs128",
            128,
            64,
            LATENT_PROXY_INPUT_VECTOR_DIM,
            2048,
            1,
            LATENT_PROXY_CANDIDATE_K,
        ),
    ]


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
