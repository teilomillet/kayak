import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path
from std.sys.info import simd_width_of

from kayak.benchmarks.latent_proxy_profile_fixtures import (
    LatentProxyPrimitiveSweepProfile,
    make_projection,
    make_query,
    require_latent_proxy_primitive_profile,
)
from kayak.index import LatentQueryProjectionBlock
from kayak.index.latent_proxy import (
    linear_block_output,
    projected_query_token_for_block,
)
from kayak.numeric import VECTOR_SCALAR_NAME, VectorScalar, zero_vector_scalar


comptime LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS = 0.05
comptime LATENT_PROXY_PROFILE_WARMUP_ITERS = 4
comptime LATENT_PROXY_PROFILE_MAX_ITERS = 32
comptime HOTSPOT_PROFILE_NAME = "q32_lat2048_b2_docs128"


struct LatentProxyLinearStorageMeasurement(Copyable):
    var profile_name: String
    var block_index: Int
    var block_input_dim: Int
    var block_output_dim: Int
    var mean_row_list_seconds: Float64
    var mean_flat_rows_seconds: Float64

    def __init__(
        out self,
        var profile_name: String,
        block_index: Int,
        block_input_dim: Int,
        block_output_dim: Int,
        mean_row_list_seconds: Float64,
        mean_flat_rows_seconds: Float64,
    ):
        self.profile_name = profile_name^
        self.block_index = block_index
        self.block_input_dim = block_input_dim
        self.block_output_dim = block_output_dim
        self.mean_row_list_seconds = mean_row_list_seconds
        self.mean_flat_rows_seconds = mean_flat_rows_seconds


def write_measurements_tsv(
    path: Path, measurements: List[LatentProxyLinearStorageMeasurement]
) raises:
    var lines = String()
    lines += (
        "profile_name\tblock_index\tblock_input_dim\tblock_output_dim\tmean_row_list_seconds\tmean_flat_rows_seconds\n"
    )
    for measurement in measurements:
        lines += measurement.profile_name
        lines += "\t"
        lines += String(measurement.block_index)
        lines += "\t"
        lines += String(measurement.block_input_dim)
        lines += "\t"
        lines += String(measurement.block_output_dim)
        lines += "\t"
        lines += String(measurement.mean_row_list_seconds)
        lines += "\t"
        lines += String(measurement.mean_flat_rows_seconds)
        lines += "\n"
    path.write_text(lines)


def flatten_linear_rows(
    read rows: List[List[VectorScalar]], input_dim: Int
) raises -> List[VectorScalar]:
    var flat = List[VectorScalar]()
    for row in rows:
        if len(row) != input_dim:
            raise Error("flat latent proxy row width mismatch")
        for value in row:
            flat.append(value)
    return flat^


def dot_product_flat_segment(
    read lhs: List[VectorScalar],
    read flat_rhs: List[VectorScalar],
    rhs_offset: Int,
    vector_dim: Int,
) -> VectorScalar:
    if VECTOR_SCALAR_NAME != "Float32":
        var total = zero_vector_scalar()
        for index in range(vector_dim):
            total += lhs[index] * flat_rhs[rhs_offset + index]
        return total

    comptime width = simd_width_of[VectorScalar]()
    var simd_limit = (vector_dim // width) * width
    var lhs_ptr = lhs.unsafe_ptr()
    var rhs_ptr = flat_rhs.unsafe_ptr() + rhs_offset
    var accum = SIMD[DType.float32, width](0.0)

    for index in range(0, simd_limit, width):
        accum += (
            (lhs_ptr + index).load[width=width]()
            * (rhs_ptr + index).load[width=width]()
        )

    var total = VectorScalar(accum.reduce_add()[0])
    for index in range(simd_limit, vector_dim):
        total += lhs.unsafe_get(index) * flat_rhs.unsafe_get(rhs_offset + index)
    return total


def linear_block_output_flat_rows(
    read input_vector: List[VectorScalar],
    read block: LatentQueryProjectionBlock,
    read flat_rows: List[VectorScalar],
) raises -> List[VectorScalar]:
    if len(input_vector) != block.input_dim:
        raise Error("latent proxy linear block input dimension mismatch")
    if len(flat_rows) != block.input_dim * block.output_dim:
        raise Error("latent proxy flat row storage size mismatch")
    var output = List[VectorScalar]()
    for row_index in range(block.output_dim):
        var row_offset = row_index * block.input_dim
        output.append(
            dot_product_flat_segment(
                input_vector,
                flat_rows,
                row_offset,
                block.input_dim,
            )
            + block.linear_bias[row_index]
        )
    return output^


def append_measurement(
    mut measurements: List[LatentProxyLinearStorageMeasurement],
    read profile: LatentProxyPrimitiveSweepProfile,
    block_index: Int,
    read block: LatentQueryProjectionBlock,
    mean_row_list_seconds: Float64,
    mean_flat_rows_seconds: Float64,
):
    measurements.append(
        LatentProxyLinearStorageMeasurement(
            profile.profile_name.copy(),
            block_index,
            block.input_dim,
            block.output_dim,
            mean_row_list_seconds,
            mean_flat_rows_seconds,
        )
    )


def main() raises:
    var profile = require_latent_proxy_primitive_profile(HOTSPOT_PROFILE_NAME)
    var query = make_query(profile.query_vector_count, profile.vector_dim)
    var projection = make_projection(profile)
    if len(projection.blocks) != 2:
        raise Error("linear storage experiment requires exactly two blocks")

    var block0 = projection.blocks[0].copy()
    var block1 = projection.blocks[1].copy()
    var block0_inputs = query.token_vectors.copy()
    var block1_inputs = List[List[VectorScalar]]()
    for token_vector in block0_inputs:
        block1_inputs.append(projected_query_token_for_block(token_vector, block0))

    var block0_flat_rows = flatten_linear_rows(block0.linear_rows, block0.input_dim)
    var block1_flat_rows = flatten_linear_rows(block1.linear_rows, block1.input_dim)

    var measurements = List[LatentProxyLinearStorageMeasurement]()

    print("profile=", profile.profile_name)
    print(
        "shape: query_vecs=",
        profile.query_vector_count,
        " dim=",
        profile.vector_dim,
        " latent=",
        profile.latent_dim,
        " blocks=",
        profile.projection_block_count,
    )
    print("")

    def block0_row_list_once() capturing raises:
        for token_vector in block0_inputs:
            bench_compiler.keep(linear_block_output(token_vector, block0))

    var block0_row_list_report = benchmark.run[block0_row_list_once](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=LATENT_PROXY_PROFILE_MAX_ITERS,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )

    def block0_flat_rows_once() capturing raises:
        for token_vector in block0_inputs:
            bench_compiler.keep(
                linear_block_output_flat_rows(token_vector, block0, block0_flat_rows)
            )

    var block0_flat_rows_report = benchmark.run[block0_flat_rows_once](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=LATENT_PROXY_PROFILE_MAX_ITERS,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )
    append_measurement(
        measurements,
        profile,
        0,
        block0,
        block0_row_list_report.mean(),
        block0_flat_rows_report.mean(),
    )

    def block1_row_list_once() capturing raises:
        for token_vector in block1_inputs:
            bench_compiler.keep(linear_block_output(token_vector, block1))

    var block1_row_list_report = benchmark.run[block1_row_list_once](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=LATENT_PROXY_PROFILE_MAX_ITERS,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )

    def block1_flat_rows_once() capturing raises:
        for token_vector in block1_inputs:
            bench_compiler.keep(
                linear_block_output_flat_rows(token_vector, block1, block1_flat_rows)
            )

    var block1_flat_rows_report = benchmark.run[block1_flat_rows_once](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=LATENT_PROXY_PROFILE_MAX_ITERS,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )
    append_measurement(
        measurements,
        profile,
        1,
        block1,
        block1_row_list_report.mean(),
        block1_flat_rows_report.mean(),
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "profile_latent_proxy_linear_storage_experiment.tsv"
    write_measurements_tsv(output_path, measurements)

    for measurement in measurements:
        print(
            "block=",
            measurement.block_index,
            " row_list=",
            measurement.mean_row_list_seconds,
            " flat_rows=",
            measurement.mean_flat_rows_seconds,
        )

    print("")
    print("wrote ", output_path)
