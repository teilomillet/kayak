import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks.latent_proxy_profile_fixtures import (
    LatentProxyPrimitiveSweepProfile,
    make_projection,
    make_query,
    require_latent_proxy_primitive_profile,
)
from kayak.index import LatentQueryProjectionBlock
from kayak.index.latent_proxy import (
    apply_latent_proxy_activation,
    linear_block_output,
    layer_normalized_output,
    projected_query_token_for_block,
)
from kayak.numeric import VectorScalar


comptime LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS = 0.05
comptime LATENT_PROXY_PROFILE_WARMUP_ITERS = 4
comptime LATENT_PROXY_PROFILE_MAX_ITERS = 32
comptime HOTSPOT_PROFILE_NAME = "q32_lat2048_b2_docs128"


struct LatentProxyBreakdownMeasurement(Copyable):
    var profile_name: String
    var block_index: Int
    var block_input_dim: Int
    var block_output_dim: Int
    var phase_kind: String
    var mean_seconds: Float64

    def __init__(
        out self,
        var profile_name: String,
        block_index: Int,
        block_input_dim: Int,
        block_output_dim: Int,
        var phase_kind: String,
        mean_seconds: Float64,
    ):
        self.profile_name = profile_name^
        self.block_index = block_index
        self.block_input_dim = block_input_dim
        self.block_output_dim = block_output_dim
        self.phase_kind = phase_kind^
        self.mean_seconds = mean_seconds


def write_measurements_tsv(
    path: Path, measurements: List[LatentProxyBreakdownMeasurement]
) raises:
    var lines = String()
    lines += (
        "profile_name\tblock_index\tblock_input_dim\tblock_output_dim\tphase_kind\tmean_seconds\n"
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
        lines += measurement.phase_kind
        lines += "\t"
        lines += String(measurement.mean_seconds)
        lines += "\n"
    path.write_text(lines)


def post_linear_block_output(
    read linear_values: List[VectorScalar],
    read block: LatentQueryProjectionBlock,
) raises -> List[VectorScalar]:
    if block.order_kind == "linear_norm_activation":
        var projected = layer_normalized_output(linear_values, block)
        for index in range(len(projected)):
            projected[index] = apply_latent_proxy_activation(
                projected[index],
                block.activation_kind,
            )
            projected[index] = (
                projected[index] * block.activation_output_scale
            )
        return projected^

    var projected = List[VectorScalar]()
    for value in linear_values:
        projected.append(
            apply_latent_proxy_activation(value, block.activation_kind)
            * block.activation_output_scale
        )
    return layer_normalized_output(projected, block)


def append_measurement(
    mut measurements: List[LatentProxyBreakdownMeasurement],
    read profile: LatentProxyPrimitiveSweepProfile,
    read block: LatentQueryProjectionBlock,
    block_index: Int,
    phase_kind: String,
    mean_seconds: Float64,
):
    measurements.append(
        LatentProxyBreakdownMeasurement(
            profile.profile_name.copy(),
            block_index,
            block.input_dim,
            block.output_dim,
            phase_kind.copy(),
            mean_seconds,
        )
    )


def main() raises:
    var profile = require_latent_proxy_primitive_profile(HOTSPOT_PROFILE_NAME)
    var query = make_query(profile.query_vector_count, profile.vector_dim)
    var projection = make_projection(profile)
    if len(projection.blocks) != 2:
        raise Error("multiblock breakdown benchmark requires exactly two blocks")

    var block0 = projection.blocks[0].copy()
    var block1 = projection.blocks[1].copy()
    var block0_outputs = List[List[VectorScalar]]()
    var block0_linear_outputs = List[List[VectorScalar]]()
    var block1_linear_outputs = List[List[VectorScalar]]()

    for token_vector in query.token_vectors:
        var block0_linear = linear_block_output(token_vector, block0)
        block0_linear_outputs.append(block0_linear.copy())
        var block0_output = post_linear_block_output(block0_linear, block0)
        block0_outputs.append(block0_output.copy())
        block1_linear_outputs.append(linear_block_output(block0_output, block1))

    var measurements = List[LatentProxyBreakdownMeasurement]()

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

    def block0_linear_once() capturing raises:
        var query_index = 0
        for token_vector in query.token_vectors:
            bench_compiler.keep(linear_block_output(token_vector, block0))
            query_index += 1
        _ = query_index

    var block0_linear_report = benchmark.run[block0_linear_once](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=LATENT_PROXY_PROFILE_MAX_ITERS,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )
    append_measurement(
        measurements,
        profile,
        block0,
        0,
        "linear",
        block0_linear_report.mean(),
    )

    def block0_post_linear_once() capturing raises:
        for linear_values in block0_linear_outputs:
            bench_compiler.keep(post_linear_block_output(linear_values, block0))

    var block0_post_linear_report = benchmark.run[block0_post_linear_once](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=LATENT_PROXY_PROFILE_MAX_ITERS,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )
    append_measurement(
        measurements,
        profile,
        block0,
        0,
        "post_linear",
        block0_post_linear_report.mean(),
    )

    def block0_full_once() capturing raises:
        for token_vector in query.token_vectors:
            bench_compiler.keep(projected_query_token_for_block(token_vector, block0))

    var block0_full_report = benchmark.run[block0_full_once](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=LATENT_PROXY_PROFILE_MAX_ITERS,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )
    append_measurement(
        measurements,
        profile,
        block0,
        0,
        "full_block",
        block0_full_report.mean(),
    )

    def block1_linear_once() capturing raises:
        for token_vector in block0_outputs:
            bench_compiler.keep(linear_block_output(token_vector, block1))

    var block1_linear_report = benchmark.run[block1_linear_once](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=LATENT_PROXY_PROFILE_MAX_ITERS,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )
    append_measurement(
        measurements,
        profile,
        block1,
        1,
        "linear",
        block1_linear_report.mean(),
    )

    def block1_post_linear_once() capturing raises:
        for linear_values in block1_linear_outputs:
            bench_compiler.keep(post_linear_block_output(linear_values, block1))

    var block1_post_linear_report = benchmark.run[block1_post_linear_once](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=LATENT_PROXY_PROFILE_MAX_ITERS,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )
    append_measurement(
        measurements,
        profile,
        block1,
        1,
        "post_linear",
        block1_post_linear_report.mean(),
    )

    def block1_full_once() capturing raises:
        for token_vector in block0_outputs:
            bench_compiler.keep(projected_query_token_for_block(token_vector, block1))

    var block1_full_report = benchmark.run[block1_full_once](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=LATENT_PROXY_PROFILE_MAX_ITERS,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )
    append_measurement(
        measurements,
        profile,
        block1,
        1,
        "full_block",
        block1_full_report.mean(),
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "profile_latent_proxy_multiblock_breakdown.tsv"
    write_measurements_tsv(output_path, measurements)

    for measurement in measurements:
        print(
            "block=",
            measurement.block_index,
            " phase=",
            measurement.phase_kind,
            " mean=",
            measurement.mean_seconds,
        )

    print("")
    print("wrote ", output_path)
