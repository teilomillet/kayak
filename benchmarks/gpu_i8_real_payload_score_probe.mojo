import std.benchmark as benchmark
from std.collections import List
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import abs
from std.pathlib import Path
from std.sys.defines import get_defined_int, get_defined_string

from kayak.storage.binary_int_codec import (
    NON_NEGATIVE_INT_PAYLOAD_ENCODING_BINARY_U64_LE,
    read_non_negative_int_payload_with_encoding,
)
from kayak.storage.binary_vector_codec import read_binary_scalar_payload


comptime DATA_ROOT = get_defined_string[
    "data_root",
    ".cache/kayak/gpu_i8_real_payload_rerank/input",
]()
comptime QUERY_COUNT = get_defined_int["query_count", 2]()
comptime QUERY_VECTOR_COUNT = get_defined_int["query_vector_count", 8]()
comptime DOCUMENT_COUNT = get_defined_int["document_count", 64]()
comptime DOCUMENT_VECTOR_COUNT = get_defined_int["document_vector_count", 16]()
comptime CANDIDATE_K = get_defined_int["candidate_k", 32]()
comptime BLOCK_SIZE = get_defined_int["block_size", 128]()
comptime VECTOR_DIM = 128
comptime QUERY_VALUE_COUNT = QUERY_COUNT * QUERY_VECTOR_COUNT * VECTOR_DIM
comptime TOTAL_DOC_VECTOR_COUNT = DOCUMENT_COUNT * DOCUMENT_VECTOR_COUNT
comptime TOKEN_CODE_COUNT = TOTAL_DOC_VECTOR_COUNT * VECTOR_DIM
comptime DOC_OFFSET_COUNT = DOCUMENT_COUNT + 1
comptime CANDIDATE_SCORE_COUNT = QUERY_COUNT * CANDIDATE_K
comptime PARTIAL_SCORE_COUNT = (
    CANDIDATE_SCORE_COUNT * QUERY_VECTOR_COUNT * DOCUMENT_VECTOR_COUNT
)
comptime GRID_X = (CANDIDATE_SCORE_COUNT + BLOCK_SIZE - 1) // BLOCK_SIZE
comptime PARTIAL_GRID_X = (PARTIAL_SCORE_COUNT + BLOCK_SIZE - 1) // BLOCK_SIZE


def require_len(name: String, actual: Int, expected: Int) raises:
    if actual != expected:
        raise Error(
            name
            + " length mismatch: got "
            + String(actual)
            + " expected "
            + String(expected)
        )


def read_int8_payload(path: Path) raises -> List[Int8]:
    var bytes = path.read_bytes()
    var values = List[Int8]()
    for index in range(len(bytes)):
        var signed = Int(bytes[index])
        if signed > 127:
            signed -= 256
        values.append(Int8(signed))

    return values^


def read_u64_payload(path: Path) raises -> List[Int]:
    return read_non_negative_int_payload_with_encoding(
        path, NON_NEGATIVE_INT_PAYLOAD_ENCODING_BINARY_U64_LE
    )


def score_i8_candidate_kernel(
    query_values: UnsafePointer[Float32, MutAnyOrigin],
    token_codes: UnsafePointer[Int8, MutAnyOrigin],
    token_scales: UnsafePointer[Float32, MutAnyOrigin],
    doc_offsets: UnsafePointer[Int64, MutAnyOrigin],
    candidate_positions: UnsafePointer[Int64, MutAnyOrigin],
    score_out: UnsafePointer[Float32, MutAnyOrigin],
):
    var score_index = Int(global_idx.x)
    if score_index >= CANDIDATE_SCORE_COUNT:
        return

    var query_index = score_index // CANDIDATE_K
    var document_index = Int(candidate_positions[score_index])
    var start_token = Int(doc_offsets[document_index])
    var stop_token = Int(doc_offsets[document_index + 1])
    var query_base = query_index * QUERY_VECTOR_COUNT * VECTOR_DIM
    var total = Float32(0.0)

    for query_vector_index in range(QUERY_VECTOR_COUNT):
        var best_score = Float32(-3.4028234663852886e38)
        var query_offset = query_base + query_vector_index * VECTOR_DIM
        for token_index in range(start_token, stop_token):
            var token_offset = token_index * VECTOR_DIM
            var dot = Float32(0.0)
            for dim_index in range(VECTOR_DIM):
                dot += query_values[query_offset + dim_index] * Float32(
                    Int(token_codes[token_offset + dim_index])
                )
            var score = dot * token_scales[token_index]
            if score > best_score:
                best_score = score
        total += best_score

    score_out[score_index] = total


def score_i8_candidate_token_kernel(
    query_values: UnsafePointer[Float32, MutAnyOrigin],
    token_codes: UnsafePointer[Int8, MutAnyOrigin],
    token_scales: UnsafePointer[Float32, MutAnyOrigin],
    doc_offsets: UnsafePointer[Int64, MutAnyOrigin],
    candidate_positions: UnsafePointer[Int64, MutAnyOrigin],
    partial_scores: UnsafePointer[Float32, MutAnyOrigin],
):
    var partial_index = Int(global_idx.x)
    if partial_index >= PARTIAL_SCORE_COUNT:
        return

    var token_lane = partial_index % DOCUMENT_VECTOR_COUNT
    var partial_query_offset = partial_index // DOCUMENT_VECTOR_COUNT
    var query_vector_index = partial_query_offset % QUERY_VECTOR_COUNT
    var score_index = partial_query_offset // QUERY_VECTOR_COUNT
    var query_index = score_index // CANDIDATE_K
    var document_index = Int(candidate_positions[score_index])
    var token_index = Int(doc_offsets[document_index]) + token_lane
    var query_offset = (
        query_index * QUERY_VECTOR_COUNT * VECTOR_DIM
        + query_vector_index * VECTOR_DIM
    )
    var token_offset = token_index * VECTOR_DIM
    var dot = Float32(0.0)

    for dim_index in range(VECTOR_DIM):
        dot += query_values[query_offset + dim_index] * Float32(
            Int(token_codes[token_offset + dim_index])
        )

    partial_scores[partial_index] = dot * token_scales[token_index]


def reduce_i8_candidate_score_kernel(
    partial_scores: UnsafePointer[Float32, MutAnyOrigin],
    score_out: UnsafePointer[Float32, MutAnyOrigin],
):
    var score_index = Int(global_idx.x)
    if score_index >= CANDIDATE_SCORE_COUNT:
        return

    var total = Float32(0.0)
    var score_base = score_index * QUERY_VECTOR_COUNT * DOCUMENT_VECTOR_COUNT
    for query_vector_index in range(QUERY_VECTOR_COUNT):
        var best_score = Float32(-3.4028234663852886e38)
        var partial_base = (
            score_base + query_vector_index * DOCUMENT_VECTOR_COUNT
        )
        for token_lane in range(DOCUMENT_VECTOR_COUNT):
            var score = partial_scores[partial_base + token_lane]
            if score > best_score:
                best_score = score
        total += best_score

    score_out[score_index] = total


def main() raises:
    var root = Path(DATA_ROOT)
    var query_values = read_binary_scalar_payload(root / "query_values.bin")
    var token_codes = read_int8_payload(root / "token_codes.bin")
    var token_scales = read_binary_scalar_payload(root / "token_scales.bin")
    var doc_offsets = read_u64_payload(root / "doc_offsets.bin")
    var candidate_positions = read_u64_payload(root / "candidate_positions.bin")
    var reference_scores = read_binary_scalar_payload(
        root / "reference_scores.bin"
    )

    require_len("query_values", len(query_values), QUERY_VALUE_COUNT)
    require_len("token_codes", len(token_codes), TOKEN_CODE_COUNT)
    require_len("token_scales", len(token_scales), TOTAL_DOC_VECTOR_COUNT)
    require_len("doc_offsets", len(doc_offsets), DOC_OFFSET_COUNT)
    require_len(
        "candidate_positions", len(candidate_positions), CANDIDATE_SCORE_COUNT
    )
    require_len(
        "reference_scores", len(reference_scores), CANDIDATE_SCORE_COUNT
    )

    with DeviceContext() as ctx:
        var query_device = ctx.enqueue_create_buffer[DType.float32](
            QUERY_VALUE_COUNT
        )
        var codes_device = ctx.enqueue_create_buffer[DType.int8](
            TOKEN_CODE_COUNT
        )
        var scales_device = ctx.enqueue_create_buffer[DType.float32](
            TOTAL_DOC_VECTOR_COUNT
        )
        var offsets_device = ctx.enqueue_create_buffer[DType.int64](
            DOC_OFFSET_COUNT
        )
        var candidate_device = ctx.enqueue_create_buffer[DType.int64](
            CANDIDATE_SCORE_COUNT
        )
        var partial_score_device = ctx.enqueue_create_buffer[DType.float32](
            PARTIAL_SCORE_COUNT
        )
        var score_device = ctx.enqueue_create_buffer[DType.float32](
            CANDIDATE_SCORE_COUNT
        )
        var twopass_score_device = ctx.enqueue_create_buffer[DType.float32](
            CANDIDATE_SCORE_COUNT
        )

        var query_host = ctx.enqueue_create_host_buffer[DType.float32](
            QUERY_VALUE_COUNT
        )
        var codes_host = ctx.enqueue_create_host_buffer[DType.int8](
            TOKEN_CODE_COUNT
        )
        var scales_host = ctx.enqueue_create_host_buffer[DType.float32](
            TOTAL_DOC_VECTOR_COUNT
        )
        var offsets_host = ctx.enqueue_create_host_buffer[DType.int64](
            DOC_OFFSET_COUNT
        )
        var candidate_host = ctx.enqueue_create_host_buffer[DType.int64](
            CANDIDATE_SCORE_COUNT
        )
        var score_host = ctx.enqueue_create_host_buffer[DType.float32](
            CANDIDATE_SCORE_COUNT
        )
        var twopass_score_host = ctx.enqueue_create_host_buffer[DType.float32](
            CANDIDATE_SCORE_COUNT
        )

        for index in range(QUERY_VALUE_COUNT):
            query_host[index] = Float32(query_values[index])

        for index in range(TOKEN_CODE_COUNT):
            codes_host[index] = token_codes[index]

        for index in range(TOTAL_DOC_VECTOR_COUNT):
            scales_host[index] = Float32(token_scales[index])

        for index in range(DOC_OFFSET_COUNT):
            offsets_host[index] = Int64(doc_offsets[index])

        for index in range(CANDIDATE_SCORE_COUNT):
            candidate_host[index] = Int64(candidate_positions[index])
            score_host[index] = Float32(0.0)
            twopass_score_host[index] = Float32(0.0)

        def h2d_once() capturing raises:
            query_device.enqueue_copy_from(query_host)
            codes_device.enqueue_copy_from(codes_host)
            scales_device.enqueue_copy_from(scales_host)
            offsets_device.enqueue_copy_from(offsets_host)
            candidate_device.enqueue_copy_from(candidate_host)
            ctx.synchronize()

        def serial_kernel_once() capturing raises:
            ctx.enqueue_function[
                score_i8_candidate_kernel,
                score_i8_candidate_kernel,
            ](
                query_device,
                codes_device,
                scales_device,
                offsets_device,
                candidate_device,
                score_device,
                grid_dim=GRID_X,
                block_dim=BLOCK_SIZE,
            )
            ctx.synchronize()

        def twopass_kernel_once() capturing raises:
            ctx.enqueue_function[
                score_i8_candidate_token_kernel,
                score_i8_candidate_token_kernel,
            ](
                query_device,
                codes_device,
                scales_device,
                offsets_device,
                candidate_device,
                partial_score_device,
                grid_dim=PARTIAL_GRID_X,
                block_dim=BLOCK_SIZE,
            )
            ctx.enqueue_function[
                reduce_i8_candidate_score_kernel,
                reduce_i8_candidate_score_kernel,
            ](
                partial_score_device,
                twopass_score_device,
                grid_dim=GRID_X,
                block_dim=BLOCK_SIZE,
            )
            ctx.synchronize()

        def d2h_once() capturing raises:
            score_device.enqueue_copy_to(score_host)
            twopass_score_device.enqueue_copy_to(twopass_score_host)
            ctx.synchronize()

        var h2d = benchmark.run[h2d_once](max_iters=3)
        var serial_kernel = benchmark.run[serial_kernel_once](max_iters=3)
        var twopass_kernel = benchmark.run[twopass_kernel_once](max_iters=3)
        var d2h = benchmark.run[d2h_once](max_iters=3)

        h2d_once()
        serial_kernel_once()
        twopass_kernel_once()
        d2h_once()

        var score_delta_max_abs = Float64(0.0)
        var twopass_score_delta_max_abs = Float64(0.0)
        for score_index in range(CANDIDATE_SCORE_COUNT):
            var reference_score = Float32(reference_scores[score_index])
            var score_delta = abs(
                Float64(score_host[score_index]) - Float64(reference_score)
            )
            if score_delta > score_delta_max_abs:
                score_delta_max_abs = score_delta

            var twopass_score_delta = abs(
                Float64(twopass_score_host[score_index])
                - Float64(reference_score)
            )
            if twopass_score_delta > twopass_score_delta_max_abs:
                twopass_score_delta_max_abs = twopass_score_delta

        print("status: ok")
        print("payload_source: real_kayak_i8_snapshot")
        print("query_count: ", QUERY_COUNT)
        print("query_vector_count: ", QUERY_VECTOR_COUNT)
        print("document_count: ", DOCUMENT_COUNT)
        print("document_vector_count: ", DOCUMENT_VECTOR_COUNT)
        print("total_document_vector_count: ", TOTAL_DOC_VECTOR_COUNT)
        print("candidate_k: ", CANDIDATE_K)
        print("candidate_score_count: ", CANDIDATE_SCORE_COUNT)
        print("vector_dim: ", VECTOR_DIM)
        print("query_value_count: ", QUERY_VALUE_COUNT)
        print("token_code_count: ", TOKEN_CODE_COUNT)
        print("token_scale_count: ", TOTAL_DOC_VECTOR_COUNT)
        print("doc_offset_count: ", DOC_OFFSET_COUNT)
        print("candidate_position_count: ", CANDIDATE_SCORE_COUNT)
        print("partial_score_count: ", PARTIAL_SCORE_COUNT)
        print("block_size: ", BLOCK_SIZE)
        print("grid_x: ", GRID_X)
        print("partial_grid_x: ", PARTIAL_GRID_X)
        print("score_delta_max_abs: ", score_delta_max_abs)
        print("score_agreement_ok: ", score_delta_max_abs <= Float64(0.0001))
        print("twopass_score_delta_max_abs: ", twopass_score_delta_max_abs)
        print(
            "twopass_score_agreement_ok: ",
            twopass_score_delta_max_abs <= Float64(0.0001),
        )
        print("host_to_device_mean_seconds: ", h2d.mean())
        print("kernel_mean_seconds: ", serial_kernel.mean())
        print("serial_kernel_mean_seconds: ", serial_kernel.mean())
        print("twopass_kernel_mean_seconds: ", twopass_kernel.mean())
        print("device_to_host_mean_seconds: ", d2h.mean())
