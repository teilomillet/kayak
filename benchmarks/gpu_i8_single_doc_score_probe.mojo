import std.benchmark as benchmark
from std.gpu.host import DeviceContext
from std.math import abs
from std.sys.defines import get_defined_int


comptime QUERY_VECTOR_COUNT = get_defined_int["query_vector_count", 2]()
comptime DOCUMENT_VECTOR_COUNT = get_defined_int["document_vector_count", 4]()
comptime VECTOR_DIM = 128
comptime QUERY_VALUE_COUNT = QUERY_VECTOR_COUNT * VECTOR_DIM
comptime TOKEN_CODE_COUNT = DOCUMENT_VECTOR_COUNT * VECTOR_DIM


def query_value(index: Int) -> Float32:
    var raw = (index * 17 + 3) % 23
    return Float32(raw - 11) * Float32(0.03125)


def token_code(index: Int) -> Int8:
    var raw = (index * 13 + 5) % 31
    return Int8(raw - 15)


def token_scale(token_index: Int) -> Float32:
    return Float32(token_index % 5 + 1) * Float32(0.0078125)


def cpu_reference_single_doc_score() -> Float32:
    var total = Float32(0.0)
    for query_vector_index in range(QUERY_VECTOR_COUNT):
        var best_score = Float32(-3.4028234663852886e38)
        var query_offset = query_vector_index * VECTOR_DIM
        for token_index in range(DOCUMENT_VECTOR_COUNT):
            var token_offset = token_index * VECTOR_DIM
            var dot = Float32(0.0)
            for dim_index in range(VECTOR_DIM):
                dot += query_value(query_offset + dim_index) * Float32(
                    Int(token_code(token_offset + dim_index))
                )
            var score = dot * token_scale(token_index)
            if score > best_score:
                best_score = score
        total += best_score

    return total


def score_i8_single_doc_kernel(
    query_values: UnsafePointer[Float32, MutAnyOrigin],
    token_codes: UnsafePointer[Int8, MutAnyOrigin],
    token_scales: UnsafePointer[Float32, MutAnyOrigin],
    score_out: UnsafePointer[Float32, MutAnyOrigin],
):
    var total = Float32(0.0)
    for query_vector_index in range(QUERY_VECTOR_COUNT):
        var best_score = Float32(-3.4028234663852886e38)
        var query_offset = query_vector_index * VECTOR_DIM
        for token_index in range(DOCUMENT_VECTOR_COUNT):
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

    score_out[0] = total


def main() raises:
    with DeviceContext() as ctx:
        var query_device = ctx.enqueue_create_buffer[DType.float32](
            QUERY_VALUE_COUNT
        )
        var codes_device = ctx.enqueue_create_buffer[DType.int8](
            TOKEN_CODE_COUNT
        )
        var scales_device = ctx.enqueue_create_buffer[DType.float32](
            DOCUMENT_VECTOR_COUNT
        )
        var score_device = ctx.enqueue_create_buffer[DType.float32](1)

        var query_host = ctx.enqueue_create_host_buffer[DType.float32](
            QUERY_VALUE_COUNT
        )
        var codes_host = ctx.enqueue_create_host_buffer[DType.int8](
            TOKEN_CODE_COUNT
        )
        var scales_host = ctx.enqueue_create_host_buffer[DType.float32](
            DOCUMENT_VECTOR_COUNT
        )
        var score_host = ctx.enqueue_create_host_buffer[DType.float32](1)

        for index in range(QUERY_VALUE_COUNT):
            query_host[index] = query_value(index)

        for index in range(TOKEN_CODE_COUNT):
            codes_host[index] = token_code(index)

        for token_index in range(DOCUMENT_VECTOR_COUNT):
            scales_host[token_index] = token_scale(token_index)

        score_host[0] = Float32(0.0)

        def h2d_once() capturing raises:
            query_device.enqueue_copy_from(query_host)
            codes_device.enqueue_copy_from(codes_host)
            scales_device.enqueue_copy_from(scales_host)
            ctx.synchronize()

        def kernel_once() capturing raises:
            ctx.enqueue_function[
                score_i8_single_doc_kernel,
                score_i8_single_doc_kernel,
            ](
                query_device,
                codes_device,
                scales_device,
                score_device,
                grid_dim=1,
                block_dim=1,
            )
            ctx.synchronize()

        def d2h_once() capturing raises:
            score_device.enqueue_copy_to(score_host)
            ctx.synchronize()

        var h2d = benchmark.run[h2d_once](max_iters=3)
        var kernel = benchmark.run[kernel_once](max_iters=3)
        var d2h = benchmark.run[d2h_once](max_iters=3)

        h2d_once()
        kernel_once()
        d2h_once()

        var cpu_score = cpu_reference_single_doc_score()
        var gpu_score = score_host[0]
        var score_delta = abs(Float64(gpu_score) - Float64(cpu_score))

        print("status: ok")
        print("query_vector_count: ", QUERY_VECTOR_COUNT)
        print("document_vector_count: ", DOCUMENT_VECTOR_COUNT)
        print("vector_dim: ", VECTOR_DIM)
        print("query_value_count: ", QUERY_VALUE_COUNT)
        print("token_code_count: ", TOKEN_CODE_COUNT)
        print("candidate_position_count: ", 1)
        print("candidate_score_count: ", 1)
        print("cpu_score: ", cpu_score)
        print("gpu_score: ", gpu_score)
        print("score_delta_abs: ", score_delta)
        print("score_agreement_ok: ", score_delta <= Float64(0.00001))
        print("host_to_device_mean_seconds: ", h2d.mean())
        print("kernel_mean_seconds: ", kernel.mean())
        print("device_to_host_mean_seconds: ", d2h.mean())
