import std.benchmark as benchmark
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import abs
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder


comptime VECTOR_DIM = 128
comptime BLOCK_SIZE = 128


def gpu_extension_device_probe() raises -> PythonObject:
    with DeviceContext() as ctx:
        ctx.synchronize()
        return Python.str("ok")


def require_len(name: String, py_values: PythonObject, expected: Int) raises:
    if len(py_values) != expected:
        raise Error(
            name
            + " length mismatch: got "
            + String(len(py_values))
            + " expected "
            + String(expected)
        )


def score_i8_candidate_token_kernel_dynamic(
    query_values: UnsafePointer[Float32, MutAnyOrigin],
    token_codes: UnsafePointer[Int8, MutAnyOrigin],
    token_scales: UnsafePointer[Float32, MutAnyOrigin],
    doc_offsets: UnsafePointer[Int64, MutAnyOrigin],
    candidate_positions: UnsafePointer[Int64, MutAnyOrigin],
    partial_scores: UnsafePointer[Float32, MutAnyOrigin],
    query_vector_count: Int,
    document_vector_count: Int,
    candidate_k: Int,
    partial_score_count: Int,
):
    var partial_index = Int(global_idx.x)
    if partial_index >= partial_score_count:
        return

    var token_lane = partial_index % document_vector_count
    var partial_query_offset = partial_index // document_vector_count
    var query_vector_index = partial_query_offset % query_vector_count
    var score_index = partial_query_offset // query_vector_count
    var query_index = score_index // candidate_k
    var document_index = Int(candidate_positions[score_index])
    var token_index = Int(doc_offsets[document_index]) + token_lane
    var query_offset = (
        query_index * query_vector_count * VECTOR_DIM
        + query_vector_index * VECTOR_DIM
    )
    var token_offset = token_index * VECTOR_DIM
    var dot = Float32(0.0)

    for dim_index in range(VECTOR_DIM):
        dot += query_values[query_offset + dim_index] * Float32(
            Int(token_codes[token_offset + dim_index])
        )

    partial_scores[partial_index] = dot * token_scales[token_index]


def reduce_i8_candidate_score_kernel_dynamic(
    partial_scores: UnsafePointer[Float32, MutAnyOrigin],
    score_out: UnsafePointer[Float32, MutAnyOrigin],
    query_vector_count: Int,
    document_vector_count: Int,
    candidate_score_count: Int,
):
    var score_index = Int(global_idx.x)
    if score_index >= candidate_score_count:
        return

    var total = Float32(0.0)
    var score_base = score_index * query_vector_count * document_vector_count
    for query_vector_index in range(query_vector_count):
        var best_score = Float32(-3.4028234663852886e38)
        var partial_base = (
            score_base + query_vector_index * document_vector_count
        )
        for token_lane in range(document_vector_count):
            var score = partial_scores[partial_base + token_lane]
            if score > best_score:
                best_score = score
        total += best_score

    score_out[score_index] = total


def score_i8_real_payload_once(py_request: PythonObject) raises -> PythonObject:
    var py_query_values = py_request[0]
    var py_token_codes = py_request[1]
    var py_token_scales = py_request[2]
    var py_doc_offsets = py_request[3]
    var py_candidate_positions = py_request[4]
    var py_reference_scores = py_request[5]
    var query_count = Int(py=py_request[6])
    var query_vector_count = Int(py=py_request[7])
    var document_count = Int(py=py_request[8])
    var document_vector_count = Int(py=py_request[9])
    var candidate_k = Int(py=py_request[10])
    var total_document_vector_count = document_count * document_vector_count
    var query_value_count = query_count * query_vector_count * VECTOR_DIM
    var token_code_count = total_document_vector_count * VECTOR_DIM
    var doc_offset_count = document_count + 1
    var candidate_score_count = query_count * candidate_k
    var partial_score_count = (
        candidate_score_count * query_vector_count * document_vector_count
    )
    var grid_x = (candidate_score_count + BLOCK_SIZE - 1) // BLOCK_SIZE
    var partial_grid_x = (partial_score_count + BLOCK_SIZE - 1) // BLOCK_SIZE

    require_len("query_values", py_query_values, query_value_count)
    require_len("token_codes", py_token_codes, token_code_count)
    require_len("token_scales", py_token_scales, total_document_vector_count)
    require_len("doc_offsets", py_doc_offsets, doc_offset_count)
    require_len(
        "candidate_positions", py_candidate_positions, candidate_score_count
    )
    require_len("reference_scores", py_reference_scores, candidate_score_count)

    with DeviceContext() as ctx:
        var query_device = ctx.enqueue_create_buffer[DType.float32](
            query_value_count
        )
        var codes_device = ctx.enqueue_create_buffer[DType.int8](
            token_code_count
        )
        var scales_device = ctx.enqueue_create_buffer[DType.float32](
            total_document_vector_count
        )
        var offsets_device = ctx.enqueue_create_buffer[DType.int64](
            doc_offset_count
        )
        var candidate_device = ctx.enqueue_create_buffer[DType.int64](
            candidate_score_count
        )
        var partial_score_device = ctx.enqueue_create_buffer[DType.float32](
            partial_score_count
        )
        var score_device = ctx.enqueue_create_buffer[DType.float32](
            candidate_score_count
        )

        var query_host = ctx.enqueue_create_host_buffer[DType.float32](
            query_value_count
        )
        var codes_host = ctx.enqueue_create_host_buffer[DType.int8](
            token_code_count
        )
        var scales_host = ctx.enqueue_create_host_buffer[DType.float32](
            total_document_vector_count
        )
        var offsets_host = ctx.enqueue_create_host_buffer[DType.int64](
            doc_offset_count
        )
        var candidate_host = ctx.enqueue_create_host_buffer[DType.int64](
            candidate_score_count
        )
        var score_host = ctx.enqueue_create_host_buffer[DType.float32](
            candidate_score_count
        )

        for index in range(query_value_count):
            query_host[index] = Float32(py=py_query_values[index])

        for index in range(token_code_count):
            codes_host[index] = Int8(py=py_token_codes[index])

        for index in range(total_document_vector_count):
            scales_host[index] = Float32(py=py_token_scales[index])

        for index in range(doc_offset_count):
            offsets_host[index] = Int64(py=py_doc_offsets[index])

        for index in range(candidate_score_count):
            candidate_host[index] = Int64(py=py_candidate_positions[index])
            score_host[index] = Float32(0.0)

        def h2d_once() capturing raises:
            query_device.enqueue_copy_from(query_host)
            codes_device.enqueue_copy_from(codes_host)
            scales_device.enqueue_copy_from(scales_host)
            offsets_device.enqueue_copy_from(offsets_host)
            candidate_device.enqueue_copy_from(candidate_host)
            ctx.synchronize()

        def twopass_kernel_once() capturing raises:
            ctx.enqueue_function[
                score_i8_candidate_token_kernel_dynamic,
                score_i8_candidate_token_kernel_dynamic,
            ](
                query_device,
                codes_device,
                scales_device,
                offsets_device,
                candidate_device,
                partial_score_device,
                query_vector_count,
                document_vector_count,
                candidate_k,
                partial_score_count,
                grid_dim=partial_grid_x,
                block_dim=BLOCK_SIZE,
            )
            ctx.enqueue_function[
                reduce_i8_candidate_score_kernel_dynamic,
                reduce_i8_candidate_score_kernel_dynamic,
            ](
                partial_score_device,
                score_device,
                query_vector_count,
                document_vector_count,
                candidate_score_count,
                grid_dim=grid_x,
                block_dim=BLOCK_SIZE,
            )
            ctx.synchronize()

        def d2h_once() capturing raises:
            score_device.enqueue_copy_to(score_host)
            ctx.synchronize()

        var h2d = benchmark.run[h2d_once](max_iters=3)
        var kernel = benchmark.run[twopass_kernel_once](max_iters=3)
        var d2h = benchmark.run[d2h_once](max_iters=3)

        h2d_once()
        twopass_kernel_once()
        d2h_once()

        var score_delta_max_abs = Float64(0.0)
        for score_index in range(candidate_score_count):
            var reference_score = Float32(py=py_reference_scores[score_index])
            var score_delta = abs(
                Float64(score_host[score_index]) - Float64(reference_score)
            )
            if score_delta > score_delta_max_abs:
                score_delta_max_abs = score_delta

        var py_scores = Python.list()
        for score_index in range(candidate_score_count):
            py_scores.append(Python.float(score_host[score_index]))

        var py_result = Python.list()
        py_result.append(Python.float(h2d.mean()))
        py_result.append(Python.float(kernel.mean()))
        py_result.append(Python.float(d2h.mean()))
        py_result.append(Python.float(score_delta_max_abs))
        py_result.append(Python.int(candidate_score_count))
        py_result.append(py_scores)
        return py_result


def profile_i8_prepared_payload_session(
    py_request: PythonObject,
) raises -> PythonObject:
    var py_query_values = py_request[0]
    var py_token_codes = py_request[1]
    var py_token_scales = py_request[2]
    var py_doc_offsets = py_request[3]
    var py_candidate_positions = py_request[4]
    var py_reference_scores = py_request[5]
    var query_count = Int(py=py_request[6])
    var query_vector_count = Int(py=py_request[7])
    var document_count = Int(py=py_request[8])
    var document_vector_count = Int(py=py_request[9])
    var candidate_k = Int(py=py_request[10])
    var warmup_iterations = Int(py=py_request[11])
    var measurement_iterations = Int(py=py_request[12])
    if query_count <= 0:
        raise Error("query_count must be positive")
    if query_vector_count <= 0:
        raise Error("query_vector_count must be positive")
    if document_count <= 0:
        raise Error("document_count must be positive")
    if document_vector_count <= 0:
        raise Error("document_vector_count must be positive")
    if candidate_k <= 0:
        raise Error("candidate_k must be positive")
    if warmup_iterations < 0:
        raise Error("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise Error("measurement_iterations must be positive")

    var total_document_vector_count = document_count * document_vector_count
    var query_value_count = query_count * query_vector_count * VECTOR_DIM
    var token_code_count = total_document_vector_count * VECTOR_DIM
    var doc_offset_count = document_count + 1
    var candidate_score_count = query_count * candidate_k
    var partial_score_count = (
        candidate_score_count * query_vector_count * document_vector_count
    )
    var grid_x = (candidate_score_count + BLOCK_SIZE - 1) // BLOCK_SIZE
    var partial_grid_x = (partial_score_count + BLOCK_SIZE - 1) // BLOCK_SIZE

    require_len("query_values", py_query_values, query_value_count)
    require_len("token_codes", py_token_codes, token_code_count)
    require_len("token_scales", py_token_scales, total_document_vector_count)
    require_len("doc_offsets", py_doc_offsets, doc_offset_count)
    require_len(
        "candidate_positions", py_candidate_positions, candidate_score_count
    )
    require_len("reference_scores", py_reference_scores, candidate_score_count)

    with DeviceContext() as ctx:
        var query_device = ctx.enqueue_create_buffer[DType.float32](
            query_value_count
        )
        var codes_device = ctx.enqueue_create_buffer[DType.int8](
            token_code_count
        )
        var scales_device = ctx.enqueue_create_buffer[DType.float32](
            total_document_vector_count
        )
        var offsets_device = ctx.enqueue_create_buffer[DType.int64](
            doc_offset_count
        )
        var candidate_device = ctx.enqueue_create_buffer[DType.int64](
            candidate_score_count
        )
        var partial_score_device = ctx.enqueue_create_buffer[DType.float32](
            partial_score_count
        )
        var score_device = ctx.enqueue_create_buffer[DType.float32](
            candidate_score_count
        )

        var query_host = ctx.enqueue_create_host_buffer[DType.float32](
            query_value_count
        )
        var codes_host = ctx.enqueue_create_host_buffer[DType.int8](
            token_code_count
        )
        var scales_host = ctx.enqueue_create_host_buffer[DType.float32](
            total_document_vector_count
        )
        var offsets_host = ctx.enqueue_create_host_buffer[DType.int64](
            doc_offset_count
        )
        var candidate_host = ctx.enqueue_create_host_buffer[DType.int64](
            candidate_score_count
        )
        var score_host = ctx.enqueue_create_host_buffer[DType.float32](
            candidate_score_count
        )

        for index in range(query_value_count):
            query_host[index] = Float32(py=py_query_values[index])

        for index in range(token_code_count):
            codes_host[index] = Int8(py=py_token_codes[index])

        for index in range(total_document_vector_count):
            scales_host[index] = Float32(py=py_token_scales[index])

        for index in range(doc_offset_count):
            offsets_host[index] = Int64(py=py_doc_offsets[index])

        for index in range(candidate_score_count):
            var candidate_position = Int64(py=py_candidate_positions[index])
            if candidate_position < 0 or candidate_position >= Int64(
                document_count
            ):
                raise Error(
                    "candidate position outside prepared index document range"
                )
            candidate_host[index] = candidate_position
            score_host[index] = Float32(0.0)

        def prepare_h2d_once() capturing raises:
            codes_device.enqueue_copy_from(codes_host)
            scales_device.enqueue_copy_from(scales_host)
            offsets_device.enqueue_copy_from(offsets_host)
            ctx.synchronize()

        def score_h2d_once() capturing raises:
            query_device.enqueue_copy_from(query_host)
            candidate_device.enqueue_copy_from(candidate_host)
            ctx.synchronize()

        def twopass_kernel_once() capturing raises:
            ctx.enqueue_function[
                score_i8_candidate_token_kernel_dynamic,
                score_i8_candidate_token_kernel_dynamic,
            ](
                query_device,
                codes_device,
                scales_device,
                offsets_device,
                candidate_device,
                partial_score_device,
                query_vector_count,
                document_vector_count,
                candidate_k,
                partial_score_count,
                grid_dim=partial_grid_x,
                block_dim=BLOCK_SIZE,
            )
            ctx.enqueue_function[
                reduce_i8_candidate_score_kernel_dynamic,
                reduce_i8_candidate_score_kernel_dynamic,
            ](
                partial_score_device,
                score_device,
                query_vector_count,
                document_vector_count,
                candidate_score_count,
                grid_dim=grid_x,
                block_dim=BLOCK_SIZE,
            )
            ctx.synchronize()

        def d2h_once() capturing raises:
            score_device.enqueue_copy_to(score_host)
            ctx.synchronize()

        var prepare_h2d = benchmark.run[prepare_h2d_once](
            max_iters=measurement_iterations
        )
        prepare_h2d_once()

        for _ in range(warmup_iterations):
            score_h2d_once()
            twopass_kernel_once()
            d2h_once()

        var score_h2d = benchmark.run[score_h2d_once](
            max_iters=measurement_iterations
        )
        var kernel = benchmark.run[twopass_kernel_once](
            max_iters=measurement_iterations
        )
        var d2h = benchmark.run[d2h_once](max_iters=measurement_iterations)

        score_h2d_once()
        twopass_kernel_once()
        d2h_once()

        var score_delta_max_abs = Float64(0.0)
        for score_index in range(candidate_score_count):
            var reference_score = Float32(py=py_reference_scores[score_index])
            var score_delta = abs(
                Float64(score_host[score_index]) - Float64(reference_score)
            )
            if score_delta > score_delta_max_abs:
                score_delta_max_abs = score_delta

        var py_scores = Python.list()
        for score_index in range(candidate_score_count):
            py_scores.append(Python.float(score_host[score_index]))

        var py_result = Python.list()
        py_result.append(Python.float(prepare_h2d.mean()))
        py_result.append(Python.float(score_h2d.mean()))
        py_result.append(Python.float(kernel.mean()))
        py_result.append(Python.float(d2h.mean()))
        py_result.append(Python.float(score_delta_max_abs))
        py_result.append(Python.int(candidate_score_count))
        py_result.append(py_scores)
        return py_result


def profile_i8_prepared_payload_session_addresses(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_address = Int(py=py_request[0])
    var token_codes_address = Int(py=py_request[1])
    var token_scales_address = Int(py=py_request[2])
    var doc_offsets_address = Int(py=py_request[3])
    var candidate_positions_address = Int(py=py_request[4])
    var reference_scores_address = Int(py=py_request[5])
    var query_count = Int(py=py_request[6])
    var query_vector_count = Int(py=py_request[7])
    var document_count = Int(py=py_request[8])
    var document_vector_count = Int(py=py_request[9])
    var candidate_k = Int(py=py_request[10])
    var warmup_iterations = Int(py=py_request[11])
    var measurement_iterations = Int(py=py_request[12])
    if query_address == 0:
        raise Error("query address must be non-zero")
    if token_codes_address == 0:
        raise Error("token_codes address must be non-zero")
    if token_scales_address == 0:
        raise Error("token_scales address must be non-zero")
    if doc_offsets_address == 0:
        raise Error("doc_offsets address must be non-zero")
    if candidate_positions_address == 0:
        raise Error("candidate_positions address must be non-zero")
    if reference_scores_address == 0:
        raise Error("reference_scores address must be non-zero")
    if query_count <= 0:
        raise Error("query_count must be positive")
    if query_vector_count <= 0:
        raise Error("query_vector_count must be positive")
    if document_count <= 0:
        raise Error("document_count must be positive")
    if document_vector_count <= 0:
        raise Error("document_vector_count must be positive")
    if candidate_k <= 0:
        raise Error("candidate_k must be positive")
    if warmup_iterations < 0:
        raise Error("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise Error("measurement_iterations must be positive")

    var py_query_values = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=query_address
    )
    var py_token_codes = UnsafePointer[Int8, MutAnyOrigin](
        unsafe_from_address=token_codes_address
    )
    var py_token_scales = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=token_scales_address
    )
    var py_doc_offsets = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=doc_offsets_address
    )
    var py_candidate_positions = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=candidate_positions_address
    )
    var py_reference_scores = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=reference_scores_address
    )

    var total_document_vector_count = document_count * document_vector_count
    var query_value_count = query_count * query_vector_count * VECTOR_DIM
    var token_code_count = total_document_vector_count * VECTOR_DIM
    var doc_offset_count = document_count + 1
    var candidate_score_count = query_count * candidate_k
    var partial_score_count = (
        candidate_score_count * query_vector_count * document_vector_count
    )
    var grid_x = (candidate_score_count + BLOCK_SIZE - 1) // BLOCK_SIZE
    var partial_grid_x = (partial_score_count + BLOCK_SIZE - 1) // BLOCK_SIZE

    with DeviceContext() as ctx:
        var query_device = ctx.enqueue_create_buffer[DType.float32](
            query_value_count
        )
        var codes_device = ctx.enqueue_create_buffer[DType.int8](
            token_code_count
        )
        var scales_device = ctx.enqueue_create_buffer[DType.float32](
            total_document_vector_count
        )
        var offsets_device = ctx.enqueue_create_buffer[DType.int64](
            doc_offset_count
        )
        var candidate_device = ctx.enqueue_create_buffer[DType.int64](
            candidate_score_count
        )
        var partial_score_device = ctx.enqueue_create_buffer[DType.float32](
            partial_score_count
        )
        var score_device = ctx.enqueue_create_buffer[DType.float32](
            candidate_score_count
        )

        var query_host = ctx.enqueue_create_host_buffer[DType.float32](
            query_value_count
        )
        var codes_host = ctx.enqueue_create_host_buffer[DType.int8](
            token_code_count
        )
        var scales_host = ctx.enqueue_create_host_buffer[DType.float32](
            total_document_vector_count
        )
        var offsets_host = ctx.enqueue_create_host_buffer[DType.int64](
            doc_offset_count
        )
        var candidate_host = ctx.enqueue_create_host_buffer[DType.int64](
            candidate_score_count
        )
        var score_host = ctx.enqueue_create_host_buffer[DType.float32](
            candidate_score_count
        )

        for index in range(query_value_count):
            query_host[index] = py_query_values[index]

        for index in range(token_code_count):
            codes_host[index] = py_token_codes[index]

        for index in range(total_document_vector_count):
            scales_host[index] = py_token_scales[index]

        for index in range(doc_offset_count):
            offsets_host[index] = py_doc_offsets[index]

        for index in range(candidate_score_count):
            var candidate_position = py_candidate_positions[index]
            if candidate_position < 0 or candidate_position >= Int64(
                document_count
            ):
                raise Error(
                    "candidate position outside prepared index document range"
                )
            candidate_host[index] = candidate_position
            score_host[index] = Float32(0.0)

        def prepare_h2d_once() capturing raises:
            codes_device.enqueue_copy_from(codes_host)
            scales_device.enqueue_copy_from(scales_host)
            offsets_device.enqueue_copy_from(offsets_host)
            ctx.synchronize()

        def score_h2d_once() capturing raises:
            query_device.enqueue_copy_from(query_host)
            candidate_device.enqueue_copy_from(candidate_host)
            ctx.synchronize()

        def twopass_kernel_once() capturing raises:
            ctx.enqueue_function[
                score_i8_candidate_token_kernel_dynamic,
                score_i8_candidate_token_kernel_dynamic,
            ](
                query_device,
                codes_device,
                scales_device,
                offsets_device,
                candidate_device,
                partial_score_device,
                query_vector_count,
                document_vector_count,
                candidate_k,
                partial_score_count,
                grid_dim=partial_grid_x,
                block_dim=BLOCK_SIZE,
            )
            ctx.enqueue_function[
                reduce_i8_candidate_score_kernel_dynamic,
                reduce_i8_candidate_score_kernel_dynamic,
            ](
                partial_score_device,
                score_device,
                query_vector_count,
                document_vector_count,
                candidate_score_count,
                grid_dim=grid_x,
                block_dim=BLOCK_SIZE,
            )
            ctx.synchronize()

        def d2h_once() capturing raises:
            score_device.enqueue_copy_to(score_host)
            ctx.synchronize()

        var prepare_h2d = benchmark.run[prepare_h2d_once](
            max_iters=measurement_iterations
        )
        prepare_h2d_once()

        for _ in range(warmup_iterations):
            score_h2d_once()
            twopass_kernel_once()
            d2h_once()

        var score_h2d = benchmark.run[score_h2d_once](
            max_iters=measurement_iterations
        )
        var kernel = benchmark.run[twopass_kernel_once](
            max_iters=measurement_iterations
        )
        var d2h = benchmark.run[d2h_once](max_iters=measurement_iterations)

        score_h2d_once()
        twopass_kernel_once()
        d2h_once()

        var score_delta_max_abs = Float64(0.0)
        for score_index in range(candidate_score_count):
            var reference_score = py_reference_scores[score_index]
            var score_delta = abs(
                Float64(score_host[score_index]) - Float64(reference_score)
            )
            if score_delta > score_delta_max_abs:
                score_delta_max_abs = score_delta

        var py_scores = Python.list()
        for score_index in range(candidate_score_count):
            py_scores.append(Python.float(score_host[score_index]))

        var py_result = Python.list()
        py_result.append(Python.float(prepare_h2d.mean()))
        py_result.append(Python.float(score_h2d.mean()))
        py_result.append(Python.float(kernel.mean()))
        py_result.append(Python.float(d2h.mean()))
        py_result.append(Python.float(score_delta_max_abs))
        py_result.append(Python.int(candidate_score_count))
        py_result.append(py_scores)
        return py_result


def score_i8_prepared_payload_session_addresses(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_address = Int(py=py_request[0])
    var token_codes_address = Int(py=py_request[1])
    var token_scales_address = Int(py=py_request[2])
    var doc_offsets_address = Int(py=py_request[3])
    var candidate_positions_address = Int(py=py_request[4])
    var reference_scores_address = Int(py=py_request[5])
    var query_count = Int(py=py_request[6])
    var query_vector_count = Int(py=py_request[7])
    var document_count = Int(py=py_request[8])
    var document_vector_count = Int(py=py_request[9])
    var candidate_k = Int(py=py_request[10])
    if query_address == 0:
        raise Error("query address must be non-zero")
    if token_codes_address == 0:
        raise Error("token_codes address must be non-zero")
    if token_scales_address == 0:
        raise Error("token_scales address must be non-zero")
    if doc_offsets_address == 0:
        raise Error("doc_offsets address must be non-zero")
    if candidate_positions_address == 0:
        raise Error("candidate_positions address must be non-zero")
    if reference_scores_address == 0:
        raise Error("reference_scores address must be non-zero")
    if query_count <= 0:
        raise Error("query_count must be positive")
    if query_vector_count <= 0:
        raise Error("query_vector_count must be positive")
    if document_count <= 0:
        raise Error("document_count must be positive")
    if document_vector_count <= 0:
        raise Error("document_vector_count must be positive")
    if candidate_k <= 0:
        raise Error("candidate_k must be positive")

    var py_query_values = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=query_address
    )
    var py_token_codes = UnsafePointer[Int8, MutAnyOrigin](
        unsafe_from_address=token_codes_address
    )
    var py_token_scales = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=token_scales_address
    )
    var py_doc_offsets = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=doc_offsets_address
    )
    var py_candidate_positions = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=candidate_positions_address
    )
    var py_reference_scores = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=reference_scores_address
    )

    var total_document_vector_count = document_count * document_vector_count
    var query_value_count = query_count * query_vector_count * VECTOR_DIM
    var token_code_count = total_document_vector_count * VECTOR_DIM
    var doc_offset_count = document_count + 1
    var candidate_score_count = query_count * candidate_k
    var partial_score_count = (
        candidate_score_count * query_vector_count * document_vector_count
    )
    var grid_x = (candidate_score_count + BLOCK_SIZE - 1) // BLOCK_SIZE
    var partial_grid_x = (partial_score_count + BLOCK_SIZE - 1) // BLOCK_SIZE

    with DeviceContext() as ctx:
        var query_device = ctx.enqueue_create_buffer[DType.float32](
            query_value_count
        )
        var codes_device = ctx.enqueue_create_buffer[DType.int8](
            token_code_count
        )
        var scales_device = ctx.enqueue_create_buffer[DType.float32](
            total_document_vector_count
        )
        var offsets_device = ctx.enqueue_create_buffer[DType.int64](
            doc_offset_count
        )
        var candidate_device = ctx.enqueue_create_buffer[DType.int64](
            candidate_score_count
        )
        var partial_score_device = ctx.enqueue_create_buffer[DType.float32](
            partial_score_count
        )
        var score_device = ctx.enqueue_create_buffer[DType.float32](
            candidate_score_count
        )

        var query_host = ctx.enqueue_create_host_buffer[DType.float32](
            query_value_count
        )
        var codes_host = ctx.enqueue_create_host_buffer[DType.int8](
            token_code_count
        )
        var scales_host = ctx.enqueue_create_host_buffer[DType.float32](
            total_document_vector_count
        )
        var offsets_host = ctx.enqueue_create_host_buffer[DType.int64](
            doc_offset_count
        )
        var candidate_host = ctx.enqueue_create_host_buffer[DType.int64](
            candidate_score_count
        )
        var score_host = ctx.enqueue_create_host_buffer[DType.float32](
            candidate_score_count
        )

        for index in range(query_value_count):
            query_host[index] = py_query_values[index]

        for index in range(token_code_count):
            codes_host[index] = py_token_codes[index]

        for index in range(total_document_vector_count):
            scales_host[index] = py_token_scales[index]

        for index in range(doc_offset_count):
            offsets_host[index] = py_doc_offsets[index]

        for index in range(candidate_score_count):
            var candidate_position = py_candidate_positions[index]
            if candidate_position < 0 or candidate_position >= Int64(
                document_count
            ):
                raise Error(
                    "candidate position outside prepared index document range"
                )
            candidate_host[index] = candidate_position
            score_host[index] = Float32(0.0)

        query_device.enqueue_copy_from(query_host)
        codes_device.enqueue_copy_from(codes_host)
        scales_device.enqueue_copy_from(scales_host)
        offsets_device.enqueue_copy_from(offsets_host)
        candidate_device.enqueue_copy_from(candidate_host)
        ctx.synchronize()

        ctx.enqueue_function[
            score_i8_candidate_token_kernel_dynamic,
            score_i8_candidate_token_kernel_dynamic,
        ](
            query_device,
            codes_device,
            scales_device,
            offsets_device,
            candidate_device,
            partial_score_device,
            query_vector_count,
            document_vector_count,
            candidate_k,
            partial_score_count,
            grid_dim=partial_grid_x,
            block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[
            reduce_i8_candidate_score_kernel_dynamic,
            reduce_i8_candidate_score_kernel_dynamic,
        ](
            partial_score_device,
            score_device,
            query_vector_count,
            document_vector_count,
            candidate_score_count,
            grid_dim=grid_x,
            block_dim=BLOCK_SIZE,
        )
        ctx.synchronize()

        score_device.enqueue_copy_to(score_host)
        ctx.synchronize()

        var score_delta_max_abs = Float64(0.0)
        for score_index in range(candidate_score_count):
            var reference_score = py_reference_scores[score_index]
            var score_delta = abs(
                Float64(score_host[score_index]) - Float64(reference_score)
            )
            if score_delta > score_delta_max_abs:
                score_delta_max_abs = score_delta

        var py_scores = Python.list()
        for score_index in range(candidate_score_count):
            py_scores.append(Python.float(score_host[score_index]))

        var py_result = Python.list()
        py_result.append(Python.float(score_delta_max_abs))
        py_result.append(Python.int(candidate_score_count))
        py_result.append(py_scores)
        return py_result


@export
def PyInit__mojo_gpu_i8_rerank_bindings() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojo_gpu_i8_rerank_bindings")
        module.def_function[gpu_extension_device_probe](
            "gpu_extension_device_probe",
            docstring=(
                "Verify that the GPU-targeted Mojo Python extension can create"
                " a device context."
            ),
        )
        module.def_function[score_i8_real_payload_once](
            "score_i8_real_payload_once",
            docstring=(
                "Score one real Kayak i8 candidate window with a GPU-targeted"
                " Mojo extension."
            ),
        )
        module.def_function[profile_i8_prepared_payload_session](
            "profile_i8_prepared_payload_session",
            docstring=(
                "Profile a session that keeps real Kayak i8 payload buffers"
                " resident on the GPU while scoring query candidate windows."
            ),
        )
        module.def_function[profile_i8_prepared_payload_session_addresses](
            "profile_i8_prepared_payload_session_addresses",
            docstring=(
                "Profile a resident-index GPU i8 session from typed host"
                " buffer addresses."
            ),
        )
        module.def_function[score_i8_prepared_payload_session_addresses](
            "score_i8_prepared_payload_session_addresses",
            docstring=(
                "Score one resident-index GPU i8 session from typed host"
                " buffer addresses without internal benchmarks."
            ),
        )
        return module.finalize()
    except e:
        abort(String("error creating kayak mojo gpu i8 rerank bindings:", e))
