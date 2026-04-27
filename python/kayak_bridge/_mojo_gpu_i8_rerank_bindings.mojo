import std.benchmark as benchmark
from std.gpu import global_idx
from std.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.math import abs
from std.memory import alloc
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


def score_i8_centroid_scores_kernel(
    query_values: UnsafePointer[Float32, MutAnyOrigin],
    token_codes: UnsafePointer[Int8, MutAnyOrigin],
    token_scales: UnsafePointer[Float32, MutAnyOrigin],
    centroid_token_indices: UnsafePointer[Int64, MutAnyOrigin],
    output_centroid_scores: UnsafePointer[Float32, MutAnyOrigin],
    query_vector_count: Int,
    centroid_count: Int,
    centroid_score_count: Int,
):
    var score_index = Int(global_idx.x)
    if score_index >= centroid_score_count:
        return

    var query_vector_global = score_index // centroid_count
    var centroid_index = score_index - query_vector_global * centroid_count
    var query_index = query_vector_global // query_vector_count
    var query_vector_index = (
        query_vector_global - query_index * query_vector_count
    )
    var token_index = Int(centroid_token_indices[centroid_index])
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

    output_centroid_scores[score_index] = dot * token_scales[token_index]


def score_position_ranks_before_float32_device(
    score: Float32,
    position: Int,
    other_score: Float32,
    other_position: Int,
) -> Bool:
    if score > other_score:
        return True
    if score < other_score:
        return False
    return position < other_position


def score_position_is_worse_float32_device(
    score: Float32,
    position: Int,
    other_score: Float32,
    other_position: Int,
) -> Bool:
    if score < other_score:
        return True
    if score > other_score:
        return False
    return position > other_position


def centroid_position_already_selected_device(
    selected_positions: UnsafePointer[Int64, MutAnyOrigin],
    selected_base: Int,
    rank: Int,
    centroid_index: Int,
) -> Bool:
    for selected_rank in range(rank):
        if (
            Int(selected_positions[selected_base + selected_rank])
            == centroid_index
        ):
            return True
    return False


def select_i8_centroid_scores_kernel(
    centroid_scores: UnsafePointer[Float32, MutAnyOrigin],
    selected_positions: UnsafePointer[Int64, MutAnyOrigin],
    selected_scores: UnsafePointer[Float32, MutAnyOrigin],
    query_vector_global_count: Int,
    centroid_count: Int,
    centroids_per_query_vector: Int,
):
    var query_vector_global = Int(global_idx.x)
    if query_vector_global >= query_vector_global_count:
        return

    var score_base = query_vector_global * centroid_count
    var selected_base = query_vector_global * centroids_per_query_vector
    for rank in range(centroids_per_query_vector):
        var best_position = 0
        var best_score = Float32(-3.4028234663852886e38)
        var seen = False
        for centroid_index in range(centroid_count):
            if centroid_position_already_selected_device(
                selected_positions, selected_base, rank, centroid_index
            ):
                continue
            var score = centroid_scores[score_base + centroid_index]
            if not seen or score_position_ranks_before_float32_device(
                score,
                centroid_index,
                best_score,
                best_position,
            ):
                best_score = score
                best_position = centroid_index
                seen = True
        selected_positions[selected_base + rank] = Int64(best_position)
        selected_scores[selected_base + rank] = best_score


def swap_centroid_selection_heap_entries_device(
    heap_positions: UnsafePointer[Int64, MutAnyOrigin],
    heap_scores: UnsafePointer[Float32, MutAnyOrigin],
    heap_base: Int,
    lhs: Int,
    rhs: Int,
):
    var lhs_index = heap_base + lhs
    var rhs_index = heap_base + rhs
    var position_value = heap_positions[lhs_index]
    heap_positions[lhs_index] = heap_positions[rhs_index]
    heap_positions[rhs_index] = position_value

    var score_value = heap_scores[lhs_index]
    heap_scores[lhs_index] = heap_scores[rhs_index]
    heap_scores[rhs_index] = score_value


def sift_up_centroid_selection_heap_device(
    heap_positions: UnsafePointer[Int64, MutAnyOrigin],
    heap_scores: UnsafePointer[Float32, MutAnyOrigin],
    heap_base: Int,
    position: Int,
):
    var cursor = position
    while cursor > 0:
        var parent = (cursor - 1) // 2
        if not score_position_is_worse_float32_device(
            heap_scores[heap_base + cursor],
            Int(heap_positions[heap_base + cursor]),
            heap_scores[heap_base + parent],
            Int(heap_positions[heap_base + parent]),
        ):
            break

        swap_centroid_selection_heap_entries_device(
            heap_positions, heap_scores, heap_base, cursor, parent
        )
        cursor = parent


def sift_down_centroid_selection_heap_device(
    heap_positions: UnsafePointer[Int64, MutAnyOrigin],
    heap_scores: UnsafePointer[Float32, MutAnyOrigin],
    heap_base: Int,
    position: Int,
    heap_size: Int,
):
    var cursor = position
    while True:
        var left = cursor * 2 + 1
        if left >= heap_size:
            break

        var next = left
        var right = left + 1
        if right < heap_size and score_position_is_worse_float32_device(
            heap_scores[heap_base + right],
            Int(heap_positions[heap_base + right]),
            heap_scores[heap_base + left],
            Int(heap_positions[heap_base + left]),
        ):
            next = right

        if not score_position_is_worse_float32_device(
            heap_scores[heap_base + next],
            Int(heap_positions[heap_base + next]),
            heap_scores[heap_base + cursor],
            Int(heap_positions[heap_base + cursor]),
        ):
            break

        swap_centroid_selection_heap_entries_device(
            heap_positions, heap_scores, heap_base, cursor, next
        )
        cursor = next


def insert_centroid_selection_heap_device(
    heap_positions: UnsafePointer[Int64, MutAnyOrigin],
    heap_scores: UnsafePointer[Float32, MutAnyOrigin],
    heap_base: Int,
    position: Int,
    score: Float32,
    heap_size: Int,
    heap_capacity: Int,
) -> Int:
    if heap_size < heap_capacity:
        heap_positions[heap_base + heap_size] = Int64(position)
        heap_scores[heap_base + heap_size] = score
        sift_up_centroid_selection_heap_device(
            heap_positions, heap_scores, heap_base, heap_size
        )
        return heap_size + 1

    if not score_position_ranks_before_float32_device(
        score,
        position,
        heap_scores[heap_base],
        Int(heap_positions[heap_base]),
    ):
        return heap_size

    heap_positions[heap_base] = Int64(position)
    heap_scores[heap_base] = score
    sift_down_centroid_selection_heap_device(
        heap_positions, heap_scores, heap_base, 0, heap_size
    )
    return heap_size


def pop_centroid_selection_heap_device(
    heap_positions: UnsafePointer[Int64, MutAnyOrigin],
    heap_scores: UnsafePointer[Float32, MutAnyOrigin],
    heap_base: Int,
    heap_size: Int,
) -> Int:
    var next_size = heap_size - 1
    heap_positions[heap_base] = heap_positions[heap_base + next_size]
    heap_scores[heap_base] = heap_scores[heap_base + next_size]
    if next_size > 0:
        sift_down_centroid_selection_heap_device(
            heap_positions, heap_scores, heap_base, 0, next_size
        )
    return next_size


def select_i8_centroid_scores_heap_kernel(
    centroid_scores: UnsafePointer[Float32, MutAnyOrigin],
    heap_positions: UnsafePointer[Int64, MutAnyOrigin],
    heap_scores: UnsafePointer[Float32, MutAnyOrigin],
    selected_positions: UnsafePointer[Int64, MutAnyOrigin],
    selected_scores: UnsafePointer[Float32, MutAnyOrigin],
    query_vector_global_count: Int,
    centroid_count: Int,
    centroids_per_query_vector: Int,
):
    var query_vector_global = Int(global_idx.x)
    if query_vector_global >= query_vector_global_count:
        return

    var score_base = query_vector_global * centroid_count
    var heap_base = query_vector_global * centroids_per_query_vector
    var heap_size = 0
    for centroid_index in range(centroid_count):
        heap_size = insert_centroid_selection_heap_device(
            heap_positions,
            heap_scores,
            heap_base,
            centroid_index,
            centroid_scores[score_base + centroid_index],
            heap_size,
            centroids_per_query_vector,
        )

    while heap_size > 0:
        var output_offset = heap_base + heap_size - 1
        selected_positions[output_offset] = heap_positions[heap_base]
        selected_scores[output_offset] = heap_scores[heap_base]
        heap_size = pop_centroid_selection_heap_device(
            heap_positions, heap_scores, heap_base, heap_size
        )


def expand_i8_selected_centroid_postings_kernel(
    selected_centroid_positions: UnsafePointer[Int64, MutAnyOrigin],
    selected_centroid_scores: UnsafePointer[Float32, MutAnyOrigin],
    centroid_doc_offsets: UnsafePointer[Int64, MutAnyOrigin],
    centroid_doc_indices: UnsafePointer[Int64, MutAnyOrigin],
    selected_posting_offsets: UnsafePointer[Int64, MutAnyOrigin],
    output_doc_indices: UnsafePointer[Int64, MutAnyOrigin],
    output_scores: UnsafePointer[Float32, MutAnyOrigin],
    selected_centroid_count: Int,
):
    var selected_index = Int(global_idx.x)
    if selected_index >= selected_centroid_count:
        return

    var centroid_position = Int(selected_centroid_positions[selected_index])
    var posting_start = Int(centroid_doc_offsets[centroid_position])
    var posting_stop = Int(centroid_doc_offsets[centroid_position + 1])
    var output_start = Int(selected_posting_offsets[selected_index])
    var selected_score = selected_centroid_scores[selected_index]

    for posting_index in range(posting_start, posting_stop):
        var output_index = output_start + posting_index - posting_start
        output_doc_indices[output_index] = centroid_doc_indices[posting_index]
        output_scores[output_index] = selected_score


def posting_contains_document(
    centroid_doc_indices: UnsafePointer[Int64, MutAnyOrigin],
    posting_start: Int,
    posting_stop: Int,
    document_index: Int,
) -> Bool:
    var low = posting_start
    var high = posting_stop
    while low < high:
        var mid = (low + high) // 2
        var mid_document = Int(centroid_doc_indices[mid])
        if mid_document < document_index:
            low = mid + 1
        else:
            high = mid

    return (
        low < posting_stop and Int(centroid_doc_indices[low]) == document_index
    )


def reduce_i8_selected_centroid_best_scores_kernel(
    best_scores_by_query_vector: UnsafePointer[Float32, MutAnyOrigin],
    output_document_scores: UnsafePointer[Float32, MutAnyOrigin],
    query_vector_count: Int,
    document_count: Int,
    score_count: Int,
):
    var score_index = Int(global_idx.x)
    if score_index >= score_count:
        return

    var query_index = score_index // document_count
    var document_index = score_index - query_index * document_count
    var best_base = query_index * query_vector_count * document_count
    var total_score = Float32(0.0)
    for query_vector_index in range(query_vector_count):
        var best_score = best_scores_by_query_vector[
            best_base + query_vector_index * document_count + document_index
        ]
        if best_score > Float32(-3.0e38):
            total_score += best_score

    output_document_scores[score_index] = total_score


def document_position_already_selected_device(
    topk_positions: UnsafePointer[Int64, MutAnyOrigin],
    topk_base: Int,
    rank: Int,
    document_index: Int,
) -> Bool:
    for selected_rank in range(rank):
        if Int(topk_positions[topk_base + selected_rank]) == document_index:
            return True
    return False


def select_i8_document_topk_kernel(
    document_scores: UnsafePointer[Float32, MutAnyOrigin],
    topk_positions: UnsafePointer[Int64, MutAnyOrigin],
    topk_scores: UnsafePointer[Float32, MutAnyOrigin],
    query_count: Int,
    document_count: Int,
    top_k: Int,
):
    var query_index = Int(global_idx.x)
    if query_index >= query_count:
        return

    var score_base = query_index * document_count
    var topk_base = query_index * top_k
    for rank in range(top_k):
        var best_position = 0
        var best_score = Float32(-3.4028234663852886e38)
        var seen = False
        for document_index in range(document_count):
            if document_position_already_selected_device(
                topk_positions, topk_base, rank, document_index
            ):
                continue
            var score = document_scores[score_base + document_index]
            if not seen or score_position_ranks_before_float32_device(
                score,
                document_index,
                best_score,
                best_position,
            ):
                best_score = score
                best_position = document_index
                seen = True
        topk_positions[topk_base + rank] = Int64(best_position)
        topk_scores[topk_base + rank] = best_score


def accumulate_i8_selected_centroid_scores_by_query_vector_document_kernel(
    selected_centroid_positions: UnsafePointer[Int64, MutAnyOrigin],
    selected_centroid_scores: UnsafePointer[Float32, MutAnyOrigin],
    centroid_doc_offsets: UnsafePointer[Int64, MutAnyOrigin],
    centroid_doc_indices: UnsafePointer[Int64, MutAnyOrigin],
    best_scores_by_query_vector: UnsafePointer[Float32, MutAnyOrigin],
    query_vector_count: Int,
    centroids_per_query_vector: Int,
    document_count: Int,
    best_score_count: Int,
):
    var best_score_index = Int(global_idx.x)
    if best_score_index >= best_score_count:
        return

    var document_index = best_score_index % document_count
    var query_vector_global = best_score_index // document_count
    var query_index = query_vector_global // query_vector_count
    var query_vector_index = (
        query_vector_global - query_index * query_vector_count
    )
    var selected_base = (
        query_index * query_vector_count + query_vector_index
    ) * centroids_per_query_vector
    var seen = False
    var best_score = Float32(0.0)

    for selected_offset in range(centroids_per_query_vector):
        var selected_index = selected_base + selected_offset
        var centroid_position = Int(selected_centroid_positions[selected_index])
        var posting_start = Int(centroid_doc_offsets[centroid_position])
        var posting_stop = Int(centroid_doc_offsets[centroid_position + 1])
        if posting_contains_document(
            centroid_doc_indices,
            posting_start,
            posting_stop,
            document_index,
        ):
            var centroid_score = selected_centroid_scores[selected_index]
            if not seen or centroid_score > best_score:
                best_score = centroid_score
                seen = True

    if seen:
        best_scores_by_query_vector[best_score_index] = best_score
    else:
        best_scores_by_query_vector[best_score_index] = Float32(0.0)


struct PreparedGpuI8AddressSession(Movable):
    var ctx: DeviceContext
    var codes_device: DeviceBuffer[DType.int8]
    var scales_device: DeviceBuffer[DType.float32]
    var offsets_device: DeviceBuffer[DType.int64]
    var query_device: DeviceBuffer[DType.float32]
    var candidate_device: DeviceBuffer[DType.int64]
    var partial_score_device: DeviceBuffer[DType.float32]
    var score_device: DeviceBuffer[DType.float32]
    var query_host: HostBuffer[DType.float32]
    var candidate_host: HostBuffer[DType.int64]
    var score_host: HostBuffer[DType.float32]
    var document_count: Int
    var document_vector_count: Int
    var query_count: Int
    var query_vector_count: Int
    var candidate_k: Int
    var query_value_count: Int
    var candidate_score_count: Int
    var partial_score_count: Int
    var grid_x: Int
    var partial_grid_x: Int

    def __init__(
        out self,
        py_token_codes: UnsafePointer[Int8, MutAnyOrigin],
        py_token_scales: UnsafePointer[Float32, MutAnyOrigin],
        py_doc_offsets: UnsafePointer[Int64, MutAnyOrigin],
        document_count: Int,
        document_vector_count: Int,
        query_count: Int,
        query_vector_count: Int,
        candidate_k: Int,
    ) raises:
        self.ctx = DeviceContext()
        self.document_count = document_count
        self.document_vector_count = document_vector_count
        self.query_count = query_count
        self.query_vector_count = query_vector_count
        self.candidate_k = candidate_k

        var total_document_vector_count = document_count * document_vector_count
        var token_code_count = total_document_vector_count * VECTOR_DIM
        var doc_offset_count = document_count + 1
        self.query_value_count = query_count * query_vector_count * VECTOR_DIM
        self.candidate_score_count = query_count * candidate_k
        self.partial_score_count = (
            self.candidate_score_count
            * query_vector_count
            * document_vector_count
        )
        self.grid_x = (
            self.candidate_score_count + BLOCK_SIZE - 1
        ) // BLOCK_SIZE
        self.partial_grid_x = (
            self.partial_score_count + BLOCK_SIZE - 1
        ) // BLOCK_SIZE

        self.codes_device = self.ctx.enqueue_create_buffer[DType.int8](
            token_code_count
        )
        self.scales_device = self.ctx.enqueue_create_buffer[DType.float32](
            total_document_vector_count
        )
        self.offsets_device = self.ctx.enqueue_create_buffer[DType.int64](
            doc_offset_count
        )
        self.query_device = self.ctx.enqueue_create_buffer[DType.float32](
            self.query_value_count
        )
        self.candidate_device = self.ctx.enqueue_create_buffer[DType.int64](
            self.candidate_score_count
        )
        self.partial_score_device = self.ctx.enqueue_create_buffer[
            DType.float32
        ](self.partial_score_count)
        self.score_device = self.ctx.enqueue_create_buffer[DType.float32](
            self.candidate_score_count
        )
        self.query_host = self.ctx.enqueue_create_host_buffer[DType.float32](
            self.query_value_count
        )
        self.candidate_host = self.ctx.enqueue_create_host_buffer[DType.int64](
            self.candidate_score_count
        )
        self.score_host = self.ctx.enqueue_create_host_buffer[DType.float32](
            self.candidate_score_count
        )

        var codes_host = self.ctx.enqueue_create_host_buffer[DType.int8](
            token_code_count
        )
        var scales_host = self.ctx.enqueue_create_host_buffer[DType.float32](
            total_document_vector_count
        )
        var offsets_host = self.ctx.enqueue_create_host_buffer[DType.int64](
            doc_offset_count
        )
        self.ctx.synchronize()

        for index in range(token_code_count):
            codes_host[index] = py_token_codes[index]
        for index in range(total_document_vector_count):
            scales_host[index] = py_token_scales[index]
        for index in range(doc_offset_count):
            offsets_host[index] = py_doc_offsets[index]

        self.codes_device.enqueue_copy_from(codes_host)
        self.scales_device.enqueue_copy_from(scales_host)
        self.offsets_device.enqueue_copy_from(offsets_host)
        self.ctx.synchronize()

    def run_score_window(
        mut self,
        py_query_values: UnsafePointer[Float32, MutAnyOrigin],
        py_candidate_positions: UnsafePointer[Int64, MutAnyOrigin],
    ) raises:
        for index in range(self.query_value_count):
            self.query_host[index] = py_query_values[index]

        for index in range(self.candidate_score_count):
            var candidate_position = py_candidate_positions[index]
            if candidate_position < 0 or candidate_position >= Int64(
                self.document_count
            ):
                raise Error(
                    "candidate position outside prepared index document range"
                )
            self.candidate_host[index] = candidate_position

        self.query_device.enqueue_copy_from(self.query_host)
        self.candidate_device.enqueue_copy_from(self.candidate_host)
        self.ctx.synchronize()

        self.ctx.enqueue_function[
            score_i8_candidate_token_kernel_dynamic,
            score_i8_candidate_token_kernel_dynamic,
        ](
            self.query_device,
            self.codes_device,
            self.scales_device,
            self.offsets_device,
            self.candidate_device,
            self.partial_score_device,
            self.query_vector_count,
            self.document_vector_count,
            self.candidate_k,
            self.partial_score_count,
            grid_dim=self.partial_grid_x,
            block_dim=BLOCK_SIZE,
        )
        self.ctx.enqueue_function[
            reduce_i8_candidate_score_kernel_dynamic,
            reduce_i8_candidate_score_kernel_dynamic,
        ](
            self.partial_score_device,
            self.score_device,
            self.query_vector_count,
            self.document_vector_count,
            self.candidate_score_count,
            grid_dim=self.grid_x,
            block_dim=BLOCK_SIZE,
        )
        self.ctx.synchronize()

        self.score_device.enqueue_copy_to(self.score_host)
        self.ctx.synchronize()

    def score_window(
        mut self,
        py_query_values: UnsafePointer[Float32, MutAnyOrigin],
        py_candidate_positions: UnsafePointer[Int64, MutAnyOrigin],
        py_reference_scores: UnsafePointer[Float32, MutAnyOrigin],
    ) raises -> PythonObject:
        self.run_score_window(py_query_values, py_candidate_positions)

        var score_delta_max_abs = Float64(0.0)
        var py_scores = Python.list()
        for score_index in range(self.candidate_score_count):
            var reference_score = py_reference_scores[score_index]
            var score_delta = abs(
                Float64(self.score_host[score_index]) - Float64(reference_score)
            )
            if score_delta > score_delta_max_abs:
                score_delta_max_abs = score_delta
            py_scores.append(Python.float(self.score_host[score_index]))

        var py_result = Python.list()
        py_result.append(Python.float(score_delta_max_abs))
        py_result.append(Python.int(self.candidate_score_count))
        py_result.append(py_scores)
        return py_result

    def score_window_topk(
        mut self,
        py_query_values: UnsafePointer[Float32, MutAnyOrigin],
        py_candidate_positions: UnsafePointer[Int64, MutAnyOrigin],
        py_reference_scores: UnsafePointer[Float32, MutAnyOrigin],
        top_k: Int,
    ) raises -> PythonObject:
        if top_k <= 0:
            raise Error("top_k must be positive")
        if top_k > self.candidate_k:
            raise Error("top_k must not exceed candidate_k")
        self.run_score_window(py_query_values, py_candidate_positions)

        var score_delta_max_abs = Float64(0.0)
        for score_index in range(self.candidate_score_count):
            var reference_score = py_reference_scores[score_index]
            var score_delta = abs(
                Float64(self.score_host[score_index]) - Float64(reference_score)
            )
            if score_delta > score_delta_max_abs:
                score_delta_max_abs = score_delta

        var py_positions = Python.list()
        var py_scores = Python.list()

        for query_index in range(self.query_count):
            var query_base = query_index * self.candidate_k
            for rank in range(top_k):
                var best_candidate_index = 0
                var best_score = Float32(-3.4028234663852886e38)
                for candidate_index in range(self.candidate_k):
                    var score_index = query_base + candidate_index
                    var score = self.score_host[score_index]
                    if score > best_score:
                        best_score = score
                        best_candidate_index = candidate_index

                var best_score_index = query_base + best_candidate_index
                var best_position = self.candidate_host[best_score_index]
                py_positions.append(Python.int(best_position))
                py_scores.append(Python.float(best_score))
                self.score_host[best_score_index] = Float32(
                    -3.4028234663852886e38
                )

        var py_result = Python.list()
        py_result.append(Python.float(score_delta_max_abs))
        py_result.append(Python.int(self.candidate_score_count))
        py_result.append(Python.int(top_k))
        py_result.append(py_positions)
        py_result.append(py_scores)
        return py_result

    def score_window_topk_no_reference(
        mut self,
        py_query_values: UnsafePointer[Float32, MutAnyOrigin],
        py_candidate_positions: UnsafePointer[Int64, MutAnyOrigin],
        top_k: Int,
    ) raises -> PythonObject:
        if top_k <= 0:
            raise Error("top_k must be positive")
        if top_k > self.candidate_k:
            raise Error("top_k must not exceed candidate_k")
        self.run_score_window(py_query_values, py_candidate_positions)

        var py_positions = Python.list()
        var py_scores = Python.list()

        for query_index in range(self.query_count):
            var query_base = query_index * self.candidate_k
            for rank in range(top_k):
                var best_candidate_index = 0
                var best_score = Float32(-3.4028234663852886e38)
                for candidate_index in range(self.candidate_k):
                    var score_index = query_base + candidate_index
                    var score = self.score_host[score_index]
                    if score > best_score:
                        best_score = score
                        best_candidate_index = candidate_index

                var best_score_index = query_base + best_candidate_index
                var best_position = self.candidate_host[best_score_index]
                py_positions.append(Python.int(best_position))
                py_scores.append(Python.float(best_score))
                self.score_host[best_score_index] = Float32(
                    -3.4028234663852886e38
                )

        var py_result = Python.list()
        py_result.append(Python.int(self.candidate_score_count))
        py_result.append(Python.int(top_k))
        py_result.append(py_positions)
        py_result.append(py_scores)
        return py_result


struct PreparedGpuI8FusedCentroidPostingSession(Movable):
    var ctx: DeviceContext
    var token_codes_device: DeviceBuffer[DType.int8]
    var token_scales_device: DeviceBuffer[DType.float32]
    var centroid_token_indices_device: DeviceBuffer[DType.int64]
    var centroid_doc_offsets_device: DeviceBuffer[DType.int64]
    var centroid_doc_indices_device: DeviceBuffer[DType.int64]
    var query_device: DeviceBuffer[DType.float32]
    var centroid_scores_device: DeviceBuffer[DType.float32]
    var selection_heap_positions_device: DeviceBuffer[DType.int64]
    var selection_heap_scores_device: DeviceBuffer[DType.float32]
    var selected_positions_device: DeviceBuffer[DType.int64]
    var selected_scores_device: DeviceBuffer[DType.float32]
    var best_scores_device: DeviceBuffer[DType.float32]
    var document_scores_device: DeviceBuffer[DType.float32]
    var query_host: HostBuffer[DType.float32]
    var document_scores_host: HostBuffer[DType.float32]
    var topk_positions_host: HostBuffer[DType.int64]
    var topk_scores_host: HostBuffer[DType.float32]
    var total_vector_count: Int
    var centroid_count: Int
    var posting_count: Int
    var document_count: Int
    var query_count: Int
    var query_vector_count: Int
    var centroids_per_query_vector: Int
    var top_k: Int
    var query_value_count: Int
    var query_vector_global_count: Int
    var centroid_score_count: Int
    var selected_centroid_count: Int
    var document_score_count: Int
    var best_score_count: Int
    var topk_position_count: Int
    var centroid_score_grid_x: Int
    var document_score_grid_x: Int
    var best_score_grid_x: Int

    def __init__(
        out self,
        py_token_codes: UnsafePointer[Int8, MutAnyOrigin],
        py_token_scales: UnsafePointer[Float32, MutAnyOrigin],
        py_centroid_token_indices: UnsafePointer[Int64, MutAnyOrigin],
        py_centroid_doc_offsets: UnsafePointer[Int64, MutAnyOrigin],
        py_centroid_doc_indices: UnsafePointer[Int64, MutAnyOrigin],
        total_vector_count: Int,
        centroid_count: Int,
        posting_count: Int,
        document_count: Int,
        query_count: Int,
        query_vector_count: Int,
        centroids_per_query_vector: Int,
        top_k: Int,
    ) raises:
        self.ctx = DeviceContext()
        self.total_vector_count = total_vector_count
        self.centroid_count = centroid_count
        self.posting_count = posting_count
        self.document_count = document_count
        self.query_count = query_count
        self.query_vector_count = query_vector_count
        self.centroids_per_query_vector = centroids_per_query_vector
        self.top_k = top_k

        var token_code_count = total_vector_count * VECTOR_DIM
        var centroid_offset_count = centroid_count + 1
        self.query_value_count = query_count * query_vector_count * VECTOR_DIM
        self.query_vector_global_count = query_count * query_vector_count
        self.centroid_score_count = (
            self.query_vector_global_count * centroid_count
        )
        self.selected_centroid_count = (
            self.query_vector_global_count * centroids_per_query_vector
        )
        self.document_score_count = query_count * document_count
        self.best_score_count = self.query_vector_global_count * document_count
        self.topk_position_count = query_count * top_k
        self.centroid_score_grid_x = (
            self.centroid_score_count + BLOCK_SIZE - 1
        ) // BLOCK_SIZE
        self.document_score_grid_x = (
            self.document_score_count + BLOCK_SIZE - 1
        ) // BLOCK_SIZE
        self.best_score_grid_x = (
            self.best_score_count + BLOCK_SIZE - 1
        ) // BLOCK_SIZE

        self.token_codes_device = self.ctx.enqueue_create_buffer[DType.int8](
            token_code_count
        )
        self.token_scales_device = self.ctx.enqueue_create_buffer[
            DType.float32
        ](total_vector_count)
        self.centroid_token_indices_device = self.ctx.enqueue_create_buffer[
            DType.int64
        ](centroid_count)
        self.centroid_doc_offsets_device = self.ctx.enqueue_create_buffer[
            DType.int64
        ](centroid_offset_count)
        self.centroid_doc_indices_device = self.ctx.enqueue_create_buffer[
            DType.int64
        ](posting_count)
        self.query_device = self.ctx.enqueue_create_buffer[DType.float32](
            self.query_value_count
        )
        self.centroid_scores_device = self.ctx.enqueue_create_buffer[
            DType.float32
        ](self.centroid_score_count)
        self.selection_heap_positions_device = self.ctx.enqueue_create_buffer[
            DType.int64
        ](self.selected_centroid_count)
        self.selection_heap_scores_device = self.ctx.enqueue_create_buffer[
            DType.float32
        ](self.selected_centroid_count)
        self.selected_positions_device = self.ctx.enqueue_create_buffer[
            DType.int64
        ](self.selected_centroid_count)
        self.selected_scores_device = self.ctx.enqueue_create_buffer[
            DType.float32
        ](self.selected_centroid_count)
        self.best_scores_device = self.ctx.enqueue_create_buffer[DType.float32](
            self.best_score_count
        )
        self.document_scores_device = self.ctx.enqueue_create_buffer[
            DType.float32
        ](self.document_score_count)
        self.query_host = self.ctx.enqueue_create_host_buffer[DType.float32](
            self.query_value_count
        )
        self.document_scores_host = self.ctx.enqueue_create_host_buffer[
            DType.float32
        ](self.document_score_count)
        self.topk_positions_host = self.ctx.enqueue_create_host_buffer[
            DType.int64
        ](self.topk_position_count)
        self.topk_scores_host = self.ctx.enqueue_create_host_buffer[
            DType.float32
        ](self.topk_position_count)

        var token_codes_host = self.ctx.enqueue_create_host_buffer[DType.int8](
            token_code_count
        )
        var token_scales_host = self.ctx.enqueue_create_host_buffer[
            DType.float32
        ](total_vector_count)
        var centroid_token_indices_host = self.ctx.enqueue_create_host_buffer[
            DType.int64
        ](centroid_count)
        var centroid_doc_offsets_host = self.ctx.enqueue_create_host_buffer[
            DType.int64
        ](centroid_offset_count)
        var centroid_doc_indices_host = self.ctx.enqueue_create_host_buffer[
            DType.int64
        ](posting_count)
        self.ctx.synchronize()

        for index in range(token_code_count):
            token_codes_host[index] = py_token_codes[index]
        for index in range(total_vector_count):
            token_scales_host[index] = py_token_scales[index]
        for index in range(centroid_count):
            centroid_token_indices_host[index] = py_centroid_token_indices[
                index
            ]
        for index in range(centroid_offset_count):
            centroid_doc_offsets_host[index] = py_centroid_doc_offsets[index]
        for index in range(posting_count):
            centroid_doc_indices_host[index] = py_centroid_doc_indices[index]

        self.token_codes_device.enqueue_copy_from(token_codes_host)
        self.token_scales_device.enqueue_copy_from(token_scales_host)
        self.centroid_token_indices_device.enqueue_copy_from(
            centroid_token_indices_host
        )
        self.centroid_doc_offsets_device.enqueue_copy_from(
            centroid_doc_offsets_host
        )
        self.centroid_doc_indices_device.enqueue_copy_from(
            centroid_doc_indices_host
        )
        self.ctx.synchronize()

    def ingest_query(
        mut self,
        py_query_values: UnsafePointer[Float32, MutAnyOrigin],
    ):
        for index in range(self.query_value_count):
            self.query_host[index] = py_query_values[index]

    def copy_query_to_device(mut self) raises:
        self.query_device.enqueue_copy_from(self.query_host)
        self.ctx.synchronize()

    def enqueue_centroid_score_kernel(mut self) raises:
        self.ctx.enqueue_function[
            score_i8_centroid_scores_kernel,
            score_i8_centroid_scores_kernel,
        ](
            self.query_device,
            self.token_codes_device,
            self.token_scales_device,
            self.centroid_token_indices_device,
            self.centroid_scores_device,
            self.query_vector_count,
            self.centroid_count,
            self.centroid_score_count,
            grid_dim=self.centroid_score_grid_x,
            block_dim=BLOCK_SIZE,
        )

    def run_centroid_score_kernel(mut self) raises:
        self.enqueue_centroid_score_kernel()
        self.ctx.synchronize()

    def enqueue_centroid_selection_kernel(mut self) raises:
        self.ctx.enqueue_function[
            select_i8_centroid_scores_heap_kernel,
            select_i8_centroid_scores_heap_kernel,
        ](
            self.centroid_scores_device,
            self.selection_heap_positions_device,
            self.selection_heap_scores_device,
            self.selected_positions_device,
            self.selected_scores_device,
            self.query_vector_global_count,
            self.centroid_count,
            self.centroids_per_query_vector,
            grid_dim=self.query_vector_global_count,
            block_dim=1,
        )

    def run_centroid_selection_kernel(mut self) raises:
        self.enqueue_centroid_selection_kernel()
        self.ctx.synchronize()

    def enqueue_accumulation_kernel(mut self) raises:
        self.ctx.enqueue_function[
            accumulate_i8_selected_centroid_scores_by_query_vector_document_kernel,
            accumulate_i8_selected_centroid_scores_by_query_vector_document_kernel,
        ](
            self.selected_positions_device,
            self.selected_scores_device,
            self.centroid_doc_offsets_device,
            self.centroid_doc_indices_device,
            self.best_scores_device,
            self.query_vector_count,
            self.centroids_per_query_vector,
            self.document_count,
            self.best_score_count,
            grid_dim=self.best_score_grid_x,
            block_dim=BLOCK_SIZE,
        )

    def run_accumulation_kernel(mut self) raises:
        self.enqueue_accumulation_kernel()
        self.ctx.synchronize()

    def enqueue_reduction_kernel(mut self) raises:
        self.ctx.enqueue_function[
            reduce_i8_selected_centroid_best_scores_kernel,
            reduce_i8_selected_centroid_best_scores_kernel,
        ](
            self.best_scores_device,
            self.document_scores_device,
            self.query_vector_count,
            self.document_count,
            self.document_score_count,
            grid_dim=self.document_score_grid_x,
            block_dim=BLOCK_SIZE,
        )

    def run_reduction_kernel(mut self) raises:
        self.enqueue_reduction_kernel()
        self.ctx.synchronize()

    def copy_document_scores_to_host(mut self) raises:
        self.document_scores_device.enqueue_copy_to(self.document_scores_host)
        self.ctx.synchronize()

    def run_window(
        mut self,
        py_query_values: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        self.ingest_query(py_query_values)
        self.copy_query_to_device()
        self.enqueue_centroid_score_kernel()
        self.enqueue_centroid_selection_kernel()
        self.enqueue_accumulation_kernel()
        self.enqueue_reduction_kernel()
        self.ctx.synchronize()
        self.copy_document_scores_to_host()
        self.host_topk_destructive()

    def topk_position_already_selected(
        self, query_topk_base: Int, rank: Int, document_index: Int
    ) -> Bool:
        for selected_rank in range(rank):
            if (
                Int(self.topk_positions_host[query_topk_base + selected_rank])
                == document_index
            ):
                return True
        return False

    def host_topk(mut self):
        for query_index in range(self.query_count):
            var score_base = query_index * self.document_count
            var query_topk_base = query_index * self.top_k
            for rank in range(self.top_k):
                var best_position = 0
                var best_score = Float32(-3.4028234663852886e38)
                var seen = False
                for document_index in range(self.document_count):
                    if self.topk_position_already_selected(
                        query_topk_base, rank, document_index
                    ):
                        continue
                    var score = self.document_scores_host[
                        score_base + document_index
                    ]
                    if not seen or score_position_ranks_before_float32_device(
                        score,
                        document_index,
                        best_score,
                        best_position,
                    ):
                        best_score = score
                        best_position = document_index
                        seen = True
                self.topk_positions_host[query_topk_base + rank] = Int64(
                    best_position
                )
                self.topk_scores_host[query_topk_base + rank] = best_score

    def host_topk_destructive(mut self):
        for query_index in range(self.query_count):
            var score_base = query_index * self.document_count
            var query_topk_base = query_index * self.top_k
            for rank in range(self.top_k):
                var best_position = 0
                var best_score = Float32(-3.4028234663852886e38)
                var seen = False
                for document_index in range(self.document_count):
                    var score_index = score_base + document_index
                    var score = self.document_scores_host[score_index]
                    if not seen or score_position_ranks_before_float32_device(
                        score,
                        document_index,
                        best_score,
                        best_position,
                    ):
                        best_score = score
                        best_position = document_index
                        seen = True
                self.topk_positions_host[query_topk_base + rank] = Int64(
                    best_position
                )
                self.topk_scores_host[query_topk_base + rank] = best_score
                self.document_scores_host[score_base + best_position] = Float32(
                    -3.4028234663852886e38
                )

    def score_topk_no_reference(
        mut self,
        py_query_values: UnsafePointer[Float32, MutAnyOrigin],
    ) raises -> PythonObject:
        self.run_window(py_query_values)

        var py_positions = Python.list()
        var py_scores = Python.list()
        for index in range(self.topk_position_count):
            py_positions.append(Python.int(self.topk_positions_host[index]))
            py_scores.append(Python.float(self.topk_scores_host[index]))

        var py_result = Python.list()
        py_result.append(Python.int(self.document_score_count))
        py_result.append(Python.int(self.top_k))
        py_result.append(py_positions)
        py_result.append(py_scores)
        return py_result

    def profile_topk_no_reference(
        mut self,
        py_query_values: UnsafePointer[Float32, MutAnyOrigin],
        warmup_iterations: Int,
        measurement_iterations: Int,
    ) raises -> PythonObject:
        if warmup_iterations < 0:
            raise Error("warmup_iterations must be non-negative")
        if measurement_iterations <= 0:
            raise Error("measurement_iterations must be positive")

        var document_scores_backup_host = self.ctx.enqueue_create_host_buffer[
            DType.float32
        ](self.document_score_count)
        self.ctx.synchronize()

        def ingest_query_once() capturing:
            self.ingest_query(py_query_values)

        def query_h2d_once() capturing raises:
            self.copy_query_to_device()

        def centroid_score_once() capturing raises:
            self.run_centroid_score_kernel()

        def centroid_selection_once() capturing raises:
            self.run_centroid_selection_kernel()

        def accumulation_once() capturing raises:
            self.run_accumulation_kernel()

        def reduction_once() capturing raises:
            self.run_reduction_kernel()

        def d2h_once() capturing raises:
            self.copy_document_scores_to_host()

        def snapshot_scores_once() capturing:
            for index in range(self.document_score_count):
                document_scores_backup_host[index] = self.document_scores_host[
                    index
                ]

        def restore_scores_once() capturing:
            for index in range(self.document_score_count):
                self.document_scores_host[index] = document_scores_backup_host[
                    index
                ]

        def host_topk_once() capturing:
            self.host_topk()

        def restore_and_destructive_host_topk_once() capturing:
            restore_scores_once()
            self.host_topk_destructive()

        def full_score_once() capturing raises:
            ingest_query_once()
            query_h2d_once()
            centroid_score_once()
            centroid_selection_once()
            accumulation_once()
            reduction_once()
            d2h_once()
            snapshot_scores_once()
            restore_and_destructive_host_topk_once()

        full_score_once()
        for _ in range(warmup_iterations):
            full_score_once()

        var ingest_query = benchmark.run[ingest_query_once](
            max_iters=measurement_iterations
        )
        ingest_query_once()
        var query_h2d = benchmark.run[query_h2d_once](
            max_iters=measurement_iterations
        )
        query_h2d_once()
        var centroid_score = benchmark.run[centroid_score_once](
            max_iters=measurement_iterations
        )
        centroid_score_once()
        var centroid_selection = benchmark.run[centroid_selection_once](
            max_iters=measurement_iterations
        )
        centroid_selection_once()
        var accumulation = benchmark.run[accumulation_once](
            max_iters=measurement_iterations
        )
        accumulation_once()
        var reduction = benchmark.run[reduction_once](
            max_iters=measurement_iterations
        )
        reduction_once()
        var d2h = benchmark.run[d2h_once](max_iters=measurement_iterations)
        d2h_once()
        snapshot_scores_once()
        var host_topk_restore = benchmark.run[restore_scores_once](
            max_iters=measurement_iterations
        )
        restore_scores_once()
        var host_topk_restore_plus_destructive = benchmark.run[
            restore_and_destructive_host_topk_once
        ](max_iters=measurement_iterations)
        restore_and_destructive_host_topk_once()
        restore_scores_once()
        var host_topk_non_destructive = benchmark.run[host_topk_once](
            max_iters=measurement_iterations
        )
        host_topk_once()

        var destructive_host_topk_estimated = (
            host_topk_restore_plus_destructive.mean() - host_topk_restore.mean()
        )
        if destructive_host_topk_estimated < 0:
            destructive_host_topk_estimated = 0

        var py_positions = Python.list()
        var py_scores = Python.list()
        for index in range(self.topk_position_count):
            py_positions.append(Python.int(self.topk_positions_host[index]))
            py_scores.append(Python.float(self.topk_scores_host[index]))

        var py_result = Python.list()
        py_result.append(Python.float(ingest_query.mean()))
        py_result.append(Python.float(query_h2d.mean()))
        py_result.append(Python.float(centroid_score.mean()))
        py_result.append(Python.float(centroid_selection.mean()))
        py_result.append(Python.float(accumulation.mean()))
        py_result.append(Python.float(reduction.mean()))
        py_result.append(Python.float(d2h.mean()))
        py_result.append(Python.float(host_topk_restore.mean()))
        py_result.append(
            Python.float(host_topk_restore_plus_destructive.mean())
        )
        py_result.append(Python.float(destructive_host_topk_estimated))
        py_result.append(Python.float(host_topk_non_destructive.mean()))
        py_result.append(Python.int(self.document_score_count))
        py_result.append(Python.int(self.top_k))
        py_result.append(py_positions)
        py_result.append(py_scores)
        return py_result


def validate_address_session_shape(
    document_count: Int,
    document_vector_count: Int,
    query_count: Int,
    query_vector_count: Int,
    candidate_k: Int,
) raises:
    if document_count <= 0:
        raise Error("document_count must be positive")
    if document_vector_count <= 0:
        raise Error("document_vector_count must be positive")
    if query_count <= 0:
        raise Error("query_count must be positive")
    if query_vector_count <= 0:
        raise Error("query_vector_count must be positive")
    if candidate_k <= 0:
        raise Error("candidate_k must be positive")
    if candidate_k > document_count:
        raise Error("candidate_k must not exceed document_count")


def validate_fused_centroid_posting_session_shape(
    total_vector_count: Int,
    centroid_count: Int,
    posting_count: Int,
    document_count: Int,
    query_count: Int,
    query_vector_count: Int,
    centroids_per_query_vector: Int,
    top_k: Int,
) raises:
    if total_vector_count <= 0:
        raise Error("total_vector_count must be positive")
    if centroid_count <= 0:
        raise Error("centroid_count must be positive")
    if posting_count <= 0:
        raise Error("posting_count must be positive")
    if document_count <= 0:
        raise Error("document_count must be positive")
    if query_count <= 0:
        raise Error("query_count must be positive")
    if query_vector_count <= 0:
        raise Error("query_vector_count must be positive")
    if centroids_per_query_vector <= 0:
        raise Error("centroids_per_query_vector must be positive")
    if centroids_per_query_vector > centroid_count:
        raise Error("centroids_per_query_vector must not exceed centroid_count")
    if top_k <= 0:
        raise Error("top_k must be positive")
    if top_k > document_count:
        raise Error("top_k must not exceed document_count")


def prepare_i8_address_session_handle(
    py_request: PythonObject,
) raises -> PythonObject:
    var token_codes_address = Int(py=py_request[0])
    var token_scales_address = Int(py=py_request[1])
    var doc_offsets_address = Int(py=py_request[2])
    var document_count = Int(py=py_request[3])
    var document_vector_count = Int(py=py_request[4])
    var query_count = Int(py=py_request[5])
    var query_vector_count = Int(py=py_request[6])
    var candidate_k = Int(py=py_request[7])
    if token_codes_address == 0:
        raise Error("token_codes address must be non-zero")
    if token_scales_address == 0:
        raise Error("token_scales address must be non-zero")
    if doc_offsets_address == 0:
        raise Error("doc_offsets address must be non-zero")
    validate_address_session_shape(
        document_count,
        document_vector_count,
        query_count,
        query_vector_count,
        candidate_k,
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
    var session = alloc[PreparedGpuI8AddressSession](1)
    session.init_pointee_move(
        PreparedGpuI8AddressSession(
            py_token_codes,
            py_token_scales,
            py_doc_offsets,
            document_count,
            document_vector_count,
            query_count,
            query_vector_count,
            candidate_k,
        )
    )
    return Python.int(session.__int__())


def score_i8_address_session_handle(
    py_request: PythonObject,
) raises -> PythonObject:
    var handle = Int(py=py_request[0])
    var query_address = Int(py=py_request[1])
    var candidate_positions_address = Int(py=py_request[2])
    var reference_scores_address = Int(py=py_request[3])
    if handle == 0:
        raise Error("prepared GPU i8 address session handle must be non-zero")
    if query_address == 0:
        raise Error("query address must be non-zero")
    if candidate_positions_address == 0:
        raise Error("candidate_positions address must be non-zero")
    if reference_scores_address == 0:
        raise Error("reference_scores address must be non-zero")

    var session = UnsafePointer[PreparedGpuI8AddressSession, MutAnyOrigin](
        unsafe_from_address=handle
    )
    var py_query_values = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=query_address
    )
    var py_candidate_positions = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=candidate_positions_address
    )
    var py_reference_scores = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=reference_scores_address
    )
    return session[].score_window(
        py_query_values,
        py_candidate_positions,
        py_reference_scores,
    )


def score_i8_address_session_handle_topk(
    py_request: PythonObject,
) raises -> PythonObject:
    var handle = Int(py=py_request[0])
    var query_address = Int(py=py_request[1])
    var candidate_positions_address = Int(py=py_request[2])
    var reference_scores_address = Int(py=py_request[3])
    var top_k = Int(py=py_request[4])
    if handle == 0:
        raise Error("prepared GPU i8 address session handle must be non-zero")
    if query_address == 0:
        raise Error("query address must be non-zero")
    if candidate_positions_address == 0:
        raise Error("candidate_positions address must be non-zero")
    if reference_scores_address == 0:
        raise Error("reference_scores address must be non-zero")

    var session = UnsafePointer[PreparedGpuI8AddressSession, MutAnyOrigin](
        unsafe_from_address=handle
    )
    var py_query_values = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=query_address
    )
    var py_candidate_positions = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=candidate_positions_address
    )
    var py_reference_scores = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=reference_scores_address
    )
    return session[].score_window_topk(
        py_query_values,
        py_candidate_positions,
        py_reference_scores,
        top_k,
    )


def score_i8_address_session_handle_topk_no_reference(
    py_request: PythonObject,
) raises -> PythonObject:
    var handle = Int(py=py_request[0])
    var query_address = Int(py=py_request[1])
    var candidate_positions_address = Int(py=py_request[2])
    var top_k = Int(py=py_request[3])
    if handle == 0:
        raise Error("prepared GPU i8 address session handle must be non-zero")
    if query_address == 0:
        raise Error("query address must be non-zero")
    if candidate_positions_address == 0:
        raise Error("candidate_positions address must be non-zero")

    var session = UnsafePointer[PreparedGpuI8AddressSession, MutAnyOrigin](
        unsafe_from_address=handle
    )
    var py_query_values = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=query_address
    )
    var py_candidate_positions = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=candidate_positions_address
    )
    return session[].score_window_topk_no_reference(
        py_query_values,
        py_candidate_positions,
        top_k,
    )


def release_i8_address_session_handle(
    py_handle: PythonObject,
) raises -> PythonObject:
    var handle = Int(py=py_handle)
    if handle == 0:
        raise Error("prepared GPU i8 address session handle must be non-zero")
    var session = UnsafePointer[PreparedGpuI8AddressSession, MutAnyOrigin](
        unsafe_from_address=handle
    )
    session[].ctx.synchronize()
    session.destroy_pointee()
    session.free()
    return Python.str("released")


def prepare_i8_fused_centroid_posting_session_handle(
    py_request: PythonObject,
) raises -> PythonObject:
    var token_codes_address = Int(py=py_request[0])
    var token_scales_address = Int(py=py_request[1])
    var centroid_token_indices_address = Int(py=py_request[2])
    var centroid_doc_offsets_address = Int(py=py_request[3])
    var centroid_doc_indices_address = Int(py=py_request[4])
    var total_vector_count = Int(py=py_request[5])
    var centroid_count = Int(py=py_request[6])
    var posting_count = Int(py=py_request[7])
    var document_count = Int(py=py_request[8])
    var query_count = Int(py=py_request[9])
    var query_vector_count = Int(py=py_request[10])
    var centroids_per_query_vector = Int(py=py_request[11])
    var top_k = Int(py=py_request[12])
    if token_codes_address == 0:
        raise Error("token_codes address must be non-zero")
    if token_scales_address == 0:
        raise Error("token_scales address must be non-zero")
    if centroid_token_indices_address == 0:
        raise Error("centroid_token_indices address must be non-zero")
    if centroid_doc_offsets_address == 0:
        raise Error("centroid_doc_offsets address must be non-zero")
    if centroid_doc_indices_address == 0:
        raise Error("centroid_doc_indices address must be non-zero")
    validate_fused_centroid_posting_session_shape(
        total_vector_count,
        centroid_count,
        posting_count,
        document_count,
        query_count,
        query_vector_count,
        centroids_per_query_vector,
        top_k,
    )

    var py_token_codes = UnsafePointer[Int8, MutAnyOrigin](
        unsafe_from_address=token_codes_address
    )
    var py_token_scales = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=token_scales_address
    )
    var py_centroid_token_indices = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=centroid_token_indices_address
    )
    var py_centroid_doc_offsets = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=centroid_doc_offsets_address
    )
    var py_centroid_doc_indices = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=centroid_doc_indices_address
    )
    var session = alloc[PreparedGpuI8FusedCentroidPostingSession](1)
    session.init_pointee_move(
        PreparedGpuI8FusedCentroidPostingSession(
            py_token_codes,
            py_token_scales,
            py_centroid_token_indices,
            py_centroid_doc_offsets,
            py_centroid_doc_indices,
            total_vector_count,
            centroid_count,
            posting_count,
            document_count,
            query_count,
            query_vector_count,
            centroids_per_query_vector,
            top_k,
        )
    )
    return Python.int(session.__int__())


def score_i8_fused_centroid_posting_session_handle_topk_no_reference(
    py_request: PythonObject,
) raises -> PythonObject:
    var handle = Int(py=py_request[0])
    var query_address = Int(py=py_request[1])
    if handle == 0:
        raise Error(
            "prepared GPU i8 fused centroid-posting handle must be non-zero"
        )
    if query_address == 0:
        raise Error("query address must be non-zero")

    var session = UnsafePointer[
        PreparedGpuI8FusedCentroidPostingSession, MutAnyOrigin
    ](unsafe_from_address=handle)
    var py_query_values = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=query_address
    )
    return session[].score_topk_no_reference(py_query_values)


def profile_i8_fused_centroid_posting_session_handle_topk_no_reference(
    py_request: PythonObject,
) raises -> PythonObject:
    var handle = Int(py=py_request[0])
    var query_address = Int(py=py_request[1])
    var warmup_iterations = Int(py=py_request[2])
    var measurement_iterations = Int(py=py_request[3])
    if handle == 0:
        raise Error(
            "prepared GPU i8 fused centroid-posting handle must be non-zero"
        )
    if query_address == 0:
        raise Error("query address must be non-zero")

    var session = UnsafePointer[
        PreparedGpuI8FusedCentroidPostingSession, MutAnyOrigin
    ](unsafe_from_address=handle)
    var py_query_values = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=query_address
    )
    return session[].profile_topk_no_reference(
        py_query_values,
        warmup_iterations,
        measurement_iterations,
    )


def release_i8_fused_centroid_posting_session_handle(
    py_handle: PythonObject,
) raises -> PythonObject:
    var handle = Int(py=py_handle)
    if handle == 0:
        raise Error(
            "prepared GPU i8 fused centroid-posting handle must be non-zero"
        )
    var session = UnsafePointer[
        PreparedGpuI8FusedCentroidPostingSession, MutAnyOrigin
    ](unsafe_from_address=handle)
    session[].ctx.synchronize()
    session.destroy_pointee()
    session.free()
    return Python.str("released")


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


def score_i8_prepared_payload_session_addresses_repeated(
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
    var session_iterations = Int(py=py_request[11])
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
    if session_iterations <= 0:
        raise Error("session_iterations must be positive")

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

        codes_device.enqueue_copy_from(codes_host)
        scales_device.enqueue_copy_from(scales_host)
        offsets_device.enqueue_copy_from(offsets_host)
        ctx.synchronize()

        for _ in range(session_iterations):
            query_device.enqueue_copy_from(query_host)
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
        py_result.append(Python.int(session_iterations))
        py_result.append(py_scores)
        return py_result


def score_i8_prepared_payload_session_addresses_multi_window(
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
    var window_count = Int(py=py_request[11])
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
    if window_count <= 0:
        raise Error("window_count must be positive")

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

        for index in range(token_code_count):
            codes_host[index] = py_token_codes[index]

        for index in range(total_document_vector_count):
            scales_host[index] = py_token_scales[index]

        for index in range(doc_offset_count):
            offsets_host[index] = py_doc_offsets[index]

        codes_device.enqueue_copy_from(codes_host)
        scales_device.enqueue_copy_from(scales_host)
        offsets_device.enqueue_copy_from(offsets_host)
        ctx.synchronize()

        var score_delta_max_abs = Float64(0.0)
        var py_scores = Python.list()
        for window_index in range(window_count):
            var query_window_offset = window_index * query_value_count
            var candidate_window_offset = window_index * candidate_score_count

            for index in range(query_value_count):
                query_host[index] = py_query_values[query_window_offset + index]

            for index in range(candidate_score_count):
                var candidate_position = py_candidate_positions[
                    candidate_window_offset + index
                ]
                if candidate_position < 0 or candidate_position >= Int64(
                    document_count
                ):
                    raise Error(
                        "candidate position outside prepared index document"
                        " range"
                    )
                candidate_host[index] = candidate_position

            query_device.enqueue_copy_from(query_host)
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

            for score_index in range(candidate_score_count):
                var reference_score = py_reference_scores[
                    candidate_window_offset + score_index
                ]
                var score_delta = abs(
                    Float64(score_host[score_index]) - Float64(reference_score)
                )
                if score_delta > score_delta_max_abs:
                    score_delta_max_abs = score_delta
                py_scores.append(Python.float(score_host[score_index]))

        var py_result = Python.list()
        py_result.append(Python.float(score_delta_max_abs))
        py_result.append(Python.int(candidate_score_count))
        py_result.append(Python.int(window_count))
        py_result.append(py_scores)
        return py_result


def profile_i8_candidate_generation_payload_addresses(
    py_request: PythonObject,
) raises -> PythonObject:
    var centroid_token_indices_address = Int(py=py_request[0])
    var centroid_doc_offsets_address = Int(py=py_request[1])
    var centroid_doc_indices_address = Int(py=py_request[2])
    var centroid_count = Int(py=py_request[3])
    var posting_count = Int(py=py_request[4])
    var document_count = Int(py=py_request[5])
    var warmup_iterations = Int(py=py_request[6])
    var measurement_iterations = Int(py=py_request[7])
    if centroid_token_indices_address == 0:
        raise Error("centroid_token_indices address must be non-zero")
    if centroid_doc_offsets_address == 0:
        raise Error("centroid_doc_offsets address must be non-zero")
    if centroid_doc_indices_address == 0:
        raise Error("centroid_doc_indices address must be non-zero")
    if centroid_count <= 0:
        raise Error("centroid_count must be positive")
    if posting_count <= 0:
        raise Error("posting_count must be positive")
    if document_count <= 0:
        raise Error("document_count must be positive")
    if warmup_iterations < 0:
        raise Error("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise Error("measurement_iterations must be positive")

    var py_centroid_token_indices = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=centroid_token_indices_address
    )
    var py_centroid_doc_offsets = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=centroid_doc_offsets_address
    )
    var py_centroid_doc_indices = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=centroid_doc_indices_address
    )
    var offset_count = centroid_count + 1

    with DeviceContext() as ctx:
        var token_indices_host = ctx.enqueue_create_host_buffer[DType.int64](
            centroid_count
        )
        var doc_offsets_host = ctx.enqueue_create_host_buffer[DType.int64](
            offset_count
        )
        var doc_indices_host = ctx.enqueue_create_host_buffer[DType.int64](
            posting_count
        )
        var token_indices_readback = ctx.enqueue_create_host_buffer[
            DType.int64
        ](centroid_count)
        var doc_offsets_readback = ctx.enqueue_create_host_buffer[DType.int64](
            offset_count
        )
        var doc_indices_readback = ctx.enqueue_create_host_buffer[DType.int64](
            posting_count
        )
        var token_indices_device = ctx.enqueue_create_buffer[DType.int64](
            centroid_count
        )
        var doc_offsets_device = ctx.enqueue_create_buffer[DType.int64](
            offset_count
        )
        var doc_indices_device = ctx.enqueue_create_buffer[DType.int64](
            posting_count
        )
        ctx.synchronize()

        def host_ingest_once() capturing raises:
            for index in range(centroid_count):
                token_indices_host[index] = py_centroid_token_indices[index]

            for index in range(offset_count):
                doc_offsets_host[index] = py_centroid_doc_offsets[index]

            for index in range(posting_count):
                doc_indices_host[index] = py_centroid_doc_indices[index]

        def h2d_once() capturing raises:
            token_indices_device.enqueue_copy_from(token_indices_host)
            doc_offsets_device.enqueue_copy_from(doc_offsets_host)
            doc_indices_device.enqueue_copy_from(doc_indices_host)
            ctx.synchronize()

        def d2h_once() capturing raises:
            token_indices_device.enqueue_copy_to(token_indices_readback)
            doc_offsets_device.enqueue_copy_to(doc_offsets_readback)
            doc_indices_device.enqueue_copy_to(doc_indices_readback)
            ctx.synchronize()

        host_ingest_once()
        h2d_once()
        d2h_once()

        for _ in range(warmup_iterations):
            host_ingest_once()
            h2d_once()
            d2h_once()

        var host_ingest = benchmark.run[host_ingest_once](
            max_iters=measurement_iterations
        )
        host_ingest_once()
        var h2d = benchmark.run[h2d_once](max_iters=measurement_iterations)
        h2d_once()
        var d2h = benchmark.run[d2h_once](max_iters=measurement_iterations)
        d2h_once()

        var mismatch_count = 0
        for index in range(centroid_count):
            if (
                token_indices_readback[index]
                != py_centroid_token_indices[index]
            ):
                mismatch_count += 1

        for index in range(offset_count):
            if doc_offsets_readback[index] != py_centroid_doc_offsets[index]:
                mismatch_count += 1

        for index in range(posting_count):
            if doc_indices_readback[index] != py_centroid_doc_indices[index]:
                mismatch_count += 1

        var monotonic_offset_violation_count = 0
        if doc_offsets_readback[0] != 0:
            monotonic_offset_violation_count += 1
        for index in range(1, offset_count):
            if doc_offsets_readback[index] < doc_offsets_readback[index - 1]:
                monotonic_offset_violation_count += 1
        if doc_offsets_readback[offset_count - 1] != Int64(posting_count):
            monotonic_offset_violation_count += 1

        var doc_index_out_of_range_count = 0
        for index in range(posting_count):
            var doc_index = doc_indices_readback[index]
            if doc_index < 0 or doc_index >= Int64(document_count):
                doc_index_out_of_range_count += 1

        var py_result = Python.list()
        py_result.append(Python.float(host_ingest.mean()))
        py_result.append(Python.float(h2d.mean()))
        py_result.append(Python.float(d2h.mean()))
        py_result.append(Python.int(mismatch_count))
        py_result.append(Python.int(monotonic_offset_violation_count))
        py_result.append(Python.int(doc_index_out_of_range_count))
        py_result.append(Python.int(centroid_count))
        py_result.append(Python.int(posting_count))
        py_result.append(Python.int(document_count))
        return py_result


def profile_i8_selected_posting_traversal_addresses(
    py_request: PythonObject,
) raises -> PythonObject:
    var centroid_doc_offsets_address = Int(py=py_request[0])
    var centroid_doc_indices_address = Int(py=py_request[1])
    var selected_positions_address = Int(py=py_request[2])
    var selected_scores_address = Int(py=py_request[3])
    var selected_posting_offsets_address = Int(py=py_request[4])
    var expected_doc_indices_address = Int(py=py_request[5])
    var expected_scores_address = Int(py=py_request[6])
    var centroid_count = Int(py=py_request[7])
    var posting_count = Int(py=py_request[8])
    var document_count = Int(py=py_request[9])
    var selected_centroid_count = Int(py=py_request[10])
    var expanded_posting_count = Int(py=py_request[11])
    var warmup_iterations = Int(py=py_request[12])
    var measurement_iterations = Int(py=py_request[13])
    if centroid_doc_offsets_address == 0:
        raise Error("centroid_doc_offsets address must be non-zero")
    if centroid_doc_indices_address == 0:
        raise Error("centroid_doc_indices address must be non-zero")
    if selected_positions_address == 0:
        raise Error("selected positions address must be non-zero")
    if selected_scores_address == 0:
        raise Error("selected scores address must be non-zero")
    if selected_posting_offsets_address == 0:
        raise Error("selected posting offsets address must be non-zero")
    if expected_doc_indices_address == 0:
        raise Error("expected doc indices address must be non-zero")
    if expected_scores_address == 0:
        raise Error("expected scores address must be non-zero")
    if centroid_count <= 0:
        raise Error("centroid_count must be positive")
    if posting_count <= 0:
        raise Error("posting_count must be positive")
    if document_count <= 0:
        raise Error("document_count must be positive")
    if selected_centroid_count <= 0:
        raise Error("selected_centroid_count must be positive")
    if expanded_posting_count <= 0:
        raise Error("expanded_posting_count must be positive")
    if warmup_iterations < 0:
        raise Error("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise Error("measurement_iterations must be positive")

    var py_centroid_doc_offsets = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=centroid_doc_offsets_address
    )
    var py_centroid_doc_indices = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=centroid_doc_indices_address
    )
    var py_selected_positions = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=selected_positions_address
    )
    var py_selected_scores = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=selected_scores_address
    )
    var py_selected_posting_offsets = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=selected_posting_offsets_address
    )
    var py_expected_doc_indices = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=expected_doc_indices_address
    )
    var py_expected_scores = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=expected_scores_address
    )
    var centroid_offset_count = centroid_count + 1
    var selected_offset_count = selected_centroid_count + 1
    var grid_x = (selected_centroid_count + BLOCK_SIZE - 1) // BLOCK_SIZE

    with DeviceContext() as ctx:
        var centroid_doc_offsets_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](centroid_offset_count)
        var centroid_doc_indices_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](posting_count)
        var selected_positions_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](selected_centroid_count)
        var selected_scores_host = ctx.enqueue_create_host_buffer[
            DType.float32
        ](selected_centroid_count)
        var selected_posting_offsets_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](selected_offset_count)
        var output_doc_indices_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](expanded_posting_count)
        var output_scores_host = ctx.enqueue_create_host_buffer[DType.float32](
            expanded_posting_count
        )

        var centroid_doc_offsets_device = ctx.enqueue_create_buffer[
            DType.int64
        ](centroid_offset_count)
        var centroid_doc_indices_device = ctx.enqueue_create_buffer[
            DType.int64
        ](posting_count)
        var selected_positions_device = ctx.enqueue_create_buffer[DType.int64](
            selected_centroid_count
        )
        var selected_scores_device = ctx.enqueue_create_buffer[DType.float32](
            selected_centroid_count
        )
        var selected_posting_offsets_device = ctx.enqueue_create_buffer[
            DType.int64
        ](selected_offset_count)
        var output_doc_indices_device = ctx.enqueue_create_buffer[DType.int64](
            expanded_posting_count
        )
        var output_scores_device = ctx.enqueue_create_buffer[DType.float32](
            expanded_posting_count
        )
        ctx.synchronize()

        def host_ingest_once() capturing raises:
            for index in range(centroid_offset_count):
                centroid_doc_offsets_host[index] = py_centroid_doc_offsets[
                    index
                ]

            for index in range(posting_count):
                centroid_doc_indices_host[index] = py_centroid_doc_indices[
                    index
                ]

            for index in range(selected_centroid_count):
                selected_positions_host[index] = py_selected_positions[index]
                selected_scores_host[index] = py_selected_scores[index]

            for index in range(selected_offset_count):
                selected_posting_offsets_host[
                    index
                ] = py_selected_posting_offsets[index]

        def payload_h2d_once() capturing raises:
            centroid_doc_offsets_device.enqueue_copy_from(
                centroid_doc_offsets_host
            )
            centroid_doc_indices_device.enqueue_copy_from(
                centroid_doc_indices_host
            )
            ctx.synchronize()

        def selected_h2d_once() capturing raises:
            selected_positions_device.enqueue_copy_from(selected_positions_host)
            selected_scores_device.enqueue_copy_from(selected_scores_host)
            selected_posting_offsets_device.enqueue_copy_from(
                selected_posting_offsets_host
            )
            ctx.synchronize()

        def traversal_kernel_once() capturing raises:
            ctx.enqueue_function[
                expand_i8_selected_centroid_postings_kernel,
                expand_i8_selected_centroid_postings_kernel,
            ](
                selected_positions_device,
                selected_scores_device,
                centroid_doc_offsets_device,
                centroid_doc_indices_device,
                selected_posting_offsets_device,
                output_doc_indices_device,
                output_scores_device,
                selected_centroid_count,
                grid_dim=grid_x,
                block_dim=BLOCK_SIZE,
            )
            ctx.synchronize()

        def d2h_once() capturing raises:
            output_doc_indices_device.enqueue_copy_to(output_doc_indices_host)
            output_scores_device.enqueue_copy_to(output_scores_host)
            ctx.synchronize()

        host_ingest_once()
        payload_h2d_once()
        selected_h2d_once()
        traversal_kernel_once()
        d2h_once()

        for _ in range(warmup_iterations):
            selected_h2d_once()
            traversal_kernel_once()
            d2h_once()

        var host_ingest = benchmark.run[host_ingest_once](
            max_iters=measurement_iterations
        )
        host_ingest_once()
        var payload_h2d = benchmark.run[payload_h2d_once](
            max_iters=measurement_iterations
        )
        payload_h2d_once()
        var selected_h2d = benchmark.run[selected_h2d_once](
            max_iters=measurement_iterations
        )
        selected_h2d_once()
        var kernel = benchmark.run[traversal_kernel_once](
            max_iters=measurement_iterations
        )
        traversal_kernel_once()
        var d2h = benchmark.run[d2h_once](max_iters=measurement_iterations)
        d2h_once()

        var selected_position_out_of_range_count = 0
        for index in range(selected_centroid_count):
            var centroid_position = selected_positions_host[index]
            if centroid_position < 0 or centroid_position >= Int64(
                centroid_count
            ):
                selected_position_out_of_range_count += 1

        var selected_offset_violation_count = 0
        if selected_posting_offsets_host[0] != 0:
            selected_offset_violation_count += 1
        for index in range(1, selected_offset_count):
            if (
                selected_posting_offsets_host[index]
                < selected_posting_offsets_host[index - 1]
            ):
                selected_offset_violation_count += 1
        if selected_posting_offsets_host[selected_offset_count - 1] != Int64(
            expanded_posting_count
        ):
            selected_offset_violation_count += 1

        var doc_mismatch_count = 0
        var doc_index_out_of_range_count = 0
        var score_delta_max_abs = Float64(0.0)
        for index in range(expanded_posting_count):
            var doc_index = output_doc_indices_host[index]
            if doc_index != py_expected_doc_indices[index]:
                doc_mismatch_count += 1
            if doc_index < 0 or doc_index >= Int64(document_count):
                doc_index_out_of_range_count += 1

            var score_delta = abs(
                Float64(output_scores_host[index])
                - Float64(py_expected_scores[index])
            )
            if score_delta > score_delta_max_abs:
                score_delta_max_abs = score_delta

        var py_result = Python.list()
        py_result.append(Python.float(host_ingest.mean()))
        py_result.append(Python.float(payload_h2d.mean()))
        py_result.append(Python.float(selected_h2d.mean()))
        py_result.append(Python.float(kernel.mean()))
        py_result.append(Python.float(d2h.mean()))
        py_result.append(Python.int(selected_position_out_of_range_count))
        py_result.append(Python.int(selected_offset_violation_count))
        py_result.append(Python.int(doc_mismatch_count))
        py_result.append(Python.int(doc_index_out_of_range_count))
        py_result.append(Python.float(score_delta_max_abs))
        py_result.append(Python.int(selected_centroid_count))
        py_result.append(Python.int(expanded_posting_count))
        py_result.append(Python.int(document_count))
        return py_result


def profile_i8_centroid_selection_addresses(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var token_codes_address = Int(py=py_request[1])
    var token_scales_address = Int(py=py_request[2])
    var centroid_token_indices_address = Int(py=py_request[3])
    var expected_selected_positions_address = Int(py=py_request[4])
    var expected_selected_scores_address = Int(py=py_request[5])
    var total_vector_count = Int(py=py_request[6])
    var centroid_count = Int(py=py_request[7])
    var query_count = Int(py=py_request[8])
    var query_vector_count = Int(py=py_request[9])
    var centroids_per_query_vector = Int(py=py_request[10])
    var warmup_iterations = Int(py=py_request[11])
    var measurement_iterations = Int(py=py_request[12])
    if query_values_address == 0:
        raise Error("query values address must be non-zero")
    if token_codes_address == 0:
        raise Error("token codes address must be non-zero")
    if token_scales_address == 0:
        raise Error("token scales address must be non-zero")
    if centroid_token_indices_address == 0:
        raise Error("centroid token indices address must be non-zero")
    if expected_selected_positions_address == 0:
        raise Error("expected selected positions address must be non-zero")
    if expected_selected_scores_address == 0:
        raise Error("expected selected scores address must be non-zero")
    if total_vector_count <= 0:
        raise Error("total_vector_count must be positive")
    if centroid_count <= 0:
        raise Error("centroid_count must be positive")
    if query_count <= 0:
        raise Error("query_count must be positive")
    if query_vector_count <= 0:
        raise Error("query_vector_count must be positive")
    if centroids_per_query_vector <= 0:
        raise Error("centroids_per_query_vector must be positive")
    if centroids_per_query_vector > centroid_count:
        raise Error("centroids_per_query_vector must not exceed centroid_count")
    if warmup_iterations < 0:
        raise Error("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise Error("measurement_iterations must be positive")

    var py_query_values = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=query_values_address
    )
    var py_token_codes = UnsafePointer[Int8, MutAnyOrigin](
        unsafe_from_address=token_codes_address
    )
    var py_token_scales = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=token_scales_address
    )
    var py_centroid_token_indices = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=centroid_token_indices_address
    )
    var py_expected_selected_positions = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=expected_selected_positions_address
    )
    var py_expected_selected_scores = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=expected_selected_scores_address
    )

    var query_value_count = query_count * query_vector_count * VECTOR_DIM
    var token_code_count = total_vector_count * VECTOR_DIM
    var centroid_score_count = query_count * query_vector_count * centroid_count
    var selected_centroid_count = (
        query_count * query_vector_count * centroids_per_query_vector
    )
    var grid_x = (centroid_score_count + BLOCK_SIZE - 1) // BLOCK_SIZE

    with DeviceContext() as ctx:
        var query_values_host = ctx.enqueue_create_host_buffer[DType.float32](
            query_value_count
        )
        var token_codes_host = ctx.enqueue_create_host_buffer[DType.int8](
            token_code_count
        )
        var token_scales_host = ctx.enqueue_create_host_buffer[DType.float32](
            total_vector_count
        )
        var centroid_token_indices_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](centroid_count)
        var output_centroid_scores_host = ctx.enqueue_create_host_buffer[
            DType.float32
        ](centroid_score_count)
        var output_selected_positions_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](selected_centroid_count)
        var output_selected_scores_host = ctx.enqueue_create_host_buffer[
            DType.float32
        ](selected_centroid_count)
        var selection_heap_positions_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](centroids_per_query_vector)
        var selection_heap_scores_host = ctx.enqueue_create_host_buffer[
            DType.float32
        ](centroids_per_query_vector)

        var query_values_device = ctx.enqueue_create_buffer[DType.float32](
            query_value_count
        )
        var token_codes_device = ctx.enqueue_create_buffer[DType.int8](
            token_code_count
        )
        var token_scales_device = ctx.enqueue_create_buffer[DType.float32](
            total_vector_count
        )
        var centroid_token_indices_device = ctx.enqueue_create_buffer[
            DType.int64
        ](centroid_count)
        var output_centroid_scores_device = ctx.enqueue_create_buffer[
            DType.float32
        ](centroid_score_count)
        ctx.synchronize()

        def host_ingest_once() capturing raises:
            for index in range(query_value_count):
                query_values_host[index] = py_query_values[index]

            for index in range(token_code_count):
                token_codes_host[index] = py_token_codes[index]

            for index in range(total_vector_count):
                token_scales_host[index] = py_token_scales[index]

            for index in range(centroid_count):
                centroid_token_indices_host[index] = py_centroid_token_indices[
                    index
                ]

        def payload_h2d_once() capturing raises:
            token_codes_device.enqueue_copy_from(token_codes_host)
            token_scales_device.enqueue_copy_from(token_scales_host)
            centroid_token_indices_device.enqueue_copy_from(
                centroid_token_indices_host
            )
            ctx.synchronize()

        def query_h2d_once() capturing raises:
            query_values_device.enqueue_copy_from(query_values_host)
            ctx.synchronize()

        def centroid_score_kernel_once() capturing raises:
            ctx.enqueue_function[
                score_i8_centroid_scores_kernel,
                score_i8_centroid_scores_kernel,
            ](
                query_values_device,
                token_codes_device,
                token_scales_device,
                centroid_token_indices_device,
                output_centroid_scores_device,
                query_vector_count,
                centroid_count,
                centroid_score_count,
                grid_dim=grid_x,
                block_dim=BLOCK_SIZE,
            )
            ctx.synchronize()

        def d2h_once() capturing raises:
            output_centroid_scores_device.enqueue_copy_to(
                output_centroid_scores_host
            )
            ctx.synchronize()

        def score_position_ranks_before_float32(
            score: Float32,
            position: Int,
            other_score: Float32,
            other_position: Int,
        ) -> Bool:
            if score > other_score:
                return True
            if score < other_score:
                return False
            return position < other_position

        def score_position_is_worse_float32(
            score: Float32,
            position: Int,
            other_score: Float32,
            other_position: Int,
        ) -> Bool:
            if score < other_score:
                return True
            if score > other_score:
                return False
            return position > other_position

        def swap_selection_heap_entries(lhs: Int, rhs: Int) capturing:
            var position_value = selection_heap_positions_host[lhs]
            selection_heap_positions_host[lhs] = selection_heap_positions_host[
                rhs
            ]
            selection_heap_positions_host[rhs] = position_value

            var score_value = selection_heap_scores_host[lhs]
            selection_heap_scores_host[lhs] = selection_heap_scores_host[rhs]
            selection_heap_scores_host[rhs] = score_value

        def sift_up_selection_heap(position: Int) capturing:
            var cursor = position
            while cursor > 0:
                var parent = (cursor - 1) // 2
                if not score_position_is_worse_float32(
                    selection_heap_scores_host[cursor],
                    Int(selection_heap_positions_host[cursor]),
                    selection_heap_scores_host[parent],
                    Int(selection_heap_positions_host[parent]),
                ):
                    break

                swap_selection_heap_entries(cursor, parent)
                cursor = parent

        def sift_down_selection_heap(position: Int, heap_size: Int) capturing:
            var cursor = position
            while True:
                var left = cursor * 2 + 1
                if left >= heap_size:
                    break

                var next = left
                var right = left + 1
                if right < heap_size and score_position_is_worse_float32(
                    selection_heap_scores_host[right],
                    Int(selection_heap_positions_host[right]),
                    selection_heap_scores_host[left],
                    Int(selection_heap_positions_host[left]),
                ):
                    next = right

                if not score_position_is_worse_float32(
                    selection_heap_scores_host[next],
                    Int(selection_heap_positions_host[next]),
                    selection_heap_scores_host[cursor],
                    Int(selection_heap_positions_host[cursor]),
                ):
                    break

                swap_selection_heap_entries(cursor, next)
                cursor = next

        def insert_selection_heap(
            position: Int, score: Float32, heap_size: Int
        ) capturing -> Int:
            if heap_size < centroids_per_query_vector:
                selection_heap_positions_host[heap_size] = Int64(position)
                selection_heap_scores_host[heap_size] = score
                sift_up_selection_heap(heap_size)
                return heap_size + 1

            if not score_position_ranks_before_float32(
                score,
                position,
                selection_heap_scores_host[0],
                Int(selection_heap_positions_host[0]),
            ):
                return heap_size

            selection_heap_positions_host[0] = Int64(position)
            selection_heap_scores_host[0] = score
            sift_down_selection_heap(0, heap_size)
            return heap_size

        def pop_selection_heap(heap_size: Int) capturing -> Int:
            var next_size = heap_size - 1
            selection_heap_positions_host[0] = selection_heap_positions_host[
                next_size
            ]
            selection_heap_scores_host[0] = selection_heap_scores_host[
                next_size
            ]
            if next_size > 0:
                sift_down_selection_heap(0, next_size)
            return next_size

        def host_select_once() capturing:
            for query_index in range(query_count):
                for query_vector_index in range(query_vector_count):
                    var score_base = (
                        query_index * query_vector_count * centroid_count
                        + query_vector_index * centroid_count
                    )
                    var selected_base = (
                        query_index
                        * query_vector_count
                        * centroids_per_query_vector
                        + query_vector_index * centroids_per_query_vector
                    )
                    var heap_size = 0
                    for centroid_index in range(centroid_count):
                        heap_size = insert_selection_heap(
                            centroid_index,
                            output_centroid_scores_host[
                                score_base + centroid_index
                            ],
                            heap_size,
                        )

                    while heap_size > 0:
                        var output_offset = selected_base + heap_size - 1
                        output_selected_positions_host[
                            output_offset
                        ] = selection_heap_positions_host[0]
                        output_selected_scores_host[
                            output_offset
                        ] = selection_heap_scores_host[0]
                        heap_size = pop_selection_heap(heap_size)

        host_ingest_once()
        payload_h2d_once()
        query_h2d_once()
        centroid_score_kernel_once()
        d2h_once()
        host_select_once()

        for _ in range(warmup_iterations):
            query_h2d_once()
            centroid_score_kernel_once()
            d2h_once()
            host_select_once()

        var host_ingest = benchmark.run[host_ingest_once](
            max_iters=measurement_iterations
        )
        host_ingest_once()
        var payload_h2d = benchmark.run[payload_h2d_once](
            max_iters=measurement_iterations
        )
        payload_h2d_once()
        var query_h2d = benchmark.run[query_h2d_once](
            max_iters=measurement_iterations
        )
        query_h2d_once()
        var kernel = benchmark.run[centroid_score_kernel_once](
            max_iters=measurement_iterations
        )
        centroid_score_kernel_once()
        var d2h = benchmark.run[d2h_once](max_iters=measurement_iterations)
        d2h_once()
        var host_selection = benchmark.run[host_select_once](
            max_iters=measurement_iterations
        )
        host_select_once()

        var centroid_token_out_of_range_count = 0
        for centroid_index in range(centroid_count):
            var token_index = centroid_token_indices_host[centroid_index]
            if token_index < 0 or token_index >= Int64(total_vector_count):
                centroid_token_out_of_range_count += 1

        var selected_position_mismatch_count = 0
        var selected_score_mismatch_count = 0
        var selected_score_delta_max_abs = Float64(0.0)
        for index in range(selected_centroid_count):
            if (
                output_selected_positions_host[index]
                != py_expected_selected_positions[index]
            ):
                selected_position_mismatch_count += 1
            var score_delta = abs(
                Float64(output_selected_scores_host[index])
                - Float64(py_expected_selected_scores[index])
            )
            if score_delta > selected_score_delta_max_abs:
                selected_score_delta_max_abs = score_delta
            if score_delta > Float64(0.0001):
                selected_score_mismatch_count += 1

        var py_result = Python.list()
        py_result.append(Python.float(host_ingest.mean()))
        py_result.append(Python.float(payload_h2d.mean()))
        py_result.append(Python.float(query_h2d.mean()))
        py_result.append(Python.float(kernel.mean()))
        py_result.append(Python.float(d2h.mean()))
        py_result.append(Python.float(host_selection.mean()))
        py_result.append(Python.int(centroid_token_out_of_range_count))
        py_result.append(Python.int(selected_position_mismatch_count))
        py_result.append(Python.int(selected_score_mismatch_count))
        py_result.append(Python.float(selected_score_delta_max_abs))
        py_result.append(Python.int(selected_centroid_count))
        py_result.append(Python.int(centroid_score_count))
        py_result.append(Python.int(centroid_count))
        py_result.append(Python.int(query_count))
        py_result.append(Python.int(query_vector_count))
        py_result.append(Python.int(centroids_per_query_vector))
        return py_result


def profile_i8_fused_centroid_posting_accumulation_addresses(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var token_codes_address = Int(py=py_request[1])
    var token_scales_address = Int(py=py_request[2])
    var centroid_token_indices_address = Int(py=py_request[3])
    var centroid_doc_offsets_address = Int(py=py_request[4])
    var centroid_doc_indices_address = Int(py=py_request[5])
    var expected_selected_positions_address = Int(py=py_request[6])
    var expected_selected_scores_address = Int(py=py_request[7])
    var expected_document_scores_address = Int(py=py_request[8])
    var total_vector_count = Int(py=py_request[9])
    var centroid_count = Int(py=py_request[10])
    var posting_count = Int(py=py_request[11])
    var document_count = Int(py=py_request[12])
    var query_count = Int(py=py_request[13])
    var query_vector_count = Int(py=py_request[14])
    var centroids_per_query_vector = Int(py=py_request[15])
    var top_k = Int(py=py_request[16])
    var warmup_iterations = Int(py=py_request[17])
    var measurement_iterations = Int(py=py_request[18])
    if query_values_address == 0:
        raise Error("query values address must be non-zero")
    if token_codes_address == 0:
        raise Error("token codes address must be non-zero")
    if token_scales_address == 0:
        raise Error("token scales address must be non-zero")
    if centroid_token_indices_address == 0:
        raise Error("centroid token indices address must be non-zero")
    if centroid_doc_offsets_address == 0:
        raise Error("centroid_doc_offsets address must be non-zero")
    if centroid_doc_indices_address == 0:
        raise Error("centroid_doc_indices address must be non-zero")
    if expected_selected_positions_address == 0:
        raise Error("expected selected positions address must be non-zero")
    if expected_selected_scores_address == 0:
        raise Error("expected selected scores address must be non-zero")
    if expected_document_scores_address == 0:
        raise Error("expected document scores address must be non-zero")
    if total_vector_count <= 0:
        raise Error("total_vector_count must be positive")
    if centroid_count <= 0:
        raise Error("centroid_count must be positive")
    if posting_count <= 0:
        raise Error("posting_count must be positive")
    if document_count <= 0:
        raise Error("document_count must be positive")
    if query_count <= 0:
        raise Error("query_count must be positive")
    if query_vector_count <= 0:
        raise Error("query_vector_count must be positive")
    if centroids_per_query_vector <= 0:
        raise Error("centroids_per_query_vector must be positive")
    if centroids_per_query_vector > centroid_count:
        raise Error("centroids_per_query_vector must not exceed centroid_count")
    if top_k <= 0:
        raise Error("top_k must be positive")
    if top_k > document_count:
        raise Error("top_k must not exceed document_count")
    if warmup_iterations < 0:
        raise Error("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise Error("measurement_iterations must be positive")

    var py_query_values = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=query_values_address
    )
    var py_token_codes = UnsafePointer[Int8, MutAnyOrigin](
        unsafe_from_address=token_codes_address
    )
    var py_token_scales = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=token_scales_address
    )
    var py_centroid_token_indices = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=centroid_token_indices_address
    )
    var py_centroid_doc_offsets = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=centroid_doc_offsets_address
    )
    var py_centroid_doc_indices = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=centroid_doc_indices_address
    )
    var py_expected_selected_positions = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=expected_selected_positions_address
    )
    var py_expected_selected_scores = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=expected_selected_scores_address
    )
    var py_expected_document_scores = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=expected_document_scores_address
    )

    var query_value_count = query_count * query_vector_count * VECTOR_DIM
    var token_code_count = total_vector_count * VECTOR_DIM
    var centroid_offset_count = centroid_count + 1
    var query_vector_global_count = query_count * query_vector_count
    var centroid_score_count = query_vector_global_count * centroid_count
    var selected_centroid_count = (
        query_vector_global_count * centroids_per_query_vector
    )
    var score_count = query_count * document_count
    var best_score_count = query_vector_global_count * document_count
    var topk_position_count = query_count * top_k
    var centroid_score_grid_x = (
        centroid_score_count + BLOCK_SIZE - 1
    ) // BLOCK_SIZE
    var selected_grid_x = query_vector_global_count
    var score_grid_x = (score_count + BLOCK_SIZE - 1) // BLOCK_SIZE
    var best_grid_x = (best_score_count + BLOCK_SIZE - 1) // BLOCK_SIZE
    var selected_score_delta_tolerance = Float64(0.0001)
    var score_delta_tolerance = (
        Float64(query_vector_count) * selected_score_delta_tolerance
    )

    with DeviceContext() as ctx:
        var query_values_host = ctx.enqueue_create_host_buffer[DType.float32](
            query_value_count
        )
        var token_codes_host = ctx.enqueue_create_host_buffer[DType.int8](
            token_code_count
        )
        var token_scales_host = ctx.enqueue_create_host_buffer[DType.float32](
            total_vector_count
        )
        var centroid_token_indices_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](centroid_count)
        var centroid_doc_offsets_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](centroid_offset_count)
        var centroid_doc_indices_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](posting_count)
        var output_selected_positions_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](selected_centroid_count)
        var output_selected_scores_host = ctx.enqueue_create_host_buffer[
            DType.float32
        ](selected_centroid_count)
        var output_document_scores_host = ctx.enqueue_create_host_buffer[
            DType.float32
        ](score_count)
        var output_topk_positions_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](topk_position_count)
        var expected_topk_positions_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](topk_position_count)

        var query_values_device = ctx.enqueue_create_buffer[DType.float32](
            query_value_count
        )
        var token_codes_device = ctx.enqueue_create_buffer[DType.int8](
            token_code_count
        )
        var token_scales_device = ctx.enqueue_create_buffer[DType.float32](
            total_vector_count
        )
        var centroid_token_indices_device = ctx.enqueue_create_buffer[
            DType.int64
        ](centroid_count)
        var centroid_doc_offsets_device = ctx.enqueue_create_buffer[
            DType.int64
        ](centroid_offset_count)
        var centroid_doc_indices_device = ctx.enqueue_create_buffer[
            DType.int64
        ](posting_count)
        var output_centroid_scores_device = ctx.enqueue_create_buffer[
            DType.float32
        ](centroid_score_count)
        var selection_heap_positions_device = ctx.enqueue_create_buffer[
            DType.int64
        ](selected_centroid_count)
        var selection_heap_scores_device = ctx.enqueue_create_buffer[
            DType.float32
        ](selected_centroid_count)
        var selected_positions_device = ctx.enqueue_create_buffer[DType.int64](
            selected_centroid_count
        )
        var selected_scores_device = ctx.enqueue_create_buffer[DType.float32](
            selected_centroid_count
        )
        var output_document_scores_device = ctx.enqueue_create_buffer[
            DType.float32
        ](score_count)
        var best_scores_device = ctx.enqueue_create_buffer[DType.float32](
            best_score_count
        )
        ctx.synchronize()

        def host_ingest_once() capturing raises:
            for index in range(query_value_count):
                query_values_host[index] = py_query_values[index]

            for index in range(token_code_count):
                token_codes_host[index] = py_token_codes[index]

            for index in range(total_vector_count):
                token_scales_host[index] = py_token_scales[index]

            for index in range(centroid_count):
                centroid_token_indices_host[index] = py_centroid_token_indices[
                    index
                ]

            for index in range(centroid_offset_count):
                centroid_doc_offsets_host[index] = py_centroid_doc_offsets[
                    index
                ]

            for index in range(posting_count):
                centroid_doc_indices_host[index] = py_centroid_doc_indices[
                    index
                ]

        def payload_h2d_once() capturing raises:
            token_codes_device.enqueue_copy_from(token_codes_host)
            token_scales_device.enqueue_copy_from(token_scales_host)
            centroid_token_indices_device.enqueue_copy_from(
                centroid_token_indices_host
            )
            centroid_doc_offsets_device.enqueue_copy_from(
                centroid_doc_offsets_host
            )
            centroid_doc_indices_device.enqueue_copy_from(
                centroid_doc_indices_host
            )
            ctx.synchronize()

        def query_h2d_once() capturing raises:
            query_values_device.enqueue_copy_from(query_values_host)
            ctx.synchronize()

        def centroid_score_kernel_once() capturing raises:
            ctx.enqueue_function[
                score_i8_centroid_scores_kernel,
                score_i8_centroid_scores_kernel,
            ](
                query_values_device,
                token_codes_device,
                token_scales_device,
                centroid_token_indices_device,
                output_centroid_scores_device,
                query_vector_count,
                centroid_count,
                centroid_score_count,
                grid_dim=centroid_score_grid_x,
                block_dim=BLOCK_SIZE,
            )
            ctx.synchronize()

        def centroid_select_kernel_once() capturing raises:
            ctx.enqueue_function[
                select_i8_centroid_scores_heap_kernel,
                select_i8_centroid_scores_heap_kernel,
            ](
                output_centroid_scores_device,
                selection_heap_positions_device,
                selection_heap_scores_device,
                selected_positions_device,
                selected_scores_device,
                query_vector_global_count,
                centroid_count,
                centroids_per_query_vector,
                grid_dim=selected_grid_x,
                block_dim=1,
            )
            ctx.synchronize()

        def accumulation_kernel_once() capturing raises:
            ctx.enqueue_function[
                accumulate_i8_selected_centroid_scores_by_query_vector_document_kernel,
                accumulate_i8_selected_centroid_scores_by_query_vector_document_kernel,
            ](
                selected_positions_device,
                selected_scores_device,
                centroid_doc_offsets_device,
                centroid_doc_indices_device,
                best_scores_device,
                query_vector_count,
                centroids_per_query_vector,
                document_count,
                best_score_count,
                grid_dim=best_grid_x,
                block_dim=BLOCK_SIZE,
            )
            ctx.enqueue_function[
                reduce_i8_selected_centroid_best_scores_kernel,
                reduce_i8_selected_centroid_best_scores_kernel,
            ](
                best_scores_device,
                output_document_scores_device,
                query_vector_count,
                document_count,
                score_count,
                grid_dim=score_grid_x,
                block_dim=BLOCK_SIZE,
            )
            ctx.synchronize()

        def d2h_once() capturing raises:
            output_document_scores_device.enqueue_copy_to(
                output_document_scores_host
            )
            ctx.synchronize()

        def selected_validation_d2h_once() capturing raises:
            selected_positions_device.enqueue_copy_to(
                output_selected_positions_host
            )
            selected_scores_device.enqueue_copy_to(output_selected_scores_host)
            ctx.synchronize()

        def output_position_already_selected(
            query_topk_base: Int, rank: Int, document_index: Int
        ) capturing -> Bool:
            for selected_rank in range(rank):
                if (
                    Int(
                        output_topk_positions_host[
                            query_topk_base + selected_rank
                        ]
                    )
                    == document_index
                ):
                    return True
            return False

        def expected_position_already_selected(
            query_topk_base: Int, rank: Int, document_index: Int
        ) capturing -> Bool:
            for selected_rank in range(rank):
                if (
                    Int(
                        expected_topk_positions_host[
                            query_topk_base + selected_rank
                        ]
                    )
                    == document_index
                ):
                    return True
            return False

        def score_position_ranks_before_float32(
            score: Float32,
            position: Int,
            other_score: Float32,
            other_position: Int,
        ) -> Bool:
            if score > other_score:
                return True
            if score < other_score:
                return False
            return position < other_position

        def host_topk_once() capturing:
            for query_index in range(query_count):
                var score_base = query_index * document_count
                var query_topk_base = query_index * top_k
                for rank in range(top_k):
                    var best_position = 0
                    var best_score = Float32(-3.4028234663852886e38)
                    var seen = False
                    for document_index in range(document_count):
                        if output_position_already_selected(
                            query_topk_base, rank, document_index
                        ):
                            continue
                        var score = output_document_scores_host[
                            score_base + document_index
                        ]
                        if not seen or score_position_ranks_before_float32(
                            score,
                            document_index,
                            best_score,
                            best_position,
                        ):
                            best_score = score
                            best_position = document_index
                            seen = True
                    output_topk_positions_host[query_topk_base + rank] = Int64(
                        best_position
                    )

        def expected_topk_once() capturing:
            for query_index in range(query_count):
                var score_base = query_index * document_count
                var query_topk_base = query_index * top_k
                for rank in range(top_k):
                    var best_position = 0
                    var best_score = Float32(-3.4028234663852886e38)
                    var seen = False
                    for document_index in range(document_count):
                        if expected_position_already_selected(
                            query_topk_base, rank, document_index
                        ):
                            continue
                        var score = py_expected_document_scores[
                            score_base + document_index
                        ]
                        if not seen or score_position_ranks_before_float32(
                            score,
                            document_index,
                            best_score,
                            best_position,
                        ):
                            best_score = score
                            best_position = document_index
                            seen = True
                    expected_topk_positions_host[
                        query_topk_base + rank
                    ] = Int64(best_position)

        host_ingest_once()
        payload_h2d_once()
        query_h2d_once()
        centroid_score_kernel_once()
        centroid_select_kernel_once()
        accumulation_kernel_once()
        d2h_once()
        expected_topk_once()
        host_topk_once()
        selected_validation_d2h_once()

        for _ in range(warmup_iterations):
            query_h2d_once()
            centroid_score_kernel_once()
            centroid_select_kernel_once()
            accumulation_kernel_once()
            d2h_once()
            host_topk_once()

        var host_ingest = benchmark.run[host_ingest_once](
            max_iters=measurement_iterations
        )
        host_ingest_once()
        var payload_h2d = benchmark.run[payload_h2d_once](
            max_iters=measurement_iterations
        )
        payload_h2d_once()
        var query_h2d = benchmark.run[query_h2d_once](
            max_iters=measurement_iterations
        )
        query_h2d_once()
        var centroid_score_kernel = benchmark.run[centroid_score_kernel_once](
            max_iters=measurement_iterations
        )
        centroid_score_kernel_once()
        var centroid_select_kernel = benchmark.run[centroid_select_kernel_once](
            max_iters=measurement_iterations
        )
        centroid_select_kernel_once()
        var accumulation_kernel = benchmark.run[accumulation_kernel_once](
            max_iters=measurement_iterations
        )
        accumulation_kernel_once()
        var d2h = benchmark.run[d2h_once](max_iters=measurement_iterations)
        d2h_once()
        var host_topk = benchmark.run[host_topk_once](
            max_iters=measurement_iterations
        )
        host_topk_once()
        var selected_validation_d2h = benchmark.run[
            selected_validation_d2h_once
        ](max_iters=measurement_iterations)
        selected_validation_d2h_once()

        var centroid_token_out_of_range_count = 0
        for centroid_index in range(centroid_count):
            var token_index = centroid_token_indices_host[centroid_index]
            if token_index < 0 or token_index >= Int64(total_vector_count):
                centroid_token_out_of_range_count += 1

        var selected_position_mismatch_count = 0
        var selected_score_mismatch_count = 0
        var selected_score_delta_max_abs = Float64(0.0)
        for index in range(selected_centroid_count):
            if (
                output_selected_positions_host[index]
                != py_expected_selected_positions[index]
            ):
                selected_position_mismatch_count += 1
            var selected_score_delta = abs(
                Float64(output_selected_scores_host[index])
                - Float64(py_expected_selected_scores[index])
            )
            if selected_score_delta > selected_score_delta_max_abs:
                selected_score_delta_max_abs = selected_score_delta
            if selected_score_delta > selected_score_delta_tolerance:
                selected_score_mismatch_count += 1

        var doc_index_out_of_range_count = 0
        for index in range(posting_count):
            var doc_index = centroid_doc_indices_host[index]
            if doc_index < 0 or doc_index >= Int64(document_count):
                doc_index_out_of_range_count += 1

        var score_mismatch_count = 0
        var score_delta_max_abs = Float64(0.0)
        for index in range(score_count):
            var score_delta = abs(
                Float64(output_document_scores_host[index])
                - Float64(py_expected_document_scores[index])
            )
            if score_delta > score_delta_max_abs:
                score_delta_max_abs = score_delta
            if score_delta > score_delta_tolerance:
                score_mismatch_count += 1

        var topk_position_mismatch_count = 0
        for index in range(topk_position_count):
            if (
                output_topk_positions_host[index]
                != expected_topk_positions_host[index]
            ):
                topk_position_mismatch_count += 1

        var py_result = Python.list()
        py_result.append(Python.float(host_ingest.mean()))
        py_result.append(Python.float(payload_h2d.mean()))
        py_result.append(Python.float(query_h2d.mean()))
        py_result.append(Python.float(centroid_score_kernel.mean()))
        py_result.append(Python.float(centroid_select_kernel.mean()))
        py_result.append(Python.float(accumulation_kernel.mean()))
        py_result.append(Python.float(d2h.mean()))
        py_result.append(Python.float(host_topk.mean()))
        py_result.append(Python.float(selected_validation_d2h.mean()))
        py_result.append(Python.int(centroid_token_out_of_range_count))
        py_result.append(Python.int(selected_position_mismatch_count))
        py_result.append(Python.int(selected_score_mismatch_count))
        py_result.append(Python.float(selected_score_delta_max_abs))
        py_result.append(Python.int(score_mismatch_count))
        py_result.append(Python.float(score_delta_max_abs))
        py_result.append(Python.float(score_delta_tolerance))
        py_result.append(Python.int(topk_position_mismatch_count))
        py_result.append(Python.int(doc_index_out_of_range_count))
        py_result.append(Python.int(top_k))
        py_result.append(Python.int(topk_position_count))
        py_result.append(Python.int(selected_centroid_count))
        py_result.append(Python.int(centroid_score_count))
        py_result.append(Python.int(score_count))
        py_result.append(Python.int(document_count))
        py_result.append(Python.int(query_count))
        py_result.append(Python.int(query_vector_count))
        py_result.append(Python.int(centroids_per_query_vector))
        return py_result


def profile_i8_selected_posting_accumulation_addresses(
    py_request: PythonObject,
) raises -> PythonObject:
    var centroid_doc_offsets_address = Int(py=py_request[0])
    var centroid_doc_indices_address = Int(py=py_request[1])
    var selected_positions_address = Int(py=py_request[2])
    var selected_scores_address = Int(py=py_request[3])
    var expected_document_scores_address = Int(py=py_request[4])
    var centroid_count = Int(py=py_request[5])
    var posting_count = Int(py=py_request[6])
    var document_count = Int(py=py_request[7])
    var query_count = Int(py=py_request[8])
    var query_vector_count = Int(py=py_request[9])
    var centroids_per_query_vector = Int(py=py_request[10])
    var top_k = Int(py=py_request[11])
    var warmup_iterations = Int(py=py_request[12])
    var measurement_iterations = Int(py=py_request[13])
    if centroid_doc_offsets_address == 0:
        raise Error("centroid_doc_offsets address must be non-zero")
    if centroid_doc_indices_address == 0:
        raise Error("centroid_doc_indices address must be non-zero")
    if selected_positions_address == 0:
        raise Error("selected positions address must be non-zero")
    if selected_scores_address == 0:
        raise Error("selected scores address must be non-zero")
    if expected_document_scores_address == 0:
        raise Error("expected document scores address must be non-zero")
    if centroid_count <= 0:
        raise Error("centroid_count must be positive")
    if posting_count <= 0:
        raise Error("posting_count must be positive")
    if document_count <= 0:
        raise Error("document_count must be positive")
    if query_count <= 0:
        raise Error("query_count must be positive")
    if query_vector_count <= 0:
        raise Error("query_vector_count must be positive")
    if centroids_per_query_vector <= 0:
        raise Error("centroids_per_query_vector must be positive")
    if top_k <= 0:
        raise Error("top_k must be positive")
    if top_k > document_count:
        raise Error("top_k must not exceed document_count")
    if warmup_iterations < 0:
        raise Error("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise Error("measurement_iterations must be positive")

    var py_centroid_doc_offsets = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=centroid_doc_offsets_address
    )
    var py_centroid_doc_indices = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=centroid_doc_indices_address
    )
    var py_selected_positions = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=selected_positions_address
    )
    var py_selected_scores = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=selected_scores_address
    )
    var py_expected_document_scores = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=expected_document_scores_address
    )
    var centroid_offset_count = centroid_count + 1
    var selected_centroid_count = (
        query_count * query_vector_count * centroids_per_query_vector
    )
    var score_count = query_count * document_count
    var best_score_count = query_count * query_vector_count * document_count
    var topk_position_count = query_count * top_k
    var grid_x = (score_count + BLOCK_SIZE - 1) // BLOCK_SIZE
    var best_grid_x = (best_score_count + BLOCK_SIZE - 1) // BLOCK_SIZE

    with DeviceContext() as ctx:
        var centroid_doc_offsets_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](centroid_offset_count)
        var centroid_doc_indices_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](posting_count)
        var selected_positions_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](selected_centroid_count)
        var selected_scores_host = ctx.enqueue_create_host_buffer[
            DType.float32
        ](selected_centroid_count)
        var output_document_scores_host = ctx.enqueue_create_host_buffer[
            DType.float32
        ](score_count)
        var output_topk_positions_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](topk_position_count)
        var device_topk_positions_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](topk_position_count)
        var device_topk_scores_host = ctx.enqueue_create_host_buffer[
            DType.float32
        ](topk_position_count)
        var expected_topk_positions_host = ctx.enqueue_create_host_buffer[
            DType.int64
        ](topk_position_count)

        var centroid_doc_offsets_device = ctx.enqueue_create_buffer[
            DType.int64
        ](centroid_offset_count)
        var centroid_doc_indices_device = ctx.enqueue_create_buffer[
            DType.int64
        ](posting_count)
        var selected_positions_device = ctx.enqueue_create_buffer[DType.int64](
            selected_centroid_count
        )
        var selected_scores_device = ctx.enqueue_create_buffer[DType.float32](
            selected_centroid_count
        )
        var output_document_scores_device = ctx.enqueue_create_buffer[
            DType.float32
        ](score_count)
        var best_scores_device = ctx.enqueue_create_buffer[DType.float32](
            best_score_count
        )
        var device_topk_positions_device = ctx.enqueue_create_buffer[
            DType.int64
        ](topk_position_count)
        var device_topk_scores_device = ctx.enqueue_create_buffer[
            DType.float32
        ](topk_position_count)
        ctx.synchronize()

        def host_ingest_once() capturing raises:
            for index in range(centroid_offset_count):
                centroid_doc_offsets_host[index] = py_centroid_doc_offsets[
                    index
                ]

            for index in range(posting_count):
                centroid_doc_indices_host[index] = py_centroid_doc_indices[
                    index
                ]

            for index in range(selected_centroid_count):
                selected_positions_host[index] = py_selected_positions[index]
                selected_scores_host[index] = py_selected_scores[index]

        def payload_h2d_once() capturing raises:
            centroid_doc_offsets_device.enqueue_copy_from(
                centroid_doc_offsets_host
            )
            centroid_doc_indices_device.enqueue_copy_from(
                centroid_doc_indices_host
            )
            ctx.synchronize()

        def selected_h2d_once() capturing raises:
            selected_positions_device.enqueue_copy_from(selected_positions_host)
            selected_scores_device.enqueue_copy_from(selected_scores_host)
            ctx.synchronize()

        def accumulation_kernel_once() capturing raises:
            ctx.enqueue_function[
                accumulate_i8_selected_centroid_scores_by_query_vector_document_kernel,
                accumulate_i8_selected_centroid_scores_by_query_vector_document_kernel,
            ](
                selected_positions_device,
                selected_scores_device,
                centroid_doc_offsets_device,
                centroid_doc_indices_device,
                best_scores_device,
                query_vector_count,
                centroids_per_query_vector,
                document_count,
                best_score_count,
                grid_dim=best_grid_x,
                block_dim=BLOCK_SIZE,
            )
            ctx.enqueue_function[
                reduce_i8_selected_centroid_best_scores_kernel,
                reduce_i8_selected_centroid_best_scores_kernel,
            ](
                best_scores_device,
                output_document_scores_device,
                query_vector_count,
                document_count,
                score_count,
                grid_dim=grid_x,
                block_dim=BLOCK_SIZE,
            )
            ctx.synchronize()

        def d2h_once() capturing raises:
            output_document_scores_device.enqueue_copy_to(
                output_document_scores_host
            )
            ctx.synchronize()

        def device_topk_kernel_once() capturing raises:
            ctx.enqueue_function[
                select_i8_document_topk_kernel,
                select_i8_document_topk_kernel,
            ](
                output_document_scores_device,
                device_topk_positions_device,
                device_topk_scores_device,
                query_count,
                document_count,
                top_k,
                grid_dim=query_count,
                block_dim=1,
            )
            ctx.synchronize()

        def device_topk_d2h_once() capturing raises:
            device_topk_positions_device.enqueue_copy_to(
                device_topk_positions_host
            )
            device_topk_scores_device.enqueue_copy_to(device_topk_scores_host)
            ctx.synchronize()

        def output_position_already_selected(
            query_topk_base: Int, rank: Int, document_index: Int
        ) capturing -> Bool:
            for selected_rank in range(rank):
                if (
                    Int(
                        output_topk_positions_host[
                            query_topk_base + selected_rank
                        ]
                    )
                    == document_index
                ):
                    return True
            return False

        def expected_position_already_selected(
            query_topk_base: Int, rank: Int, document_index: Int
        ) capturing -> Bool:
            for selected_rank in range(rank):
                if (
                    Int(
                        expected_topk_positions_host[
                            query_topk_base + selected_rank
                        ]
                    )
                    == document_index
                ):
                    return True
            return False

        def score_position_ranks_before_float32(
            score: Float32,
            position: Int,
            other_score: Float32,
            other_position: Int,
        ) -> Bool:
            if score > other_score:
                return True
            if score < other_score:
                return False
            return position < other_position

        def host_topk_once() capturing:
            for query_index in range(query_count):
                var score_base = query_index * document_count
                var query_topk_base = query_index * top_k
                for rank in range(top_k):
                    var best_position = 0
                    var best_score = Float32(-3.4028234663852886e38)
                    var seen = False
                    for document_index in range(document_count):
                        if output_position_already_selected(
                            query_topk_base, rank, document_index
                        ):
                            continue
                        var score = output_document_scores_host[
                            score_base + document_index
                        ]
                        if not seen or score_position_ranks_before_float32(
                            score,
                            document_index,
                            best_score,
                            best_position,
                        ):
                            best_score = score
                            best_position = document_index
                            seen = True
                    output_topk_positions_host[query_topk_base + rank] = Int64(
                        best_position
                    )

        def expected_topk_once() capturing:
            for query_index in range(query_count):
                var score_base = query_index * document_count
                var query_topk_base = query_index * top_k
                for rank in range(top_k):
                    var best_position = 0
                    var best_score = Float32(-3.4028234663852886e38)
                    var seen = False
                    for document_index in range(document_count):
                        if expected_position_already_selected(
                            query_topk_base, rank, document_index
                        ):
                            continue
                        var score = py_expected_document_scores[
                            score_base + document_index
                        ]
                        if not seen or score_position_ranks_before_float32(
                            score,
                            document_index,
                            best_score,
                            best_position,
                        ):
                            best_score = score
                            best_position = document_index
                            seen = True
                    expected_topk_positions_host[
                        query_topk_base + rank
                    ] = Int64(best_position)

        host_ingest_once()
        payload_h2d_once()
        selected_h2d_once()
        accumulation_kernel_once()
        device_topk_kernel_once()
        device_topk_d2h_once()
        d2h_once()
        expected_topk_once()
        host_topk_once()

        for _ in range(warmup_iterations):
            selected_h2d_once()
            accumulation_kernel_once()
            device_topk_kernel_once()
            device_topk_d2h_once()
            d2h_once()
            host_topk_once()

        var host_ingest = benchmark.run[host_ingest_once](
            max_iters=measurement_iterations
        )
        host_ingest_once()
        var payload_h2d = benchmark.run[payload_h2d_once](
            max_iters=measurement_iterations
        )
        payload_h2d_once()
        var selected_h2d = benchmark.run[selected_h2d_once](
            max_iters=measurement_iterations
        )
        selected_h2d_once()
        var kernel = benchmark.run[accumulation_kernel_once](
            max_iters=measurement_iterations
        )
        accumulation_kernel_once()
        var device_topk_kernel = benchmark.run[device_topk_kernel_once](
            max_iters=measurement_iterations
        )
        device_topk_kernel_once()
        var device_topk_d2h = benchmark.run[device_topk_d2h_once](
            max_iters=measurement_iterations
        )
        device_topk_d2h_once()
        var d2h = benchmark.run[d2h_once](max_iters=measurement_iterations)
        d2h_once()
        var host_topk = benchmark.run[host_topk_once](
            max_iters=measurement_iterations
        )
        host_topk_once()

        var selected_position_out_of_range_count = 0
        for index in range(selected_centroid_count):
            var centroid_position = selected_positions_host[index]
            if centroid_position < 0 or centroid_position >= Int64(
                centroid_count
            ):
                selected_position_out_of_range_count += 1

        var doc_index_out_of_range_count = 0
        for index in range(posting_count):
            var doc_index = centroid_doc_indices_host[index]
            if doc_index < 0 or doc_index >= Int64(document_count):
                doc_index_out_of_range_count += 1

        var score_mismatch_count = 0
        var score_delta_max_abs = Float64(0.0)
        for index in range(score_count):
            var score_delta = abs(
                Float64(output_document_scores_host[index])
                - Float64(py_expected_document_scores[index])
            )
            if score_delta > score_delta_max_abs:
                score_delta_max_abs = score_delta
            if score_delta > Float64(0.00001):
                score_mismatch_count += 1

        var topk_position_mismatch_count = 0
        for index in range(topk_position_count):
            if (
                output_topk_positions_host[index]
                != expected_topk_positions_host[index]
            ):
                topk_position_mismatch_count += 1

        var device_topk_position_mismatch_count = 0
        for index in range(topk_position_count):
            if (
                device_topk_positions_host[index]
                != expected_topk_positions_host[index]
            ):
                device_topk_position_mismatch_count += 1

        var py_result = Python.list()
        py_result.append(Python.float(host_ingest.mean()))
        py_result.append(Python.float(payload_h2d.mean()))
        py_result.append(Python.float(selected_h2d.mean()))
        py_result.append(Python.float(kernel.mean()))
        py_result.append(Python.float(d2h.mean()))
        py_result.append(Python.float(host_topk.mean()))
        py_result.append(Python.float(device_topk_kernel.mean()))
        py_result.append(Python.float(device_topk_d2h.mean()))
        py_result.append(Python.int(selected_position_out_of_range_count))
        py_result.append(Python.int(doc_index_out_of_range_count))
        py_result.append(Python.int(score_mismatch_count))
        py_result.append(Python.float(score_delta_max_abs))
        py_result.append(Python.int(topk_position_mismatch_count))
        py_result.append(Python.int(device_topk_position_mismatch_count))
        py_result.append(Python.int(top_k))
        py_result.append(Python.int(topk_position_count))
        py_result.append(Python.int(selected_centroid_count))
        py_result.append(Python.int(score_count))
        py_result.append(Python.int(document_count))
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
        module.def_function[prepare_i8_address_session_handle](
            "prepare_i8_address_session_handle",
            docstring=(
                "Prepare an explicit heap-owned GPU i8 address session handle."
            ),
        )
        module.def_function[score_i8_address_session_handle](
            "score_i8_address_session_handle",
            docstring=(
                "Score one query/candidate window with an explicit GPU i8"
                " address session handle."
            ),
        )
        module.def_function[score_i8_address_session_handle_topk](
            "score_i8_address_session_handle_topk",
            docstring=(
                "Score one query/candidate window with an explicit GPU i8"
                " address session handle and return top-k positions."
            ),
        )
        module.def_function[score_i8_address_session_handle_topk_no_reference](
            "score_i8_address_session_handle_topk_no_reference",
            docstring=(
                "Score one query/candidate window with an explicit GPU i8"
                " address session handle and return top-k positions without"
                " CPU reference scores."
            ),
        )
        module.def_function[release_i8_address_session_handle](
            "release_i8_address_session_handle",
            docstring="Release an explicit GPU i8 address session handle.",
        )
        module.def_function[prepare_i8_fused_centroid_posting_session_handle](
            "prepare_i8_fused_centroid_posting_session_handle",
            docstring=(
                "Prepare an explicit GPU i8 fused centroid-posting handle."
            ),
        )
        module.def_function[
            score_i8_fused_centroid_posting_session_handle_topk_no_reference
        ](
            "score_i8_fused_centroid_posting_session_handle_topk_no_reference",
            docstring=(
                "Run one fused GPU i8 centroid-posting score call and return"
                " top-k positions without CPU reference inputs."
            ),
        )
        module.def_function[
            profile_i8_fused_centroid_posting_session_handle_topk_no_reference
        ](
            "profile_i8_fused_centroid_posting_session_handle_topk_no_reference",
            docstring=(
                "Profile one fused GPU i8 centroid-posting score call through"
                " an explicit prepared handle."
            ),
        )
        module.def_function[release_i8_fused_centroid_posting_session_handle](
            "release_i8_fused_centroid_posting_session_handle",
            docstring=(
                "Release an explicit GPU i8 fused centroid-posting handle."
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
        module.def_function[
            score_i8_prepared_payload_session_addresses_repeated
        ](
            "score_i8_prepared_payload_session_addresses_repeated",
            docstring=(
                "Score repeated typed-address GPU i8 windows after one"
                " in-call prepared-index device copy."
            ),
        )
        module.def_function[
            score_i8_prepared_payload_session_addresses_multi_window
        ](
            "score_i8_prepared_payload_session_addresses_multi_window",
            docstring=(
                "Score different typed-address GPU i8 windows after one"
                " in-call prepared-index device copy."
            ),
        )
        module.def_function[profile_i8_candidate_generation_payload_addresses](
            "profile_i8_candidate_generation_payload_addresses",
            docstring=(
                "Profile typed-address GPU preparation for i8 candidate"
                " generation centroid-posting payload tensors."
            ),
        )
        module.def_function[profile_i8_selected_posting_traversal_addresses](
            "profile_i8_selected_posting_traversal_addresses",
            docstring=(
                "Profile GPU traversal of selected i8 centroid posting lists"
                " without document score accumulation."
            ),
        )
        module.def_function[profile_i8_centroid_selection_addresses](
            "profile_i8_centroid_selection_addresses",
            docstring=(
                "Profile GPU scoring of i8 sampled centroids plus host"
                " centroid top-k selection."
            ),
        )
        module.def_function[
            profile_i8_fused_centroid_posting_accumulation_addresses
        ](
            "profile_i8_fused_centroid_posting_accumulation_addresses",
            docstring=(
                "Profile GPU centroid scoring, device centroid selection,"
                " and posting accumulation as one internal primitive."
            ),
        )
        module.def_function[profile_i8_selected_posting_accumulation_addresses](
            "profile_i8_selected_posting_accumulation_addresses",
            docstring=(
                "Profile GPU accumulation of selected i8 centroid posting"
                " scores into dense per-document scores."
            ),
        )
        return module.finalize()
    except e:
        abort(String("error creating kayak mojo gpu i8 rerank bindings:", e))
