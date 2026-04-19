# Benchmark-only support for latent-proxy linear-kernel experiments.
# Owns synthetic row-layout helpers and fused-output kernels.
# Does not own runtime kernel selection.

from std.collections import List
from std.sys.info import simd_width_of

from kayak.index import LatentQueryProjectionBlock
from kayak.numeric import VECTOR_SCALAR_NAME, VectorScalar, zero_vector_scalar


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


def linear_block_output_flat_rows_tiled4(
    read input_vector: List[VectorScalar],
    read block: LatentQueryProjectionBlock,
    read flat_rows: List[VectorScalar],
) raises -> List[VectorScalar]:
    if len(input_vector) != block.input_dim:
        raise Error("latent proxy linear block input dimension mismatch")
    if len(flat_rows) != block.input_dim * block.output_dim:
        raise Error("latent proxy flat row storage size mismatch")
    if VECTOR_SCALAR_NAME != "Float32":
        return linear_block_output_flat_rows(input_vector, block, flat_rows)

    comptime width = simd_width_of[VectorScalar]()
    var simd_limit = (block.input_dim // width) * width
    var input_ptr = input_vector.unsafe_ptr()
    var flat_rows_ptr = flat_rows.unsafe_ptr()
    var output = List[VectorScalar]()
    var row_index = 0

    # Reuse each loaded input chunk across four output rows before moving on.
    while row_index + 3 < block.output_dim:
        var row_offset0 = row_index * block.input_dim
        var row_offset1 = (row_index + 1) * block.input_dim
        var row_offset2 = (row_index + 2) * block.input_dim
        var row_offset3 = (row_index + 3) * block.input_dim
        var accum0 = SIMD[DType.float32, width](0.0)
        var accum1 = SIMD[DType.float32, width](0.0)
        var accum2 = SIMD[DType.float32, width](0.0)
        var accum3 = SIMD[DType.float32, width](0.0)

        for dim in range(0, simd_limit, width):
            var input_chunk = (input_ptr + dim).load[width=width]()
            accum0 += input_chunk * (
                flat_rows_ptr + row_offset0 + dim
            ).load[width=width]()
            accum1 += input_chunk * (
                flat_rows_ptr + row_offset1 + dim
            ).load[width=width]()
            accum2 += input_chunk * (
                flat_rows_ptr + row_offset2 + dim
            ).load[width=width]()
            accum3 += input_chunk * (
                flat_rows_ptr + row_offset3 + dim
            ).load[width=width]()

        var total0 = VectorScalar(accum0.reduce_add()[0]) + block.linear_bias[row_index]
        var total1 = (
            VectorScalar(accum1.reduce_add()[0]) + block.linear_bias[row_index + 1]
        )
        var total2 = (
            VectorScalar(accum2.reduce_add()[0]) + block.linear_bias[row_index + 2]
        )
        var total3 = (
            VectorScalar(accum3.reduce_add()[0]) + block.linear_bias[row_index + 3]
        )

        for dim in range(simd_limit, block.input_dim):
            var input_value = input_vector.unsafe_get(dim)
            total0 += input_value * flat_rows.unsafe_get(row_offset0 + dim)
            total1 += input_value * flat_rows.unsafe_get(row_offset1 + dim)
            total2 += input_value * flat_rows.unsafe_get(row_offset2 + dim)
            total3 += input_value * flat_rows.unsafe_get(row_offset3 + dim)

        output.append(total0)
        output.append(total1)
        output.append(total2)
        output.append(total3)
        row_index += 4

    while row_index < block.output_dim:
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
        row_index += 1

    return output^
