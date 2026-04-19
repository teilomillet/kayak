from std.collections import List
from std.math import erf, exp, log, sqrt, tanh
from std.sys.info import simd_width_of

from kayak.contracts import EncodedQuery
from kayak.numeric import VECTOR_SCALAR_NAME, VectorScalar, zero_vector_scalar
from kayak.scoring.dot import dot_product


comptime LATENT_PROXY_BLOCK_ORDER_LINEAR_ACTIVATION_NORM = (
    "linear_activation_norm"
)
comptime LATENT_PROXY_BLOCK_ORDER_LINEAR_NORM_ACTIVATION = (
    "linear_norm_activation"
)

comptime LATENT_PROXY_ACTIVATION_GELU = "gelu"
comptime LATENT_PROXY_ACTIVATION_MISH = "mish"
comptime LATENT_PROXY_ACTIVATION_RELU = "relu"
comptime LATENT_PROXY_ACTIVATION_SILU = "silu"


def require_supported_latent_proxy_block_order(order_kind: String) raises -> String:
    if order_kind == LATENT_PROXY_BLOCK_ORDER_LINEAR_ACTIVATION_NORM:
        return order_kind.copy()
    if order_kind == LATENT_PROXY_BLOCK_ORDER_LINEAR_NORM_ACTIVATION:
        return order_kind.copy()
    raise Error("unsupported latent proxy block_order: " + order_kind)


def require_supported_latent_proxy_activation(
    activation_kind: String
) raises -> String:
    if activation_kind == LATENT_PROXY_ACTIVATION_RELU:
        return activation_kind.copy()
    if activation_kind == LATENT_PROXY_ACTIVATION_GELU:
        return activation_kind.copy()
    if activation_kind == LATENT_PROXY_ACTIVATION_SILU:
        return activation_kind.copy()
    if activation_kind == LATENT_PROXY_ACTIVATION_MISH:
        return activation_kind.copy()
    raise Error("unsupported latent proxy activation: " + activation_kind)


struct LatentQueryProjectionBlock(Copyable):
    var order_kind: String
    var activation_kind: String
    var input_dim: Int
    var output_dim: Int
    var linear_rows: List[List[VectorScalar]]
    var linear_bias: List[VectorScalar]
    var activation_output_scale: VectorScalar
    var layer_norm_affine: Bool
    var layer_norm_epsilon: VectorScalar
    var layer_norm_weight: List[VectorScalar]
    var layer_norm_bias: List[VectorScalar]

    def __init__(
        out self,
        var order_kind: String,
        var activation_kind: String,
        input_dim: Int,
        output_dim: Int,
        read linear_rows: List[List[VectorScalar]],
        read linear_bias: List[VectorScalar],
        activation_output_scale: VectorScalar,
        layer_norm_affine: Bool,
        layer_norm_epsilon: VectorScalar,
        read layer_norm_weight: List[VectorScalar],
        read layer_norm_bias: List[VectorScalar],
    ) raises:
        self.order_kind = require_supported_latent_proxy_block_order(order_kind)
        self.activation_kind = require_supported_latent_proxy_activation(
            activation_kind
        )
        if input_dim <= 0:
            raise Error("latent proxy block input_dim must be positive")
        if output_dim <= 0:
            raise Error("latent proxy block output_dim must be positive")
        if len(linear_rows) != output_dim:
            raise Error("latent proxy block linear_rows length must match output_dim")
        for row in linear_rows:
            if len(row) != input_dim:
                raise Error(
                    "latent proxy block linear_rows width must match input_dim"
                )
        if len(linear_bias) != output_dim:
            raise Error("latent proxy block linear_bias length must match output_dim")
        if activation_output_scale <= VectorScalar(0.0):
            raise Error("latent proxy block activation_output_scale must be positive")
        if layer_norm_epsilon <= VectorScalar(0.0):
            raise Error("latent proxy block layer_norm_epsilon must be positive")
        if layer_norm_affine:
            if len(layer_norm_weight) != output_dim:
                raise Error(
                    "latent proxy block layer_norm_weight length must match output_dim"
                )
            if len(layer_norm_bias) != output_dim:
                raise Error(
                    "latent proxy block layer_norm_bias length must match output_dim"
                )
        elif len(layer_norm_weight) != 0 or len(layer_norm_bias) != 0:
            raise Error(
                "non-affine latent proxy blocks must not carry layer norm weights"
            )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_rows = linear_rows.copy()
        self.linear_bias = linear_bias.copy()
        self.activation_output_scale = activation_output_scale
        self.layer_norm_affine = layer_norm_affine
        self.layer_norm_epsilon = layer_norm_epsilon
        self.layer_norm_weight = layer_norm_weight.copy()
        self.layer_norm_bias = layer_norm_bias.copy()


struct LatentQueryProjection(Copyable):
    var input_vector_dim: Int
    var output_vector_dim: Int
    var query_divisor: VectorScalar
    var blocks: List[LatentQueryProjectionBlock]

    def __init__(
        out self,
        input_vector_dim: Int,
        output_vector_dim: Int,
        query_divisor: VectorScalar,
        read blocks: List[LatentQueryProjectionBlock],
    ) raises:
        if input_vector_dim <= 0:
            raise Error("latent query projection input_vector_dim must be positive")
        if output_vector_dim <= 0:
            raise Error("latent query projection output_vector_dim must be positive")
        if query_divisor <= VectorScalar(0.0):
            raise Error("latent query projection query_divisor must be positive")
        if len(blocks) != 0:
            if blocks[0].input_dim != input_vector_dim:
                raise Error(
                    "latent query projection input_vector_dim must match the first block"
                )
            if blocks[len(blocks) - 1].output_dim != output_vector_dim:
                raise Error(
                    "latent query projection output_vector_dim must match the final block"
                )
            for index in range(len(blocks) - 1):
                if blocks[index].output_dim != blocks[index + 1].input_dim:
                    raise Error(
                        "latent query projection blocks must chain by dimension"
                    )
        self.input_vector_dim = input_vector_dim
        self.output_vector_dim = output_vector_dim
        self.query_divisor = query_divisor
        self.blocks = blocks.copy()


struct LatentProxyIndex(Copyable):
    var doc_ids: List[String]
    var proxy_vectors: List[List[VectorScalar]]
    var vector_dim: Int
    var document_count: Int

    def __init__(
        out self,
        var doc_ids: List[String],
        var proxy_vectors: List[List[VectorScalar]],
        vector_dim: Int,
    ) raises:
        if vector_dim <= 0:
            raise Error("latent proxy index vector_dim must be positive")
        if len(doc_ids) != len(proxy_vectors):
            raise Error(
                "latent proxy index doc_ids and proxy_vectors must have matching lengths"
            )
        for proxy_vector in proxy_vectors:
            if len(proxy_vector) != vector_dim:
                raise Error(
                    "latent proxy index proxy vector dimension must match vector_dim"
                )
        self.doc_ids = doc_ids^
        self.proxy_vectors = proxy_vectors^
        self.vector_dim = vector_dim
        self.document_count = len(self.doc_ids)


struct MutableLatentQueryProjectionScratch:
    var linear_values: List[VectorScalar]
    var buffer_a: List[VectorScalar]
    var buffer_b: List[VectorScalar]

    def __init__(out self):
        self.linear_values = List[VectorScalar]()
        self.buffer_a = List[VectorScalar]()
        self.buffer_b = List[VectorScalar]()

    def ensure_projection(mut self, read projection: LatentQueryProjection):
        var max_width = projection.input_vector_dim
        for block in projection.blocks:
            if block.input_dim > max_width:
                max_width = block.input_dim
            if block.output_dim > max_width:
                max_width = block.output_dim

        while len(self.linear_values) < max_width:
            self.linear_values.append(zero_vector_scalar())
        while len(self.buffer_a) < max_width:
            self.buffer_a.append(zero_vector_scalar())
        while len(self.buffer_b) < max_width:
            self.buffer_b.append(zero_vector_scalar())


def latent_proxy_gelu(value: VectorScalar) -> VectorScalar:
    var x = Float64(value)
    return VectorScalar(
        Float64(0.5) * x * (Float64(1.0) + erf(x / sqrt(Float64(2.0))))
    )


def latent_proxy_mish(value: VectorScalar) -> VectorScalar:
    var x = Float64(value)
    if x > Float64(20.0):
        return value
    if x < Float64(-20.0):
        return VectorScalar(x * exp(x))
    var softplus = log(Float64(1.0) + exp(x))
    return VectorScalar(x * tanh(softplus))


def apply_latent_proxy_activation(
    value: VectorScalar, activation_kind: String
) raises -> VectorScalar:
    if activation_kind == LATENT_PROXY_ACTIVATION_RELU:
        if value < VectorScalar(0.0):
            return VectorScalar(0.0)
        return value
    if activation_kind == LATENT_PROXY_ACTIVATION_GELU:
        return latent_proxy_gelu(value)
    if activation_kind == LATENT_PROXY_ACTIVATION_SILU:
        var x = Float64(value)
        return VectorScalar(x / (Float64(1.0) + exp(-x)))
    if activation_kind == LATENT_PROXY_ACTIVATION_MISH:
        return latent_proxy_mish(value)
    raise Error("unsupported latent proxy activation: " + activation_kind)


def linear_block_output_reference(
    read input_vector: List[VectorScalar],
    read block: LatentQueryProjectionBlock,
) raises -> List[VectorScalar]:
    if len(input_vector) != block.input_dim:
        raise Error("latent proxy linear block input dimension mismatch")
    var output = List[VectorScalar]()
    for row_index in range(block.output_dim):
        output.append(
            VectorScalar(dot_product(block.linear_rows[row_index], input_vector))
            + block.linear_bias[row_index]
        )
    return output^


def linear_block_output_tiled4(
    read input_vector: List[VectorScalar],
    read block: LatentQueryProjectionBlock,
) raises -> List[VectorScalar]:
    if len(input_vector) != block.input_dim:
        raise Error("latent proxy linear block input dimension mismatch")
    if VECTOR_SCALAR_NAME != "Float32" or block.output_dim < 4:
        return linear_block_output_reference(input_vector, block)

    comptime width = simd_width_of[VectorScalar]()
    var simd_limit = (block.input_dim // width) * width
    var input_ptr = input_vector.unsafe_ptr()
    var output = List[VectorScalar]()
    var row_index = 0

    # The tiled kernel keeps one input chunk live while accumulating four
    # output rows, which is the reuse pattern that the hotspot benchmark showed
    # was missing from the reference row-wise loop.
    while row_index + 3 < block.output_dim:
        var row0_ptr = block.linear_rows[row_index].unsafe_ptr()
        var row1_ptr = block.linear_rows[row_index + 1].unsafe_ptr()
        var row2_ptr = block.linear_rows[row_index + 2].unsafe_ptr()
        var row3_ptr = block.linear_rows[row_index + 3].unsafe_ptr()
        var accum0 = SIMD[DType.float32, width](0.0)
        var accum1 = SIMD[DType.float32, width](0.0)
        var accum2 = SIMD[DType.float32, width](0.0)
        var accum3 = SIMD[DType.float32, width](0.0)

        for dim in range(0, simd_limit, width):
            var input_chunk = (input_ptr + dim).load[width=width]()
            accum0 += input_chunk * (row0_ptr + dim).load[width=width]()
            accum1 += input_chunk * (row1_ptr + dim).load[width=width]()
            accum2 += input_chunk * (row2_ptr + dim).load[width=width]()
            accum3 += input_chunk * (row3_ptr + dim).load[width=width]()

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
            total0 += input_value * block.linear_rows[row_index].unsafe_get(dim)
            total1 += input_value * block.linear_rows[row_index + 1].unsafe_get(dim)
            total2 += input_value * block.linear_rows[row_index + 2].unsafe_get(dim)
            total3 += input_value * block.linear_rows[row_index + 3].unsafe_get(dim)

        output.append(total0)
        output.append(total1)
        output.append(total2)
        output.append(total3)
        row_index += 4

    while row_index < block.output_dim:
        output.append(
            VectorScalar(dot_product(block.linear_rows[row_index], input_vector))
            + block.linear_bias[row_index]
        )
        row_index += 1

    return output^


def linear_block_output(
    read input_vector: List[VectorScalar],
    read block: LatentQueryProjectionBlock,
) raises -> List[VectorScalar]:
    return linear_block_output_tiled4(input_vector, block)


def linear_block_output_into_reference(
    read input_vector: List[VectorScalar],
    read block: LatentQueryProjectionBlock,
    mut output: List[VectorScalar],
) raises:
    if len(input_vector) < block.input_dim:
        raise Error("latent proxy linear block input dimension mismatch")
    if len(output) < block.output_dim:
        raise Error("latent proxy linear block output buffer is too small")
    for row_index in range(block.output_dim):
        output[row_index] = (
            VectorScalar(dot_product(block.linear_rows[row_index], input_vector))
            + block.linear_bias[row_index]
        )


def linear_block_output_into(
    read input_vector: List[VectorScalar],
    read block: LatentQueryProjectionBlock,
    mut output: List[VectorScalar],
) raises:
    linear_block_output_into_reference(input_vector, block, output)


def layer_normalized_output(
    read values: List[VectorScalar],
    read block: LatentQueryProjectionBlock,
) raises -> List[VectorScalar]:
    if len(values) != block.output_dim:
        raise Error("latent proxy layer norm dimension mismatch")
    var mean = Float64(0.0)
    for value in values:
        mean += Float64(value)
    mean = mean / Float64(len(values))
    var variance = Float64(0.0)
    for value in values:
        var centered = Float64(value) - mean
        variance += centered * centered
    variance = variance / Float64(len(values))
    var denom = sqrt(variance + Float64(block.layer_norm_epsilon))
    var normalized = List[VectorScalar]()
    for index in range(len(values)):
        var value = VectorScalar((Float64(values[index]) - mean) / denom)
        if block.layer_norm_affine:
            value = (
                value * block.layer_norm_weight[index] + block.layer_norm_bias[index]
            )
        normalized.append(value)
    return normalized^


def layer_normalized_output_into(
    read values: List[VectorScalar],
    read block: LatentQueryProjectionBlock,
    mut output: List[VectorScalar],
) raises:
    if len(values) < block.output_dim:
        raise Error("latent proxy layer norm dimension mismatch")
    if len(output) < block.output_dim:
        raise Error("latent proxy layer norm output buffer is too small")
    var mean = Float64(0.0)
    for index in range(block.output_dim):
        mean += Float64(values[index])
    mean = mean / Float64(block.output_dim)
    var variance = Float64(0.0)
    for index in range(block.output_dim):
        var centered = Float64(values[index]) - mean
        variance += centered * centered
    variance = variance / Float64(block.output_dim)
    var denom = sqrt(variance + Float64(block.layer_norm_epsilon))
    for index in range(block.output_dim):
        var value = VectorScalar((Float64(values[index]) - mean) / denom)
        if block.layer_norm_affine:
            value = (
                value * block.layer_norm_weight[index] + block.layer_norm_bias[index]
            )
        output[index] = value


def projected_query_token_for_block(
    read input_vector: List[VectorScalar],
    read block: LatentQueryProjectionBlock,
) raises -> List[VectorScalar]:
    var projected = linear_block_output(input_vector, block)
    if block.order_kind == LATENT_PROXY_BLOCK_ORDER_LINEAR_NORM_ACTIVATION:
        projected = layer_normalized_output(projected, block)
        for index in range(len(projected)):
            projected[index] = apply_latent_proxy_activation(
                projected[index], block.activation_kind
            )
            projected[index] = (
                projected[index] * block.activation_output_scale
            )
        return projected^
    for index in range(len(projected)):
        projected[index] = apply_latent_proxy_activation(
            projected[index], block.activation_kind
        )
        projected[index] = projected[index] * block.activation_output_scale
    return layer_normalized_output(projected, block)


def projected_query_token_for_block_into(
    read input_vector: List[VectorScalar],
    read block: LatentQueryProjectionBlock,
    mut linear_values: List[VectorScalar],
    mut output: List[VectorScalar],
) raises:
    linear_block_output_into(input_vector, block, linear_values)
    if block.order_kind == LATENT_PROXY_BLOCK_ORDER_LINEAR_NORM_ACTIVATION:
        layer_normalized_output_into(linear_values, block, output)
        for index in range(block.output_dim):
            output[index] = apply_latent_proxy_activation(
                output[index], block.activation_kind
            )
            output[index] = output[index] * block.activation_output_scale
        return

    for index in range(block.output_dim):
        linear_values[index] = apply_latent_proxy_activation(
            linear_values[index],
            block.activation_kind,
        )
        linear_values[index] = (
            linear_values[index] * block.activation_output_scale
        )
    layer_normalized_output_into(linear_values, block, output)


def build_query_latent_proxy_vector_single_block(
    read query: EncodedQuery,
    read projection: LatentQueryProjection,
) raises -> List[VectorScalar]:
    if query.vector_dim != projection.input_vector_dim:
        raise Error(
            "latent query projection input_vector_dim does not match query vector_dim"
        )
    if len(projection.blocks) == 0:
        raise Error("latent query projection requires at least one block")
    if len(projection.blocks) != 1:
        raise Error("single-block latent query projection requires exactly one block")
    var block = projection.blocks[0].copy()
    var pooled = List[VectorScalar]()
    for _ in range(projection.output_vector_dim):
        pooled.append(zero_vector_scalar())
    var projected = List[VectorScalar]()
    for _ in range(projection.output_vector_dim):
        projected.append(zero_vector_scalar())

    for token_vector in query.token_vectors:
        for row_index in range(block.output_dim):
            projected[row_index] = (
                VectorScalar(
                    dot_product(
                        block.linear_rows[row_index],
                        token_vector,
                    )
                )
                + block.linear_bias[row_index]
            )

        if block.order_kind == LATENT_PROXY_BLOCK_ORDER_LINEAR_NORM_ACTIVATION:
            var mean = Float64(0.0)
            for value in projected:
                mean += Float64(value)
            mean = mean / Float64(len(projected))

            var variance = Float64(0.0)
            for value in projected:
                var centered = Float64(value) - mean
                variance += centered * centered
            variance = variance / Float64(len(projected))
            var denom = sqrt(variance + Float64(block.layer_norm_epsilon))

            for dim_index in range(block.output_dim):
                var value = VectorScalar(
                    (Float64(projected[dim_index]) - mean) / denom
                )
                if block.layer_norm_affine:
                    value = (
                        value * block.layer_norm_weight[dim_index]
                        + block.layer_norm_bias[dim_index]
                    )
                value = apply_latent_proxy_activation(
                    value,
                    block.activation_kind,
                )
                pooled[dim_index] += (
                    value * block.activation_output_scale
                )
            continue

        for dim_index in range(block.output_dim):
            projected[dim_index] = (
                apply_latent_proxy_activation(
                    projected[dim_index],
                    block.activation_kind,
                )
                * block.activation_output_scale
            )

        var mean = Float64(0.0)
        for value in projected:
            mean += Float64(value)
        mean = mean / Float64(len(projected))

        var variance = Float64(0.0)
        for value in projected:
            var centered = Float64(value) - mean
            variance += centered * centered
        variance = variance / Float64(len(projected))
        var denom = sqrt(variance + Float64(block.layer_norm_epsilon))

        for dim_index in range(block.output_dim):
            var value = VectorScalar(
                (Float64(projected[dim_index]) - mean) / denom
            )
            if block.layer_norm_affine:
                value = (
                    value * block.layer_norm_weight[dim_index]
                    + block.layer_norm_bias[dim_index]
                )
            pooled[dim_index] += value
    for dim_index in range(projection.output_vector_dim):
        pooled[dim_index] = pooled[dim_index] / projection.query_divisor
    return pooled^


def build_query_latent_proxy_vector_multi_block_reference(
    read query: EncodedQuery,
    read projection: LatentQueryProjection,
) raises -> List[VectorScalar]:
    if query.vector_dim != projection.input_vector_dim:
        raise Error(
            "latent query projection input_vector_dim does not match query vector_dim"
        )
    if len(projection.blocks) == 0:
        raise Error("latent query projection requires at least one block")
    var pooled = List[VectorScalar]()
    for _ in range(projection.output_vector_dim):
        pooled.append(zero_vector_scalar())
    for token_vector in query.token_vectors:
        var current = token_vector.copy()
        for block in projection.blocks:
            current = projected_query_token_for_block(current, block)
        for dim_index in range(projection.output_vector_dim):
            pooled[dim_index] += current[dim_index]
    for dim_index in range(projection.output_vector_dim):
        pooled[dim_index] = pooled[dim_index] / projection.query_divisor
    return pooled^


def build_query_latent_proxy_vector_multi_block_with_scratch(
    read query: EncodedQuery,
    read projection: LatentQueryProjection,
    mut scratch: MutableLatentQueryProjectionScratch,
) raises -> List[VectorScalar]:
    if query.vector_dim != projection.input_vector_dim:
        raise Error(
            "latent query projection input_vector_dim does not match query vector_dim"
        )
    if len(projection.blocks) == 0:
        raise Error("latent query projection requires at least one block")
    scratch.ensure_projection(projection)

    var pooled = List[VectorScalar]()
    for _ in range(projection.output_vector_dim):
        pooled.append(zero_vector_scalar())

    for token_vector in query.token_vectors:
        projected_query_token_for_block_into(
            token_vector,
            projection.blocks[0],
            scratch.linear_values,
            scratch.buffer_a,
        )
        var final_in_buffer_a = True
        for block_index in range(1, len(projection.blocks)):
            if final_in_buffer_a:
                projected_query_token_for_block_into(
                    scratch.buffer_a,
                    projection.blocks[block_index],
                    scratch.linear_values,
                    scratch.buffer_b,
                )
                final_in_buffer_a = False
            else:
                projected_query_token_for_block_into(
                    scratch.buffer_b,
                    projection.blocks[block_index],
                    scratch.linear_values,
                    scratch.buffer_a,
                )
                final_in_buffer_a = True
        if final_in_buffer_a:
            for dim_index in range(projection.output_vector_dim):
                pooled[dim_index] += scratch.buffer_a[dim_index]
        else:
            for dim_index in range(projection.output_vector_dim):
                pooled[dim_index] += scratch.buffer_b[dim_index]
    for dim_index in range(projection.output_vector_dim):
        pooled[dim_index] = pooled[dim_index] / projection.query_divisor
    return pooled^


def build_query_latent_proxy_vector_multi_block(
    read query: EncodedQuery,
    read projection: LatentQueryProjection,
) raises -> List[VectorScalar]:
    return build_query_latent_proxy_vector_multi_block_reference(
        query, projection
    )


def build_query_latent_proxy_vector(
    read query: EncodedQuery,
    read projection: LatentQueryProjection,
) raises -> List[VectorScalar]:
    return build_query_latent_proxy_vector_multi_block(query, projection)


def score_query_against_latent_proxy(
    read query: EncodedQuery,
    read projection: LatentQueryProjection,
    read proxy_index: LatentProxyIndex,
    document_index: Int,
) raises -> VectorScalar:
    if document_index < 0 or document_index >= proxy_index.document_count:
        raise Error("latent proxy document_index is out of range")
    var query_proxy = build_query_latent_proxy_vector(query, projection)
    if len(query_proxy) != proxy_index.vector_dim:
        raise Error(
            "latent query projection output_vector_dim does not match proxy index vector_dim"
        )
    return dot_product(query_proxy, proxy_index.proxy_vectors[document_index])
