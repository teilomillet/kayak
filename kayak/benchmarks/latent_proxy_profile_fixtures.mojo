from std.collections import List

from kayak.contracts import EncodedQuery
from kayak.index import (
    LATENT_PROXY_ACTIVATION_GELU,
    LATENT_PROXY_BLOCK_ORDER_LINEAR_NORM_ACTIVATION,
    LatentProxyIndex,
    LatentQueryProjection,
    LatentQueryProjectionBlock,
)
from kayak.numeric import VectorScalar


comptime DEFAULT_LATENT_PROXY_INPUT_VECTOR_DIM = 128
comptime DEFAULT_LATENT_PROXY_CANDIDATE_K = 10


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
            DEFAULT_LATENT_PROXY_INPUT_VECTOR_DIM,
            256,
            1,
            DEFAULT_LATENT_PROXY_CANDIDATE_K,
        ),
        LatentProxyPrimitiveSweepProfile(
            "q32_lat256_b1_docs128",
            128,
            32,
            DEFAULT_LATENT_PROXY_INPUT_VECTOR_DIM,
            256,
            1,
            DEFAULT_LATENT_PROXY_CANDIDATE_K,
        ),
        LatentProxyPrimitiveSweepProfile(
            "q32_lat1024_b1_docs128",
            128,
            32,
            DEFAULT_LATENT_PROXY_INPUT_VECTOR_DIM,
            1024,
            1,
            DEFAULT_LATENT_PROXY_CANDIDATE_K,
        ),
        LatentProxyPrimitiveSweepProfile(
            "q32_lat2048_b1_docs128",
            128,
            32,
            DEFAULT_LATENT_PROXY_INPUT_VECTOR_DIM,
            2048,
            1,
            DEFAULT_LATENT_PROXY_CANDIDATE_K,
        ),
        LatentProxyPrimitiveSweepProfile(
            "q32_lat2048_b2_docs128",
            128,
            32,
            DEFAULT_LATENT_PROXY_INPUT_VECTOR_DIM,
            2048,
            2,
            DEFAULT_LATENT_PROXY_CANDIDATE_K,
        ),
        LatentProxyPrimitiveSweepProfile(
            "q64_lat2048_b1_docs128",
            128,
            64,
            DEFAULT_LATENT_PROXY_INPUT_VECTOR_DIM,
            2048,
            1,
            DEFAULT_LATENT_PROXY_CANDIDATE_K,
        ),
    ]


def require_latent_proxy_primitive_profile(
    profile_name: String
) raises -> LatentProxyPrimitiveSweepProfile:
    for profile in default_latent_proxy_primitive_profiles():
        if profile.profile_name == profile_name:
            return profile.copy()
    raise Error("unknown latent proxy primitive profile: " + profile_name)
