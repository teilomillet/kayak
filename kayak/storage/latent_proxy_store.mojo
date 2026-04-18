from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.index import (
    LatentProxyIndex,
    LatentQueryProjection,
    LatentQueryProjectionBlock,
)
from kayak.numeric import STORAGE_FORMAT_VERSION, VECTOR_SCALAR_NAME

from .binary_vector_codec import (
    read_binary_vector_payload_with_encoding,
    write_binary_vector_payload_with_encoding,
)
from .manifest import (
    ManifestEntry,
    read_manifest,
    require_manifest_value,
    require_supported_storage_format,
    write_manifest,
)
from .metadata import StoredLatentProxyIndex
from .text_codec import (
    append_line,
    parse_int,
    parse_vector_scalar,
    read_non_empty_lines,
)
from .vector_payload_encoding import (
    VECTOR_PAYLOAD_ENCODING_BINARY_LE,
    require_supported_packed_index_vector_payload_encoding,
)


comptime LATENT_PROXY_ARTIFACT_KIND = "latent_proxy_index"


def latent_proxy_manifest_path(root: Path) -> Path:
    return root / "manifest.tsv"


def latent_proxy_index_exists(root: Path) -> Bool:
    return latent_proxy_manifest_path(root).exists()


def latent_proxy_doc_ids_path(root: Path) -> Path:
    return root / "doc_ids.tsv"


def latent_proxy_vectors_path(root: Path) -> Path:
    return root / "proxy_vectors.bin"


def latent_proxy_block_linear_rows_path(root: Path, block_index: Int) -> Path:
    var filename = "projection_block_" + String(block_index) + "_linear_rows.bin"
    return root / filename


def latent_proxy_block_linear_bias_path(root: Path, block_index: Int) -> Path:
    var filename = "projection_block_" + String(block_index) + "_linear_bias.bin"
    return root / filename


def latent_proxy_block_layer_norm_weight_path(
    root: Path, block_index: Int
) -> Path:
    var filename = (
        "projection_block_" + String(block_index) + "_layer_norm_weight.bin"
    )
    return root / filename


def latent_proxy_block_layer_norm_bias_path(root: Path, block_index: Int) -> Path:
    var filename = "projection_block_" + String(block_index) + "_layer_norm_bias.bin"
    return root / filename


def file_size_bytes(path: Path) raises -> Int:
    if path.suffix() == ".bin":
        return len(path.read_bytes())
    return path.read_text().byte_length()


def latent_proxy_storage_byte_size(root: Path) raises -> Int:
    var manifest = read_manifest(latent_proxy_manifest_path(root))
    _ = require_supported_storage_format(manifest)
    var total = file_size_bytes(latent_proxy_manifest_path(root))
    total += file_size_bytes(latent_proxy_doc_ids_path(root))
    total += file_size_bytes(latent_proxy_vectors_path(root))
    var block_count = parse_int(
        require_manifest_value(manifest, "projection_block_count"),
        "projection_block_count",
    )
    for block_index in range(block_count):
        total += file_size_bytes(latent_proxy_block_linear_rows_path(root, block_index))
        total += file_size_bytes(latent_proxy_block_linear_bias_path(root, block_index))
        if (
            require_manifest_value(
                manifest,
                "projection_block_" + String(block_index) + "_layer_norm_affine",
            )
            == "1"
        ):
            total += file_size_bytes(
                latent_proxy_block_layer_norm_weight_path(root, block_index)
            )
            total += file_size_bytes(
                latent_proxy_block_layer_norm_bias_path(root, block_index)
            )
    return total


def bool_manifest_string(value: Bool) -> String:
    if value:
        return "1"
    return "0"


def latent_proxy_manifest_entries(
    read stored: StoredLatentProxyIndex,
    vector_payload_encoding: String,
    artifact_byte_size: Int,
) raises -> List[ManifestEntry]:
    var entries = List[ManifestEntry]()
    entries.append(ManifestEntry("format_version", String(STORAGE_FORMAT_VERSION)))
    entries.append(ManifestEntry("artifact_kind", LATENT_PROXY_ARTIFACT_KIND))
    entries.append(ManifestEntry("vector_scalar_name", stored.vector_scalar_name))
    entries.append(ManifestEntry("dataset_id", stored.dataset_id))
    entries.append(ManifestEntry("model_name", stored.model_name))
    entries.append(
        ManifestEntry("vector_payload_encoding", vector_payload_encoding)
    )
    entries.append(
        ManifestEntry("input_vector_dim", String(stored.input_vector_dim))
    )
    entries.append(ManifestEntry("latent_dim", String(stored.index.vector_dim)))
    entries.append(
        ManifestEntry("document_count", String(stored.index.document_count))
    )
    entries.append(
        ManifestEntry("query_divisor", String(stored.query_projection.query_divisor))
    )
    entries.append(
        ManifestEntry(
            "projection_block_count", String(len(stored.query_projection.blocks))
        )
    )
    for block_index in range(len(stored.query_projection.blocks)):
        var block = stored.query_projection.blocks[block_index].copy()
        entries.append(
            ManifestEntry(
                "projection_block_" + String(block_index) + "_order_kind",
                block.order_kind,
            )
        )
        entries.append(
            ManifestEntry(
                "projection_block_" + String(block_index) + "_activation_kind",
                block.activation_kind,
            )
        )
        entries.append(
            ManifestEntry(
                "projection_block_" + String(block_index) + "_input_dim",
                String(block.input_dim),
            )
        )
        entries.append(
            ManifestEntry(
                "projection_block_" + String(block_index) + "_output_dim",
                String(block.output_dim),
            )
        )
        entries.append(
            ManifestEntry(
                "projection_block_" + String(block_index)
                + "_activation_output_scale",
                String(block.activation_output_scale),
            )
        )
        entries.append(
            ManifestEntry(
                "projection_block_" + String(block_index) + "_layer_norm_affine",
                bool_manifest_string(block.layer_norm_affine),
            )
        )
        entries.append(
            ManifestEntry(
                "projection_block_" + String(block_index) + "_layer_norm_epsilon",
                String(block.layer_norm_epsilon),
            )
        )
    entries.append(ManifestEntry("artifact_byte_size", String(artifact_byte_size)))
    return entries^


def save_stored_latent_proxy_index(
    root: Path,
    read stored: StoredLatentProxyIndex,
    vector_payload_encoding: String = VECTOR_PAYLOAD_ENCODING_BINARY_LE,
) raises:
    require_supported_packed_index_vector_payload_encoding(vector_payload_encoding)
    makedirs(root, exist_ok=True)
    write_manifest(
        latent_proxy_manifest_path(root),
        latent_proxy_manifest_entries(stored, vector_payload_encoding, 0),
    )

    var doc_id_lines = String()
    for doc_id in stored.index.doc_ids:
        append_line(doc_id_lines, doc_id)
    latent_proxy_doc_ids_path(root).write_text(doc_id_lines)

    write_binary_vector_payload_with_encoding(
        latent_proxy_vectors_path(root),
        stored.index.proxy_vectors,
        vector_payload_encoding,
    )
    for block_index in range(len(stored.query_projection.blocks)):
        var block = stored.query_projection.blocks[block_index].copy()
        var linear_bias_rows = List[List[Float32]]()
        linear_bias_rows.append(block.linear_bias.copy())
        write_binary_vector_payload_with_encoding(
            latent_proxy_block_linear_rows_path(root, block_index),
            block.linear_rows,
            vector_payload_encoding,
        )
        write_binary_vector_payload_with_encoding(
            latent_proxy_block_linear_bias_path(root, block_index),
            linear_bias_rows,
            vector_payload_encoding,
        )
        if block.layer_norm_affine:
            var layer_norm_weight_rows = List[List[Float32]]()
            layer_norm_weight_rows.append(block.layer_norm_weight.copy())
            var layer_norm_bias_rows = List[List[Float32]]()
            layer_norm_bias_rows.append(block.layer_norm_bias.copy())
            write_binary_vector_payload_with_encoding(
                latent_proxy_block_layer_norm_weight_path(root, block_index),
                layer_norm_weight_rows,
                vector_payload_encoding,
            )
            write_binary_vector_payload_with_encoding(
                latent_proxy_block_layer_norm_bias_path(root, block_index),
                layer_norm_bias_rows,
                vector_payload_encoding,
            )
    var artifact_byte_size = latent_proxy_storage_byte_size(root)
    while True:
        write_manifest(
            latent_proxy_manifest_path(root),
            latent_proxy_manifest_entries(
                stored, vector_payload_encoding, artifact_byte_size
            ),
        )
        var measured = latent_proxy_storage_byte_size(root)
        if measured == artifact_byte_size:
            break
        artifact_byte_size = measured


def load_single_vector_payload(
    path: Path,
    vector_dim: Int,
    vector_payload_encoding: String,
    owner: String,
) raises -> List[Float32]:
    var rows = read_binary_vector_payload_with_encoding(
        path, vector_dim, vector_payload_encoding
    )
    if len(rows) != 1:
        raise Error(owner + " must contain exactly one vector")
    return rows[0].copy()


def load_stored_latent_proxy_index(root: Path) raises -> StoredLatentProxyIndex:
    var manifest = read_manifest(latent_proxy_manifest_path(root))
    _ = require_supported_storage_format(manifest)
    if require_manifest_value(manifest, "artifact_kind") != LATENT_PROXY_ARTIFACT_KIND:
        raise Error("storage artifact is not a latent proxy index")
    var input_vector_dim = parse_int(
        require_manifest_value(manifest, "input_vector_dim"),
        "input_vector_dim",
    )
    var latent_dim = parse_int(
        require_manifest_value(manifest, "latent_dim"),
        "latent_dim",
    )
    var vector_payload_encoding = require_manifest_value(
        manifest, "vector_payload_encoding"
    )
    var doc_ids = read_non_empty_lines(latent_proxy_doc_ids_path(root))
    var proxy_vectors = read_binary_vector_payload_with_encoding(
        latent_proxy_vectors_path(root),
        latent_dim,
        vector_payload_encoding,
    )
    if len(doc_ids) != len(proxy_vectors):
        raise Error("stored latent proxy doc_ids length does not match payload")
    var block_count = parse_int(
        require_manifest_value(manifest, "projection_block_count"),
        "projection_block_count",
    )
    var blocks = List[LatentQueryProjectionBlock]()
    for block_index in range(block_count):
        var block_prefix = "projection_block_" + String(block_index) + "_"
        var output_dim = parse_int(
            require_manifest_value(manifest, block_prefix + "output_dim"),
            block_prefix + "output_dim",
        )
        var layer_norm_affine = (
            require_manifest_value(manifest, block_prefix + "layer_norm_affine")
            == "1"
        )
        var layer_norm_weight = List[Float32]()
        var layer_norm_bias = List[Float32]()
        if layer_norm_affine:
            layer_norm_weight = load_single_vector_payload(
                latent_proxy_block_layer_norm_weight_path(root, block_index),
                output_dim,
                vector_payload_encoding,
                block_prefix + "layer_norm_weight",
            )
            layer_norm_bias = load_single_vector_payload(
                latent_proxy_block_layer_norm_bias_path(root, block_index),
                output_dim,
                vector_payload_encoding,
                block_prefix + "layer_norm_bias",
            )
        blocks.append(
            LatentQueryProjectionBlock(
                require_manifest_value(manifest, block_prefix + "order_kind"),
                require_manifest_value(manifest, block_prefix + "activation_kind"),
                parse_int(
                    require_manifest_value(manifest, block_prefix + "input_dim"),
                    block_prefix + "input_dim",
                ),
                output_dim,
                read_binary_vector_payload_with_encoding(
                    latent_proxy_block_linear_rows_path(root, block_index),
                    parse_int(
                        require_manifest_value(manifest, block_prefix + "input_dim"),
                        block_prefix + "input_dim",
                    ),
                    vector_payload_encoding,
                ),
                load_single_vector_payload(
                    latent_proxy_block_linear_bias_path(root, block_index),
                    output_dim,
                    vector_payload_encoding,
                    block_prefix + "linear_bias",
                ),
                parse_vector_scalar(
                    require_manifest_value(
                        manifest,
                        block_prefix + "activation_output_scale",
                    ),
                    block_prefix + "activation_output_scale",
                ),
                layer_norm_affine,
                parse_vector_scalar(
                    require_manifest_value(manifest, block_prefix + "layer_norm_epsilon"),
                    block_prefix + "layer_norm_epsilon",
                ),
                layer_norm_weight^,
                layer_norm_bias^,
            )
        )
    return StoredLatentProxyIndex(
        require_manifest_value(manifest, "dataset_id"),
        require_manifest_value(manifest, "model_name"),
        VECTOR_SCALAR_NAME,
        input_vector_dim,
        parse_int(
            require_manifest_value(manifest, "artifact_byte_size"),
            "artifact_byte_size",
        ),
        LatentQueryProjection(
            input_vector_dim,
            latent_dim,
            parse_vector_scalar(
                require_manifest_value(manifest, "query_divisor"),
                "query_divisor",
            ),
            blocks^,
        ),
        LatentProxyIndex(doc_ids^, proxy_vectors^, latent_dim),
    )
