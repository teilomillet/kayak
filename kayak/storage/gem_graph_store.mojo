from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.index import (
    GEM_GRAPH_ADAPTIVE_LABEL_POLICY_FIRST_RELEVANT_CLUSTER_RANK,
    GemGraphBuildConfig,
    GemGraphIndex,
    build_gem_graph_index_with_config,
    DEFAULT_GEM_GRAPH_CONSTRUCTION_NEIGHBOR_COUNT,
    DEFAULT_GEM_GRAPH_DEGREE_LIMIT,
)
from kayak.numeric import SCORE_SCALAR_NAME, STORAGE_FORMAT_VERSION, VECTOR_SCALAR_NAME

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
from .metadata import StoredGemGraphIndex, StoredPackedIndex
from .text_codec import append_line, parse_int, read_non_empty_lines
from .vector_payload_encoding import (
    VECTOR_PAYLOAD_ENCODING_BINARY_LE,
    require_supported_packed_index_vector_payload_encoding,
)


def gem_graph_manifest_path(root: Path) -> Path:
    return root / "manifest.tsv"


def file_size_bytes(path: Path) raises -> Int:
    if path.suffix() == ".bin":
        return len(path.read_bytes())
    return path.read_text().byte_length()


def write_int_lines(path: Path, read values: List[Int]) raises:
    var lines = String()
    for value in values:
        append_line(lines, String(value))
    path.write_text(lines)


def read_int_lines(path: Path, name: String) raises -> List[Int]:
    var values = List[Int]()
    for line in read_non_empty_lines(path):
        values.append(parse_int(line, name))
    return values^


def load_optional_manifest_value(read entries: List[ManifestEntry], key: String) -> String:
    for entry in entries:
        if entry.key == key:
            return entry.value.copy()
    return ""


def bool_text(value: Bool) -> String:
    if value:
        return "true"
    return "false"


def parse_optional_bool_manifest_value(
    read entries: List[ManifestEntry], key: String
) raises -> Bool:
    var value = load_optional_manifest_value(entries, key)
    if value == "":
        return False
    if value == "true" or value == "True":
        return True
    if value == "false" or value == "False":
        return False
    raise Error(key + " manifest value must be true or false")


def write_score_lines(path: Path, read values: List[Float32]) raises:
    var lines = String()
    for value in values:
        append_line(lines, String(value))
    path.write_text(lines)


def read_score_lines(path: Path, owner: String) raises -> List[Float32]:
    if SCORE_SCALAR_NAME != "Float32":
        raise Error("gem graph score storage currently expects Float32")

    var values = List[Float32]()
    for line in read_non_empty_lines(path):
        try:
            values.append(Float32(atof(line)))
        except:
            raise Error(owner + " score is not a valid Float32: " + line)
    return values^


def gem_graph_index_exists(root: Path) -> Bool:
    return gem_graph_manifest_path(root).exists()


def gem_graph_storage_byte_size(root: Path) raises -> Int:
    var total = file_size_bytes(root / "manifest.tsv")
    total += file_size_bytes(root / "doc_ids.tsv")
    total += file_size_bytes(root / "doc_code_offsets.tsv")
    total += file_size_bytes(root / "doc_code_ids.tsv")
    total += file_size_bytes(root / "doc_code_counts.tsv")
    total += file_size_bytes(root / "quantization_centroids.bin")
    total += file_size_bytes(root / "index_centroids.bin")
    total += file_size_bytes(root / "quantization_to_index.tsv")
    total += file_size_bytes(root / "doc_profile_offsets.tsv")
    total += file_size_bytes(root / "doc_profile_cluster_ids.tsv")
    total += file_size_bytes(root / "doc_profile_scores.tsv")
    total += file_size_bytes(root / "cluster_offsets.tsv")
    total += file_size_bytes(root / "cluster_doc_indices.tsv")
    total += file_size_bytes(root / "entry_doc_indices.tsv")
    total += file_size_bytes(root / "neighbor_offsets.tsv")
    total += file_size_bytes(root / "neighbor_doc_indices.tsv")
    return total


def count_entry_points(read index: GemGraphIndex) -> Int:
    var total = 0
    for entry_doc in index.entry_doc_indices:
        if entry_doc != -1:
            total += 1
    return total


def build_stored_gem_graph_index(
    read stored_packed_index: StoredPackedIndex,
    fine_cluster_count: Int,
    coarse_cluster_count: Int,
    cluster_cutoff: Int,
    construction_neighbor_count: Int = DEFAULT_GEM_GRAPH_CONSTRUCTION_NEIGHBOR_COUNT,
    degree_limit: Int = DEFAULT_GEM_GRAPH_DEGREE_LIMIT,
) raises -> StoredGemGraphIndex:
    var config = GemGraphBuildConfig(
        fine_cluster_count,
        coarse_cluster_count,
        cluster_cutoff,
        construction_neighbor_count,
        degree_limit,
    )
    # This convenience builder only exposes graph-density knobs, so keep the
    # latent shortcut candidate budget aligned with that explicit request.
    config.shortcut_candidate_k = construction_neighbor_count
    return build_stored_gem_graph_index_with_config(
        stored_packed_index,
        config,
    )


def build_stored_gem_graph_index_with_config(
    read stored_packed_index: StoredPackedIndex, read config: GemGraphBuildConfig
) raises -> StoredGemGraphIndex:
    var index = build_gem_graph_index_with_config(
        stored_packed_index.index,
        config,
    )
    var adaptive_label_policy = index.adaptive_label_policy.copy()
    return StoredGemGraphIndex(
        stored_packed_index.dataset_id.copy(),
        stored_packed_index.model_name.copy(),
        stored_packed_index.vector_scalar_name.copy(),
        index.cluster_cutoff,
        index.adaptive_cluster_cutoff_enabled,
        index.adaptive_cluster_cutoff_max,
        index.construction_neighbor_count,
        index.degree_limit,
        index.shortcut_candidate_k,
        index.shortcuts_enabled,
        index.document_count,
        index.cluster_count,
        index.graph_edge_count,
        index.shortcut_edge_count,
        count_entry_points(index),
        index.quantization_centroid_count,
        0,
        index^,
        adaptive_label_policy^,
    )


def write_gem_graph_manifest(
    root: Path,
    read stored: StoredGemGraphIndex,
    vector_payload_encoding: String,
    artifact_byte_size: Int,
) raises:
    write_manifest(
        gem_graph_manifest_path(root),
        [
            ManifestEntry("format_version", String(STORAGE_FORMAT_VERSION)),
            ManifestEntry("artifact_kind", "gem_graph_index"),
            ManifestEntry("vector_scalar_name", stored.vector_scalar_name),
            ManifestEntry("dataset_id", stored.dataset_id),
            ManifestEntry("model_name", stored.model_name),
            ManifestEntry("vector_payload_encoding", vector_payload_encoding),
            ManifestEntry("vector_dim", String(stored.index.vector_dim)),
            ManifestEntry("cluster_cutoff", String(stored.cluster_cutoff)),
            ManifestEntry(
                "adaptive_cluster_cutoff_enabled",
                bool_text(stored.adaptive_cluster_cutoff_enabled),
            ),
            ManifestEntry(
                "adaptive_cluster_cutoff_max",
                String(stored.adaptive_cluster_cutoff_max),
            ),
            ManifestEntry(
                "adaptive_label_policy",
                stored.adaptive_label_policy,
            ),
            ManifestEntry(
                "construction_neighbor_count",
                String(stored.construction_neighbor_count),
            ),
            ManifestEntry("degree_limit", String(stored.degree_limit)),
            ManifestEntry(
                "shortcut_candidate_k",
                String(stored.shortcut_candidate_k),
            ),
            ManifestEntry("shortcuts_enabled", bool_text(stored.shortcuts_enabled)),
            ManifestEntry("document_count", String(stored.document_count)),
            ManifestEntry("cluster_count", String(stored.cluster_count)),
            ManifestEntry("graph_edge_count", String(stored.graph_edge_count)),
            ManifestEntry(
                "shortcut_edge_count", String(stored.shortcut_edge_count)
            ),
            ManifestEntry("entry_point_count", String(stored.entry_point_count)),
            ManifestEntry(
                "quantization_centroid_count",
                String(stored.quantization_centroid_count),
            ),
            ManifestEntry("artifact_byte_size", String(artifact_byte_size)),
        ],
    )


def save_stored_gem_graph_index(
    root: Path,
    read stored: StoredGemGraphIndex,
    vector_payload_encoding: String = VECTOR_PAYLOAD_ENCODING_BINARY_LE,
) raises:
    require_supported_packed_index_vector_payload_encoding(vector_payload_encoding)
    makedirs(root, exist_ok=True)

    write_gem_graph_manifest(root, stored, vector_payload_encoding, 0)

    var doc_id_lines = String()
    for doc_id in stored.index.doc_ids:
        append_line(doc_id_lines, doc_id)
    (root / "doc_ids.tsv").write_text(doc_id_lines)

    write_int_lines(root / "doc_code_offsets.tsv", stored.index.doc_code_offsets)
    write_int_lines(root / "doc_code_ids.tsv", stored.index.doc_code_ids)
    write_int_lines(root / "doc_code_counts.tsv", stored.index.doc_code_counts)
    write_binary_vector_payload_with_encoding(
        root / "quantization_centroids.bin",
        stored.index.quantization_centroids,
        vector_payload_encoding,
    )
    write_binary_vector_payload_with_encoding(
        root / "index_centroids.bin",
        stored.index.index_centroids,
        vector_payload_encoding,
    )
    write_int_lines(
        root / "quantization_to_index.tsv", stored.index.quantization_to_index
    )
    write_int_lines(root / "doc_profile_offsets.tsv", stored.index.doc_profile_offsets)
    write_int_lines(
        root / "doc_profile_cluster_ids.tsv", stored.index.doc_profile_cluster_ids
    )
    write_score_lines(root / "doc_profile_scores.tsv", stored.index.doc_profile_scores)
    write_int_lines(root / "cluster_offsets.tsv", stored.index.cluster_offsets)
    write_int_lines(root / "cluster_doc_indices.tsv", stored.index.cluster_doc_indices)
    write_int_lines(root / "entry_doc_indices.tsv", stored.index.entry_doc_indices)
    write_int_lines(root / "neighbor_offsets.tsv", stored.index.neighbor_offsets)
    write_int_lines(root / "neighbor_doc_indices.tsv", stored.index.neighbor_doc_indices)

    write_gem_graph_manifest(
        root,
        stored,
        vector_payload_encoding,
        gem_graph_storage_byte_size(root),
    )


def load_stored_gem_graph_index(root: Path) raises -> StoredGemGraphIndex:
    var manifest = read_manifest(gem_graph_manifest_path(root))
    _ = require_supported_storage_format(manifest)

    if require_manifest_value(manifest, "artifact_kind") != "gem_graph_index":
        raise Error("storage artifact is not a gem graph index")

    var vector_dim = parse_int(
        require_manifest_value(manifest, "vector_dim"), "vector_dim"
    )
    var vector_payload_encoding = require_manifest_value(
        manifest, "vector_payload_encoding"
    )
    var doc_ids = read_non_empty_lines(root / "doc_ids.tsv")
    var doc_code_offsets = read_int_lines(root / "doc_code_offsets.tsv", "doc_code_offset")
    var doc_code_ids = read_int_lines(root / "doc_code_ids.tsv", "doc_code_id")
    var doc_code_counts = read_int_lines(root / "doc_code_counts.tsv", "doc_code_count")
    var quantization_centroids = read_binary_vector_payload_with_encoding(
        root / "quantization_centroids.bin",
        vector_dim,
        vector_payload_encoding,
    )
    var index_centroids = read_binary_vector_payload_with_encoding(
        root / "index_centroids.bin",
        vector_dim,
        vector_payload_encoding,
    )
    var quantization_to_index = read_int_lines(
        root / "quantization_to_index.tsv", "quantization_to_index"
    )
    var doc_profile_offsets = read_int_lines(
        root / "doc_profile_offsets.tsv", "doc_profile_offset"
    )
    var doc_profile_cluster_ids = read_int_lines(
        root / "doc_profile_cluster_ids.tsv", "doc_profile_cluster_id"
    )
    var doc_profile_scores = read_score_lines(
        root / "doc_profile_scores.tsv", "doc_profile"
    )
    var cluster_offsets = read_int_lines(root / "cluster_offsets.tsv", "cluster_offset")
    var cluster_doc_indices = read_int_lines(
        root / "cluster_doc_indices.tsv", "cluster_doc_index"
    )
    var entry_doc_indices = read_int_lines(
        root / "entry_doc_indices.tsv", "entry_doc_index"
    )
    var neighbor_offsets = read_int_lines(root / "neighbor_offsets.tsv", "neighbor_offset")
    var neighbor_doc_indices = read_int_lines(
        root / "neighbor_doc_indices.tsv", "neighbor_doc_index"
    )
    var adaptive_cluster_cutoff_max_text = load_optional_manifest_value(
        manifest, "adaptive_cluster_cutoff_max"
    )
    if adaptive_cluster_cutoff_max_text == "":
        adaptive_cluster_cutoff_max_text = require_manifest_value(
            manifest, "cluster_cutoff"
        )
    var adaptive_cluster_cutoff_max = parse_int(
        adaptive_cluster_cutoff_max_text, "adaptive_cluster_cutoff_max"
    )
    var adaptive_label_policy = load_optional_manifest_value(
        manifest, "adaptive_label_policy"
    )
    if adaptive_label_policy == "":
        adaptive_label_policy = (
            GEM_GRAPH_ADAPTIVE_LABEL_POLICY_FIRST_RELEVANT_CLUSTER_RANK
        )
    var construction_neighbor_count = parse_int(
        require_manifest_value(manifest, "construction_neighbor_count"),
        "construction_neighbor_count",
    )
    var degree_limit = parse_int(
        require_manifest_value(manifest, "degree_limit"), "degree_limit"
    )
    var shortcut_candidate_k_text = load_optional_manifest_value(
        manifest, "shortcut_candidate_k"
    )
    if shortcut_candidate_k_text == "":
        # Older artifacts did not persist this scalar. Fallback to the
        # construction neighbor count, which was also the historical default.
        shortcut_candidate_k_text = String(construction_neighbor_count)
    var shortcut_candidate_k = parse_int(
        shortcut_candidate_k_text, "shortcut_candidate_k"
    )

    var index = GemGraphIndex(
        doc_ids^,
        doc_code_offsets^,
        doc_code_ids^,
        doc_code_counts^,
        quantization_centroids^,
        index_centroids^,
        quantization_to_index^,
        doc_profile_offsets^,
        doc_profile_cluster_ids^,
        doc_profile_scores^,
        cluster_offsets^,
        cluster_doc_indices^,
        entry_doc_indices^,
        neighbor_offsets^,
        neighbor_doc_indices^,
        vector_dim,
        parse_int(
            require_manifest_value(manifest, "shortcut_edge_count"),
            "shortcut_edge_count",
        ),
        parse_int(require_manifest_value(manifest, "cluster_cutoff"), "cluster_cutoff"),
        parse_optional_bool_manifest_value(
            manifest, "adaptive_cluster_cutoff_enabled"
        ),
        adaptive_cluster_cutoff_max,
        construction_neighbor_count,
        degree_limit,
        parse_optional_bool_manifest_value(manifest, "shortcuts_enabled"),
        shortcut_candidate_k,
        adaptive_label_policy.copy(),
    )

    return StoredGemGraphIndex(
        require_manifest_value(manifest, "dataset_id"),
        require_manifest_value(manifest, "model_name"),
        VECTOR_SCALAR_NAME,
        parse_int(require_manifest_value(manifest, "cluster_cutoff"), "cluster_cutoff"),
        parse_optional_bool_manifest_value(
            manifest, "adaptive_cluster_cutoff_enabled"
        ),
        adaptive_cluster_cutoff_max,
        construction_neighbor_count,
        degree_limit,
        shortcut_candidate_k,
        parse_optional_bool_manifest_value(manifest, "shortcuts_enabled"),
        parse_int(require_manifest_value(manifest, "document_count"), "document_count"),
        parse_int(require_manifest_value(manifest, "cluster_count"), "cluster_count"),
        parse_int(
            require_manifest_value(manifest, "graph_edge_count"),
            "graph_edge_count",
        ),
        parse_int(
            require_manifest_value(manifest, "shortcut_edge_count"),
            "shortcut_edge_count",
        ),
        parse_int(
            require_manifest_value(manifest, "entry_point_count"),
            "entry_point_count",
        ),
        parse_int(
            require_manifest_value(manifest, "quantization_centroid_count"),
            "quantization_centroid_count",
        ),
        parse_int(
            require_manifest_value(manifest, "artifact_byte_size"),
            "artifact_byte_size",
        ),
        index^,
        adaptive_label_policy^,
    )
