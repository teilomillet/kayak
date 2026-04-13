from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.storage.manifest import ManifestEntry, require_manifest_value
from kayak.storage.text_codec import parse_int

from .artifact_manifest import (
    read_collection_artifact_manifest,
    require_current_vector_scalar_name,
    write_collection_artifact_manifest,
)
from .document_representation_transform import (
    DocumentRepresentationTransformConfigEntry,
    DocumentRepresentationTransformManifest,
)
from .ids import CollectionId, NamespaceId, SegmentId, TenantId
from .manifest_util import load_optional_manifest_value
from .paths import require_relative_artifact_root, segment_manifest_path
from .search_artifact import (
    SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS,
    SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
    SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY,
    SearchArtifactManifest,
    centroid_heads_search_artifact,
    centroid_postings_search_artifact,
    document_proxy_search_artifact,
    search_artifact_root,
)
from .segment import SealedSegmentManifest
from .stats_manifest import (
    load_segment_stats_from_manifest,
    segment_stats_manifest_entries,
)


def sealed_segment_manifest_exists(root: Path) -> Bool:
    return segment_manifest_path(root).exists()


def encode_optional_root(root: String) -> String:
    if root.byte_length() == 0:
        return "-"

    return root.copy()


def decode_optional_root(root: String) -> String:
    if root == "-":
        return ""

    return root.copy()


def save_sealed_segment_manifest(
    root: Path, read manifest: SealedSegmentManifest
) raises:
    makedirs(root, exist_ok=True)
    var packed_index_root = require_relative_artifact_root(
        manifest.packed_index_root, "packed_index_root"
    )
    var validated_search_artifacts = List[SearchArtifactManifest]()
    for artifact in manifest.search_artifacts:
        validated_search_artifacts.append(
            SearchArtifactManifest(
                artifact.family.copy(),
                require_relative_artifact_root(
                    artifact.root, "search_artifact root for " + artifact.family
                ),
            )
        )

    var text_corpus_root = String()
    if manifest.text_corpus_root.byte_length() != 0:
        text_corpus_root = require_relative_artifact_root(
            manifest.text_corpus_root, "text_corpus_root"
        )

    var centroid_postings_root = search_artifact_root(
        validated_search_artifacts, SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS
    )
    var centroid_heads_root = search_artifact_root(
        validated_search_artifacts, SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS
    )
    var document_proxy_root = search_artifact_root(
        validated_search_artifacts, SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY
    )

    var entries = List[ManifestEntry]()
    entries.append(ManifestEntry("segment_id", manifest.segment_id.value))
    entries.append(ManifestEntry("collection_id", manifest.collection_id.value))
    entries.append(ManifestEntry("tenant_id", manifest.tenant_id.value))
    entries.append(ManifestEntry("namespace_id", manifest.namespace_id.value))
    entries.append(ManifestEntry("generation", String(manifest.generation)))
    entries.append(ManifestEntry("model_name", manifest.model_name))
    entries.append(
        ManifestEntry("vector_scalar_name", manifest.vector_scalar_name)
    )
    entries.append(ManifestEntry("vector_dim", String(manifest.vector_dim)))
    entries.append(ManifestEntry("packed_index_root", packed_index_root))
    entries.append(
        ManifestEntry(
            "document_representation_transform_count",
            String(len(manifest.document_representation_transforms)),
        )
    )
    for transform_index in range(len(manifest.document_representation_transforms)):
        var transform = (
            manifest.document_representation_transforms[transform_index].copy()
        )
        entries.append(
            ManifestEntry(
                "document_representation_transform_" + String(transform_index) + "_kind",
                transform.kind,
            )
        )
        entries.append(
            ManifestEntry(
                "document_representation_transform_" + String(transform_index)
                + "_config_count",
                String(len(transform.config)),
            )
        )
        for config_index in range(len(transform.config)):
            entries.append(
                ManifestEntry(
                    "document_representation_transform_" + String(transform_index)
                    + "_config_" + String(config_index) + "_key",
                    transform.config[config_index].key,
                )
            )
            entries.append(
                ManifestEntry(
                    "document_representation_transform_" + String(transform_index)
                    + "_config_" + String(config_index) + "_value",
                    transform.config[config_index].value,
                )
            )
    entries.append(
        ManifestEntry("search_artifact_count", String(len(validated_search_artifacts)))
    )
    for index in range(len(validated_search_artifacts)):
        entries.append(
            ManifestEntry(
                "search_artifact_" + String(index) + "_family",
                validated_search_artifacts[index].family,
            )
        )
        entries.append(
            ManifestEntry(
                "search_artifact_" + String(index) + "_root",
                encode_optional_root(validated_search_artifacts[index].root),
            )
        )
    entries.append(
        ManifestEntry(
            "centroid_postings_root",
            encode_optional_root(centroid_postings_root),
        )
    )
    entries.append(
        ManifestEntry(
            "centroid_heads_root",
            encode_optional_root(centroid_heads_root),
        )
    )
    entries.append(
        ManifestEntry(
            "document_proxy_root",
            encode_optional_root(document_proxy_root),
        )
    )
    entries.append(
        ManifestEntry(
            "text_corpus_root",
            encode_optional_root(text_corpus_root),
        )
    )

    for entry in segment_stats_manifest_entries(manifest.stats):
        entries.append(entry.copy())

    write_collection_artifact_manifest(
        segment_manifest_path(root), "sealed_segment_manifest", entries
    )


def load_sealed_segment_manifest(root: Path) raises -> SealedSegmentManifest:
    var entries = read_collection_artifact_manifest(
        segment_manifest_path(root), "sealed_segment_manifest"
    )
    var document_representation_transforms = List[
        DocumentRepresentationTransformManifest
    ]()
    var document_representation_transform_count_value = load_optional_manifest_value(
        entries, "document_representation_transform_count"
    )
    if document_representation_transform_count_value.byte_length() != 0:
        var document_representation_transform_count = parse_int(
            document_representation_transform_count_value,
            "document_representation_transform_count",
        )
        for transform_index in range(document_representation_transform_count):
            var config = List[DocumentRepresentationTransformConfigEntry]()
            var config_count = parse_int(
                require_manifest_value(
                    entries,
                    "document_representation_transform_" + String(transform_index)
                    + "_config_count",
                ),
                "document_representation_transform config_count",
            )
            for config_index in range(config_count):
                config.append(
                    DocumentRepresentationTransformConfigEntry(
                        require_manifest_value(
                            entries,
                            "document_representation_transform_"
                            + String(transform_index)
                            + "_config_"
                            + String(config_index)
                            + "_key",
                        ),
                        require_manifest_value(
                            entries,
                            "document_representation_transform_"
                            + String(transform_index)
                            + "_config_"
                            + String(config_index)
                            + "_value",
                        ),
                    )
                )
            document_representation_transforms.append(
                DocumentRepresentationTransformManifest(
                    require_manifest_value(
                        entries,
                        "document_representation_transform_" + String(transform_index)
                        + "_kind",
                    ),
                    config^,
                )
            )
    var search_artifacts = List[SearchArtifactManifest]()
    var search_artifact_count_value = load_optional_manifest_value(
        entries, "search_artifact_count"
    )
    if search_artifact_count_value.byte_length() != 0:
        var search_artifact_count = parse_int(
            search_artifact_count_value, "search_artifact_count"
        )
        for index in range(search_artifact_count):
            search_artifacts.append(
                SearchArtifactManifest(
                    require_manifest_value(
                        entries, "search_artifact_" + String(index) + "_family"
                    ),
                    decode_optional_root(
                        require_manifest_value(
                            entries, "search_artifact_" + String(index) + "_root"
                        )
                    ),
                )
            )
    else:
        var centroid_postings_root = decode_optional_root(
            load_optional_manifest_value(entries, "centroid_postings_root")
        )
        if centroid_postings_root.byte_length() != 0:
            search_artifacts.append(
                centroid_postings_search_artifact(centroid_postings_root)
            )

        var centroid_heads_root = decode_optional_root(
            load_optional_manifest_value(entries, "centroid_heads_root")
        )
        if centroid_heads_root.byte_length() != 0:
            search_artifacts.append(centroid_heads_search_artifact(centroid_heads_root))

        var document_proxy_root = decode_optional_root(
            load_optional_manifest_value(entries, "document_proxy_root")
        )
        if document_proxy_root.byte_length() != 0:
            search_artifacts.append(document_proxy_search_artifact(document_proxy_root))

    return SealedSegmentManifest(
        SegmentId(require_manifest_value(entries, "segment_id")),
        CollectionId(require_manifest_value(entries, "collection_id")),
        TenantId(require_manifest_value(entries, "tenant_id")),
        NamespaceId(require_manifest_value(entries, "namespace_id")),
        parse_int(require_manifest_value(entries, "generation"), "generation"),
        require_manifest_value(entries, "model_name"),
        require_current_vector_scalar_name(entries, "sealed segment manifest"),
        parse_int(require_manifest_value(entries, "vector_dim"), "vector_dim"),
        require_manifest_value(entries, "packed_index_root"),
        document_representation_transforms^,
        search_artifacts^,
        decode_optional_root(load_optional_manifest_value(entries, "text_corpus_root")),
        load_segment_stats_from_manifest(entries),
    )
