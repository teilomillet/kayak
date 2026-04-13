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
from .collection_layout import default_collection_layout_family
from .collection import CollectionManifest
from .ids import CollectionId, NamespaceId, TenantId
from .manifest_util import load_optional_manifest_value
from .paths import collection_manifest_path
from .search_artifact_policy import (
    SearchArtifactBuildPolicy,
    SearchArtifactBuildConfigEntry,
    SearchArtifactBuildSpec,
    default_search_artifact_build_policy,
)


def collection_manifest_exists(root: Path) -> Bool:
    return collection_manifest_path(root).exists()


def save_collection_manifest(root: Path, read manifest: CollectionManifest) raises:
    makedirs(root, exist_ok=True)

    var entries = List[ManifestEntry]()
    entries.append(ManifestEntry("collection_id", manifest.collection_id.value))
    entries.append(ManifestEntry("tenant_id", manifest.tenant_id.value))
    entries.append(ManifestEntry("namespace_id", manifest.namespace_id.value))
    entries.append(
        ManifestEntry(
            "collection_layout_family",
            manifest.collection_layout_family,
        )
    )
    entries.append(ManifestEntry("model_name", manifest.model_name))
    entries.append(
        ManifestEntry("vector_scalar_name", manifest.vector_scalar_name)
    )
    entries.append(ManifestEntry("vector_dim", String(manifest.vector_dim)))
    entries.append(
        ManifestEntry("latest_generation", String(manifest.latest_generation))
    )
    entries.append(
        ManifestEntry(
            "default_keep_latest_inactive_count",
            String(manifest.default_keep_latest_inactive_count),
        )
    )
    entries.append(
        ManifestEntry(
            "search_artifact_build_count",
            String(len(manifest.search_artifact_build_policy.stage1_artifacts)),
        )
    )
    for index in range(len(manifest.search_artifact_build_policy.stage1_artifacts)):
        var spec = (
            manifest.search_artifact_build_policy.stage1_artifacts[index].copy()
        )
        entries.append(
            ManifestEntry(
                "search_artifact_build_" + String(index) + "_family",
                spec.family,
            )
        )
        entries.append(
            ManifestEntry(
                "search_artifact_build_" + String(index) + "_root",
                spec.root,
            )
        )
        entries.append(
            ManifestEntry(
                "search_artifact_build_" + String(index) + "_config_count",
                String(len(spec.config)),
            )
        )
        for config_index in range(len(spec.config)):
            var entry = spec.config[config_index].copy()
            entries.append(
                ManifestEntry(
                    "search_artifact_build_"
                        + String(index)
                        + "_config_"
                        + String(config_index)
                        + "_key",
                    entry.key,
                )
            )
            entries.append(
                ManifestEntry(
                    "search_artifact_build_"
                        + String(index)
                        + "_config_"
                        + String(config_index)
                        + "_value",
                    entry.value,
                )
            )
    if manifest.active_snapshot_id.byte_length() != 0:
        entries.append(
            ManifestEntry("active_snapshot_id", manifest.active_snapshot_id)
        )

    write_collection_artifact_manifest(
        collection_manifest_path(root), "collection_manifest", entries
    )


def load_collection_manifest(root: Path) raises -> CollectionManifest:
    var entries = read_collection_artifact_manifest(
        collection_manifest_path(root), "collection_manifest"
    )
    var default_keep_latest_inactive_count = load_optional_manifest_value(
        entries, "default_keep_latest_inactive_count"
    )
    if default_keep_latest_inactive_count.byte_length() == 0:
        default_keep_latest_inactive_count = "1"
    var collection_layout_family = load_optional_manifest_value(
        entries, "collection_layout_family"
    )
    if collection_layout_family.byte_length() == 0:
        collection_layout_family = default_collection_layout_family()
    var search_artifact_build_policy = default_search_artifact_build_policy()
    var search_artifact_build_count = load_optional_manifest_value(
        entries, "search_artifact_build_count"
    )
    if search_artifact_build_count.byte_length() != 0:
        var build_specs = List[SearchArtifactBuildSpec]()
        for index in range(
            parse_int(search_artifact_build_count, "search_artifact_build_count")
        ):
            var config_entries = List[SearchArtifactBuildConfigEntry]()
            var config_count = load_optional_manifest_value(
                entries,
                "search_artifact_build_" + String(index) + "_config_count",
            )
            if config_count.byte_length() != 0:
                for config_index in range(
                    parse_int(
                        config_count,
                        "search_artifact_build_" + String(index) + "_config_count",
                    )
                ):
                    config_entries.append(
                        SearchArtifactBuildConfigEntry(
                            require_manifest_value(
                                entries,
                                "search_artifact_build_"
                                    + String(index)
                                    + "_config_"
                                    + String(config_index)
                                    + "_key",
                            ),
                            require_manifest_value(
                                entries,
                                "search_artifact_build_"
                                    + String(index)
                                    + "_config_"
                                    + String(config_index)
                                    + "_value",
                            ),
                        )
                    )
            build_specs.append(
                SearchArtifactBuildSpec(
                    require_manifest_value(
                        entries,
                        "search_artifact_build_" + String(index) + "_family",
                    ),
                    require_manifest_value(
                        entries,
                        "search_artifact_build_" + String(index) + "_root",
                    ),
                    config_entries,
                )
            )
        search_artifact_build_policy = SearchArtifactBuildPolicy(build_specs)

    return CollectionManifest(
        CollectionId(require_manifest_value(entries, "collection_id")),
        TenantId(require_manifest_value(entries, "tenant_id")),
        NamespaceId(require_manifest_value(entries, "namespace_id")),
        require_manifest_value(entries, "model_name"),
        require_current_vector_scalar_name(entries, "collection manifest"),
        parse_int(require_manifest_value(entries, "vector_dim"), "vector_dim"),
        parse_int(
            require_manifest_value(entries, "latest_generation"),
            "latest_generation",
        ),
        load_optional_manifest_value(entries, "active_snapshot_id"),
        parse_int(
            default_keep_latest_inactive_count,
            "default_keep_latest_inactive_count",
        ),
        search_artifact_build_policy,
        collection_layout_family,
    )
