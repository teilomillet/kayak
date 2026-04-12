from std.pathlib import Path

from kayak import (
    CollectionId,
    CreateCollectionRequest,
    CreateSnapshotRequest,
    EncodedDocument,
    EncodedQuery,
    ExactCpuBackend,
    NamespaceId,
    SnapshotId,
    TenantId,
    UpsertDocument,
    UpsertDocumentsRequest,
    VECTOR_SCALAR_NAME,
    create_collection,
    create_snapshot,
    debug_search_response_json,
    default_exact_search_request,
    execute_debug_search,
    upsert_documents,
)


def unique_service_root(prefix: String) -> Path:
    var suffix = 0

    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1


def main() raises:
    var service_root = unique_service_root("kayak-hosted-collection-smoke")
    _ = create_collection(
        service_root,
        CreateCollectionRequest(
            CollectionId("demo"),
            TenantId("public"),
            NamespaceId("scratch"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("demo"),
            TenantId("public"),
            NamespaceId("scratch"),
            [
                UpsertDocument(
                    EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
                    "alpha",
                ),
                UpsertDocument(
                    EncodedDocument("doc-b", [[0.0, 1.0], [0.0, 1.0]]),
                    "beta",
                ),
            ],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("demo"),
            TenantId("public"),
            NamespaceId("scratch"),
            SnapshotId("snapshot-0001"),
            "demo snapshot",
        ),
    )

    var response = execute_debug_search(
        ExactCpuBackend(),
        service_root,
        default_exact_search_request(
            CollectionId("demo"),
            TenantId("public"),
            NamespaceId("scratch"),
            SnapshotId("snapshot-0001"),
            EncodedQuery([[1.0, 0.0], [0.0, 1.0]]),
            2,
            True,
        ),
    )
    print(debug_search_response_json(response))
