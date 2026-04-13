from std.pathlib import Path

from kayak.collections import CollectionId, NamespaceId, TenantId


def service_collections_root(service_root: Path) -> Path:
    return service_root / "collections"


def service_collection_root(
    service_root: Path,
    tenant_id: TenantId,
    namespace_id: NamespaceId,
    collection_id: CollectionId,
) -> Path:
    return (
        service_collections_root(service_root)
        / tenant_id.value
        / namespace_id.value
        / collection_id.value
    )


def draft_state_root(collection_root: Path) -> Path:
    return collection_root / "draft"


def draft_state_manifest_path(draft_root: Path) -> Path:
    return draft_root / "manifest.tsv"


def draft_state_packed_index_root(draft_root: Path) -> Path:
    return draft_root / "packed_index"


def draft_state_text_corpus_root(draft_root: Path) -> Path:
    return draft_root / "text_corpus"


def draft_state_document_metadata_root(draft_root: Path) -> Path:
    return draft_root / "document_metadata"


def draft_state_mutations_root(draft_root: Path) -> Path:
    return draft_root / "mutations"


def draft_state_mutation_root(draft_root: Path, mutation_index: Int) -> Path:
    return draft_state_mutations_root(draft_root) / (
        "mutation-" + String(mutation_index)
    )


def draft_state_mutation_manifest_path(mutation_root: Path) -> Path:
    return mutation_root / "manifest.tsv"


def draft_state_mutation_packed_index_root(mutation_root: Path) -> Path:
    return mutation_root / "packed_index"


def draft_state_mutation_text_corpus_root(mutation_root: Path) -> Path:
    return mutation_root / "text_corpus"


def draft_state_mutation_document_metadata_root(mutation_root: Path) -> Path:
    return mutation_root / "document_metadata"


def draft_state_mutation_doc_ids_path(mutation_root: Path) -> Path:
    return mutation_root / "doc_ids.tsv"
