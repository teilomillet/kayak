# Explicit collection-level physical layout families.
#
# This module owns the persisted contract for whether collection isolation comes
# from the storage layout itself or must be re-applied at query time. It does
# not own per-segment filter-index storage or planner policy.

from .validation import require_non_empty_string


comptime COLLECTION_LAYOUT_FAMILY_TENANT_ISOLATED = "tenant_isolated"
comptime COLLECTION_LAYOUT_FAMILY_SHARED_POOL = "shared_pool"


def default_collection_layout_family() -> String:
    return COLLECTION_LAYOUT_FAMILY_TENANT_ISOLATED


def require_collection_layout_family_supported(
    collection_layout_family: String,
) raises -> String:
    var normalized = require_non_empty_string(
        collection_layout_family,
        "collection_layout_family",
    )
    if normalized == COLLECTION_LAYOUT_FAMILY_TENANT_ISOLATED:
        return normalized
    if normalized == COLLECTION_LAYOUT_FAMILY_SHARED_POOL:
        return normalized

    raise Error(
        "unsupported collection_layout_family: " + normalized
    )


def collection_layout_family_is_shared_pool(
    collection_layout_family: String,
) -> Bool:
    return collection_layout_family == COLLECTION_LAYOUT_FAMILY_SHARED_POOL
