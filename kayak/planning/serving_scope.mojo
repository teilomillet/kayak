# Search-time serving-scope semantics.
#
# This module owns the query-time contract for whether collection identity is
# enforced purely by layout or must be re-applied through logical filter
# pushdown. It does not own the filter-expression composition itself.

from kayak.collections import (
    CollectionManifest,
    collection_layout_family_is_shared_pool,
)
from kayak.collections.validation import require_non_empty_string


comptime SEARCH_SERVING_SCOPE_KIND_LAYOUT_ROOTED = "layout_rooted"
comptime SEARCH_SERVING_SCOPE_KIND_LOGICAL_FILTER_PUSHDOWN = (
    "logical_filter_pushdown"
)


struct SearchServingScope(Copyable):
    var kind: String
    var requires_logical_scope_pushdown: Bool

    def __init__(
        out self,
        var kind: String,
        requires_logical_scope_pushdown: Bool,
    ) raises:
        self.kind = require_non_empty_string(kind, "search_serving_scope kind")
        self.requires_logical_scope_pushdown = requires_logical_scope_pushdown


def layout_rooted_search_serving_scope() raises -> SearchServingScope:
    return SearchServingScope(
        SEARCH_SERVING_SCOPE_KIND_LAYOUT_ROOTED,
        False,
    )


def logical_filter_pushdown_search_serving_scope() raises -> SearchServingScope:
    return SearchServingScope(
        SEARCH_SERVING_SCOPE_KIND_LOGICAL_FILTER_PUSHDOWN,
        True,
    )


def search_serving_scope_for_collection(
    read collection: CollectionManifest
) raises -> SearchServingScope:
    if collection_layout_family_is_shared_pool(collection.collection_layout_family):
        return logical_filter_pushdown_search_serving_scope()

    return layout_rooted_search_serving_scope()
