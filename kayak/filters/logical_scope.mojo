# Internal logical-scope filter helpers for hosted collection identity.
#
# This module owns the reserved filter field names used by the engine to carry
# collection, tenant, and namespace scope inside internal filter expressions.
# It does not own public request validation or stage-execution policy.

from .expression import FilterExpression, and_filter, match_all_filter
from .field import FilterField
from .term import FilterTerm

from kayak.collections.validation import require_non_empty_string


comptime FILTER_FIELD_INTERNAL_COLLECTION_ID = "__kayak_collection_id"
comptime FILTER_FIELD_INTERNAL_TENANT_ID = "__kayak_tenant_id"
comptime FILTER_FIELD_INTERNAL_NAMESPACE_ID = "__kayak_namespace_id"


struct LogicalFilterScope(Copyable):
    var collection_id: String
    var tenant_id: String
    var namespace_id: String

    def __init__(out self):
        self.collection_id = ""
        self.tenant_id = ""
        self.namespace_id = ""

    def __init__(
        out self,
        collection_id: String,
        tenant_id: String,
        namespace_id: String,
    ) raises:
        self.collection_id = require_non_empty_string(
            collection_id, "logical_filter_scope collection_id"
        )
        self.tenant_id = require_non_empty_string(
            tenant_id, "logical_filter_scope tenant_id"
        )
        self.namespace_id = require_non_empty_string(
            namespace_id, "logical_filter_scope namespace_id"
        )

    def is_unscoped(self) -> Bool:
        return (
            self.collection_id.byte_length() == 0
            and self.tenant_id.byte_length() == 0
            and self.namespace_id.byte_length() == 0
        )


def unscoped_logical_filter_scope() -> LogicalFilterScope:
    return LogicalFilterScope()


def filter_field_is_internal_logical_scope(field_name: String) -> Bool:
    return (
        field_name == FILTER_FIELD_INTERNAL_COLLECTION_ID
        or field_name == FILTER_FIELD_INTERNAL_TENANT_ID
        or field_name == FILTER_FIELD_INTERNAL_NAMESPACE_ID
    )


def document_metadata_key_is_reserved_for_logical_scope(key: String) -> Bool:
    return filter_field_is_internal_logical_scope(key)


def logical_scope_value_for_filter_field(
    read scope: LogicalFilterScope, field_name: String
) -> String:
    if field_name == FILTER_FIELD_INTERNAL_COLLECTION_ID:
        return scope.collection_id.copy()
    if field_name == FILTER_FIELD_INTERNAL_TENANT_ID:
        return scope.tenant_id.copy()
    if field_name == FILTER_FIELD_INTERNAL_NAMESPACE_ID:
        return scope.namespace_id.copy()

    return ""


def logical_scope_filter(read scope: LogicalFilterScope) raises -> FilterExpression:
    if scope.is_unscoped():
        return match_all_filter()

    return and_filter(
        [
            FilterTerm(
                FilterField(FILTER_FIELD_INTERNAL_COLLECTION_ID),
                "eq",
                [scope.collection_id.copy()],
            ),
            FilterTerm(
                FilterField(FILTER_FIELD_INTERNAL_TENANT_ID),
                "eq",
                [scope.tenant_id.copy()],
            ),
            FilterTerm(
                FilterField(FILTER_FIELD_INTERNAL_NAMESPACE_ID),
                "eq",
                [scope.namespace_id.copy()],
            ),
        ]
    )


def filter_expression_uses_internal_logical_scope(
    read expression: FilterExpression
) -> Bool:
    for clause in expression.clauses:
        for term in clause.terms:
            if filter_field_is_internal_logical_scope(term.field.name):
                return True

    return False


def require_user_visible_filter_expression(
    read expression: FilterExpression,
    boundary_name: String,
) raises:
    if not filter_expression_uses_internal_logical_scope(expression):
        return

    raise Error(
        boundary_name
        + " must not reference reserved internal logical-scope fields"
    )
