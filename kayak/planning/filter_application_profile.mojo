# Candidate-stage filter application summary.
#
# This module owns measurable facts about how stage 1 narrowed the document
# search space. It does not own filter execution itself.

from kayak.filters import (
    FilterExpression,
    filter_expression_uses_internal_logical_scope,
)
from kayak.numeric import MetricScalar


struct FilterApplicationProfile(Copyable):
    var public_filter_applied: Bool
    var logical_scope_applied: Bool
    var uses_document_filter_index: Bool
    var input_document_count: Int
    var matching_document_count: Int
    var artifact_byte_size: Int

    def __init__(out self):
        self.public_filter_applied = False
        self.logical_scope_applied = False
        self.uses_document_filter_index = False
        self.input_document_count = 0
        self.matching_document_count = 0
        self.artifact_byte_size = 0

    def __init__(
        out self,
        public_filter_applied: Bool,
        logical_scope_applied: Bool,
        uses_document_filter_index: Bool,
        input_document_count: Int,
        matching_document_count: Int,
        artifact_byte_size: Int,
    ) raises:
        if input_document_count < 0:
            raise Error(
                "filter application input_document_count must be non-negative"
            )

        if matching_document_count < 0:
            raise Error(
                "filter application matching_document_count must be non-negative"
            )

        if matching_document_count > input_document_count:
            raise Error(
                "filter application matching_document_count exceeds input_document_count"
            )

        if artifact_byte_size < 0:
            raise Error(
                "filter application artifact_byte_size must be non-negative"
            )

        self.public_filter_applied = public_filter_applied
        self.logical_scope_applied = logical_scope_applied
        self.uses_document_filter_index = uses_document_filter_index
        self.input_document_count = input_document_count
        self.matching_document_count = matching_document_count
        self.artifact_byte_size = artifact_byte_size

    def selectivity(self) -> MetricScalar:
        if self.input_document_count == 0:
            return MetricScalar(0.0)

        return MetricScalar(self.matching_document_count) / MetricScalar(
            self.input_document_count
        )


def identity_filter_application_profile(
    document_count: Int
) raises -> FilterApplicationProfile:
    return FilterApplicationProfile(
        False,
        False,
        False,
        document_count,
        document_count,
        0,
    )


def filter_application_profile_for_effective_filter(
    read request_filter_expression: FilterExpression,
    read effective_filter_expression: FilterExpression,
    input_document_count: Int,
    matching_document_count: Int,
    artifact_byte_size: Int,
    uses_document_filter_index: Bool,
) raises -> FilterApplicationProfile:
    return FilterApplicationProfile(
        not request_filter_expression.is_match_all(),
        filter_expression_uses_internal_logical_scope(
            effective_filter_expression
        ),
        uses_document_filter_index,
        input_document_count,
        matching_document_count,
        artifact_byte_size,
    )
