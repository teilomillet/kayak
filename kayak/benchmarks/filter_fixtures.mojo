from std.collections import List

from kayak.filters import FilterExpression, one_of_filter
from kayak.numeric import MetricScalar


struct FilterSelectivityFixture(Copyable):
    var name: String
    var layout_name: String
    var filter_expression: FilterExpression
    var document_count: Int
    var matching_document_count: Int
    var segment_count: Int

    def __init__(
        out self,
        var name: String,
        var layout_name: String,
        filter_expression: FilterExpression,
        document_count: Int,
        matching_document_count: Int,
        segment_count: Int,
    ) raises:
        if document_count <= 0:
            raise Error("document_count must be positive")

        if matching_document_count < 0:
            raise Error("matching_document_count must be non-negative")

        if matching_document_count > document_count:
            raise Error(
                "matching_document_count must not exceed document_count"
            )

        if segment_count <= 0:
            raise Error("segment_count must be positive")

        self.name = name^
        self.layout_name = layout_name^
        self.filter_expression = filter_expression.copy()
        self.document_count = document_count
        self.matching_document_count = matching_document_count
        self.segment_count = segment_count

    def selectivity(self) -> MetricScalar:
        return MetricScalar(self.matching_document_count) / MetricScalar(
            self.document_count
        )


def low_selectivity_filter_fixture() raises -> FilterSelectivityFixture:
    return FilterSelectivityFixture(
        "low_selectivity_shared_pool",
        "shared_pool",
        one_of_filter("source", ["wire"]),
        1000,
        20,
        8,
    )


def high_selectivity_filter_fixture() raises -> FilterSelectivityFixture:
    return FilterSelectivityFixture(
        "high_selectivity_tenant_isolated",
        "tenant_isolated",
        one_of_filter("language", ["en"]),
        1000,
        800,
        8,
    )


def default_filter_selectivity_fixtures() raises -> List[FilterSelectivityFixture]:
    return [
        low_selectivity_filter_fixture(),
        high_selectivity_filter_fixture(),
    ]
