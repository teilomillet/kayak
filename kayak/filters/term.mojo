from std.collections import List

from kayak.collections.validation import require_non_empty_string

from .field import FilterField


struct FilterTerm(Copyable):
    var field: FilterField
    var operator: String
    var values: List[String]

    def __init__(
        out self,
        field: FilterField,
        operator: String,
        read values: List[String],
    ) raises:
        if operator != "eq" and operator != "one_of":
            raise Error("unsupported filter operator: " + operator)

        if len(values) == 0:
            raise Error("filter term must contain at least one value")

        if operator == "eq" and len(values) != 1:
            raise Error("eq filter term must contain exactly one value")

        var validated_values = List[String]()
        for value in values:
            validated_values.append(
                require_non_empty_string(value, "filter value")
            )

        self.field = field.copy()
        self.operator = operator.copy()
        self.values = validated_values^
