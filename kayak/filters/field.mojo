# Metadata field reference for service-side filtering.

from kayak.collections.validation import require_non_empty_string


struct FilterField(Copyable):
    var name: String

    def __init__(out self, name: String) raises:
        self.name = require_non_empty_string(name, "filter field")
