# Explicit candidate-window artifact materialization for one search stage.


struct StageArtifactMaterialization(Copyable):
    var family: String
    var segment_count: Int
    var document_count: Int
    var token_count: Int
    var vector_count: Int
    var byte_size: Int

    def __init__(
        out self,
        var family: String,
        segment_count: Int,
        document_count: Int,
        token_count: Int,
        vector_count: Int,
        byte_size: Int,
    ) raises:
        if family.byte_length() == 0:
            raise Error("stage artifact materialization family must be non-empty")

        if segment_count < 0:
            raise Error(
                "stage artifact materialization segment_count must be non-negative"
            )

        if document_count < 0:
            raise Error(
                "stage artifact materialization document_count must be non-negative"
            )

        if token_count < 0:
            raise Error(
                "stage artifact materialization token_count must be non-negative"
            )

        if vector_count < 0:
            raise Error(
                "stage artifact materialization vector_count must be non-negative"
            )

        if byte_size < 0:
            raise Error(
                "stage artifact materialization byte_size must be non-negative"
            )

        self.family = family^
        self.segment_count = segment_count
        self.document_count = document_count
        self.token_count = token_count
        self.vector_count = vector_count
        self.byte_size = byte_size
