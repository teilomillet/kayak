from std.collections import List


struct DocumentTextCorpus(Copyable):
    var doc_ids: List[String]
    var texts: List[String]

    def __init__(
        out self,
        var doc_ids: List[String],
        var texts: List[String],
    ) raises:
        if len(doc_ids) != len(texts):
            raise Error("document text corpus requires aligned doc ids and texts")

        self.doc_ids = doc_ids^
        self.texts = texts^


def document_text_for_doc_id(
    read corpus: DocumentTextCorpus,
    doc_id: String,
) raises -> String:
    for index in range(len(corpus.doc_ids)):
        if corpus.doc_ids[index] == doc_id:
            return corpus.texts[index].copy()

    raise Error("document text corpus is missing doc id: " + doc_id)
