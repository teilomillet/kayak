from std.collections import List

from kayak.contracts import EncodedDocument
from kayak.eval import JudgedQuery, JudgedTask

from .proxy_vectors import make_proxy_document, make_proxy_query


def single_relevant_doc_id(doc_id: String) -> List[String]:
    var doc_ids = List[String]()
    doc_ids.append(doc_id.copy())
    return doc_ids^


def two_relevant_doc_ids(doc_id_a: String, doc_id_b: String) -> List[String]:
    var doc_ids = List[String]()
    doc_ids.append(doc_id_a.copy())
    doc_ids.append(doc_id_b.copy())
    return doc_ids^


def make_lotte_forum_proxy_task() raises -> JudgedTask:
    var documents = List[EncodedDocument]()
    documents.append(make_proxy_document("lotte-bike-brakes", 16, [0, 1, 2, 12]))
    documents.append(make_proxy_document("lotte-bike-chain", 16, [0, 3, 4, 12]))
    documents.append(make_proxy_document("lotte-latex-bib", 16, [5, 6, 7, 13]))
    documents.append(make_proxy_document("lotte-terminal-usb", 16, [8, 9, 10, 14]))
    documents.append(make_proxy_document("lotte-hiking-pack", 16, [11, 1, 4, 15]))
    documents.append(make_proxy_document("lotte-bike-maintenance", 16, [0, 1, 4, 15]))

    var queries = List[JudgedQuery]()
    queries.append(
        JudgedQuery(
            "forum-bike-brake-noise",
            "Forum-style niche query with a close bike-maintenance distractor.",
            make_proxy_query(16, [0, 1, 2]),
            single_relevant_doc_id("lotte-bike-brakes"),
        )
    )
    queries.append(
        JudgedQuery(
            "forum-latex-citations",
            "Forum-style niche query over technical writing concepts.",
            make_proxy_query(16, [5, 6, 7]),
            single_relevant_doc_id("lotte-latex-bib"),
        )
    )

    return JudgedTask(
        "lotte",
        "forum_proxy",
        "Proxy for LoTTE-style domain forum retrieval with partial distractors.",
        "success",
        5,
        3,
        4,
        16,
        documents^,
        queries^,
    )


def make_beir_fact_proxy_task() raises -> JudgedTask:
    var documents = List[EncodedDocument]()
    documents.append(make_proxy_document("beir-vaccine-antibody", 16, [0, 1, 2, 12]))
    documents.append(make_proxy_document("beir-vaccine-safety", 16, [0, 3, 4, 12]))
    documents.append(make_proxy_document("beir-margin-call", 16, [5, 6, 7, 13]))
    documents.append(make_proxy_document("beir-carbon-tax", 16, [8, 9, 10, 14]))
    documents.append(make_proxy_document("beir-market-volatility", 16, [5, 6, 11, 15]))
    documents.append(make_proxy_document("beir-clinical-trial", 16, [0, 1, 4, 15]))

    var queries = List[JudgedQuery]()
    queries.append(
        JudgedQuery(
            "fact-antibody-evidence",
            "Scientific fact lookup with overlapping biomedical distractors.",
            make_proxy_query(16, [0, 1, 2]),
            single_relevant_doc_id("beir-vaccine-antibody"),
        )
    )
    queries.append(
        JudgedQuery(
            "fact-margin-call",
            "Finance-style retrieval with domain-near but incomplete distractors.",
            make_proxy_query(16, [5, 6, 7]),
            single_relevant_doc_id("beir-margin-call"),
        )
    )

    return JudgedTask(
        "beir",
        "fact_proxy",
        "Proxy for heterogeneous factual retrieval inspired by BEIR task diversity.",
        "mrr",
        5,
        3,
        4,
        16,
        documents^,
        queries^,
    )


def make_msmarco_passage_proxy_task() raises -> JudgedTask:
    var documents = List[EncodedDocument]()
    documents.append(make_proxy_document("msmarco-boil-eggs", 16, [0, 1, 2]))
    documents.append(make_proxy_document("msmarco-freeze-herbs", 16, [3, 4, 5]))
    documents.append(make_proxy_document("msmarco-reset-router", 16, [6, 7, 8]))
    documents.append(make_proxy_document("msmarco-oven-clean", 16, [0, 9, 10]))
    documents.append(make_proxy_document("msmarco-router-wifi", 16, [6, 7, 11]))

    var queries = List[JudgedQuery]()
    queries.append(
        JudgedQuery(
            "passage-router-reset",
            "Short web question that should reward tight passage matches.",
            make_proxy_query(16, [6, 7, 8]),
            single_relevant_doc_id("msmarco-reset-router"),
        )
    )
    queries.append(
        JudgedQuery(
            "passage-boil-eggs",
            "Short web question with a cooking distractor sharing one concept.",
            make_proxy_query(16, [0, 1, 2]),
            single_relevant_doc_id("msmarco-boil-eggs"),
        )
    )

    return JudgedTask(
        "msmarco",
        "passage_proxy",
        "Proxy for short answer-passage retrieval with compact document vectors.",
        "mrr",
        3,
        3,
        3,
        16,
        documents^,
        queries^,
    )


def make_bright_reasoning_proxy_task() raises -> JudgedTask:
    var documents = List[EncodedDocument]()
    documents.append(make_proxy_document("bright-train-budget-mountain", 16, [0, 1, 2, 3, 4]))
    documents.append(make_proxy_document("bright-train-budget-city", 16, [0, 1, 2, 5, 6]))
    documents.append(make_proxy_document("bright-flight-budget-mountain", 16, [7, 1, 2, 3, 4]))
    documents.append(make_proxy_document("bright-train-luxury-mountain", 16, [0, 8, 9, 3, 4]))
    documents.append(make_proxy_document("bright-library-quiet-late", 16, [10, 11, 12, 13, 14]))
    documents.append(make_proxy_document("bright-library-late-cafe", 16, [10, 11, 15, 13, 6]))

    var queries = List[JudgedQuery]()
    queries.append(
        JudgedQuery(
            "reasoning-budget-train-mountain",
            "Multi-constraint query with several strong partial-match distractors.",
            make_proxy_query(16, [0, 1, 2, 3]),
            single_relevant_doc_id("bright-train-budget-mountain"),
        )
    )
    queries.append(
        JudgedQuery(
            "reasoning-quiet-late-library",
            "Constraint-heavy query that tests conjunction rather than single-term recall.",
            make_proxy_query(16, [10, 11, 12, 13]),
            single_relevant_doc_id("bright-library-quiet-late"),
        )
    )

    return JudgedTask(
        "bright",
        "reasoning_proxy",
        "Proxy for BRIGHT-like reasoning retrieval where conjunction beats shallow overlap.",
        "success",
        3,
        4,
        5,
        16,
        documents^,
        queries^,
    )


def make_miracl_multilingual_proxy_task() raises -> JudgedTask:
    var documents = List[EncodedDocument]()
    documents.append(make_proxy_document("miracl-en-passport", 16, [0, 1, 2, 12]))
    documents.append(make_proxy_document("miracl-fr-passeport", 16, [0, 1, 2, 13]))
    documents.append(make_proxy_document("miracl-ar-identity", 16, [0, 3, 4, 14]))
    documents.append(make_proxy_document("miracl-ja-train-pass", 16, [5, 6, 7, 15]))
    documents.append(make_proxy_document("miracl-es-passport-fee", 16, [0, 1, 8, 13]))

    var queries = List[JudgedQuery]()
    queries.append(
        JudgedQuery(
            "multilingual-passport-renewal",
            "Cross-lingual query with two relevant language variants.",
            make_proxy_query(16, [0, 1, 2]),
            two_relevant_doc_ids("miracl-en-passport", "miracl-fr-passeport"),
        )
    )
    queries.append(
        JudgedQuery(
            "multilingual-id-renewal",
            "Cross-lingual query where only the Arabic-aligned document is complete.",
            make_proxy_query(16, [0, 3, 4]),
            single_relevant_doc_id("miracl-ar-identity"),
        )
    )

    return JudgedTask(
        "miracl",
        "multilingual_proxy",
        "Proxy for multilingual retrieval with shared semantics across language variants.",
        "recall",
        3,
        3,
        4,
        16,
        documents^,
        queries^,
    )


def default_proxy_tasks() raises -> List[JudgedTask]:
    var tasks = List[JudgedTask]()
    tasks.append(make_lotte_forum_proxy_task())
    tasks.append(make_beir_fact_proxy_task())
    tasks.append(make_msmarco_passage_proxy_task())
    tasks.append(make_bright_reasoning_proxy_task())
    tasks.append(make_miracl_multilingual_proxy_task())
    return tasks^
