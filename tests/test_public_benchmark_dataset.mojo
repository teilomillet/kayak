from std.testing import TestSuite, assert_equal

from kayak.benchmarks import (
    load_public_benchmark_dataset,
    public_benchmark_dataset_collection_root,
    public_benchmark_dataset_has_loaded_text_corpus,
    public_benchmark_dataset_has_text_sidecars,
    require_public_benchmark_dataset_loaded_text_corpus,
)


def test_browsecomp_gold_reports_text_capability_without_eager_load() raises:
    var dataset = load_public_benchmark_dataset("browsecomp_plus_gold")

    assert_equal(public_benchmark_dataset_has_text_sidecars(dataset), True)
    assert_equal(public_benchmark_dataset_has_loaded_text_corpus(dataset), False)
    assert_equal(
        String(public_benchmark_dataset_collection_root(dataset, "probe"))
            == ".cache/kayak/browsecomp_plus_gold_probe",
        True,
    )

    var raised = False
    try:
        _ = require_public_benchmark_dataset_loaded_text_corpus(dataset)
    except:
        raised = True
    assert_equal(raised, True)


def test_browsecomp_gold_loads_and_validates_text_corpus_on_request() raises:
    var dataset = load_public_benchmark_dataset(
        "browsecomp_plus_gold",
        load_text_corpus=True,
    )
    var corpus = require_public_benchmark_dataset_loaded_text_corpus(dataset)

    assert_equal(public_benchmark_dataset_has_text_sidecars(dataset), True)
    assert_equal(public_benchmark_dataset_has_loaded_text_corpus(dataset), True)
    assert_equal(len(corpus.doc_ids), dataset.stored_index.index.document_count)
    assert_equal(corpus.doc_ids[0], dataset.stored_index.index.doc_ids[0])


def test_scifact_reports_no_text_sidecars() raises:
    var dataset = load_public_benchmark_dataset(
        "scifact_real_subset",
        load_text_corpus=True,
    )

    assert_equal(public_benchmark_dataset_has_text_sidecars(dataset), False)
    assert_equal(public_benchmark_dataset_has_loaded_text_corpus(dataset), False)

    var raised = False
    try:
        _ = require_public_benchmark_dataset_loaded_text_corpus(dataset)
    except:
        raised = True
    assert_equal(raised, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
