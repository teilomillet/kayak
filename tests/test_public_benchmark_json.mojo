from std.testing import TestSuite, assert_equal

from kayak.benchmarks import (
    BackendPolicyBenchmarkSummary,
    ProxyTaskEvaluationSummary,
    SearchBreakdownBenchmarkSummary,
    WorkloadBenchmarkSummary,
    backend_policy_benchmark_summaries_json,
    proxy_task_evaluation_summaries_json,
    search_breakdown_benchmark_summaries_json,
    workload_benchmark_summaries_json,
)


def test_proxy_task_evaluation_json_contains_primary_metric() raises:
    var json = proxy_task_evaluation_summaries_json(
        [
            ProxyTaskEvaluationSummary(
                "beir",
                "fact_proxy",
                "mrr",
                0.75,
                10,
                8,
                64,
                0.70,
                0.75,
                0.80,
                0.90,
            )
        ]
    )

    assert_equal(json.find("\"primary_metric\":\"mrr\"") != -1, True)
    assert_equal(json.find("\"document_count\":64") != -1, True)


def test_workload_benchmark_json_contains_shape_fields() raises:
    var json = workload_benchmark_summaries_json(
        [
            WorkloadBenchmarkSummary(
                "lotte",
                "forum_fast",
                "proxy workload",
                192,
                20,
                12,
                32,
                10,
                0.012,
            )
        ]
    )

    assert_equal(json.find("\"family\":\"lotte\"") != -1, True)
    assert_equal(json.find("\"mean_search_seconds\":0.012") != -1, True)


def test_backend_policy_benchmark_json_contains_backend_name() raises:
    var json = backend_policy_benchmark_summaries_json(
        [
            BackendPolicyBenchmarkSummary(
                "SciFact",
                "real_subset",
                "storage",
                "storage",
                "serial",
                10,
                128,
                12,
                32,
                128,
                10,
                0.034,
            )
        ]
    )

    assert_equal(json.find("\"backend_name\":\"serial\"") != -1, True)
    assert_equal(json.find("\"top_k\":10") != -1, True)


def test_search_breakdown_benchmark_json_contains_component_name() raises:
    var json = search_breakdown_benchmark_summaries_json(
        [
            SearchBreakdownBenchmarkSummary(
                "FIQA",
                "real_subset",
                "storage",
                "storage",
                "score_all",
                10,
                128,
                12,
                32,
                128,
                10,
                0.056,
            )
        ]
    )

    assert_equal(json.find("\"component_name\":\"score_all\"") != -1, True)
    assert_equal(json.find("\"mean_seconds\":0.056") != -1, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
