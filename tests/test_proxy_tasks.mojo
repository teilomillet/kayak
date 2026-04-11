from std.testing import TestSuite, assert_equal

from kayak.benchmarks import default_proxy_tasks
from kayak.eval import evaluate_task
from kayak.runtime import ExactCpuBackend


def test_proxy_tasks_retrieve_expected_documents() raises:
    var backend = ExactCpuBackend()

    for task in default_proxy_tasks():
        var evaluation = evaluate_task(backend, task)

        assert_equal(evaluation.success_rate_at_k, 1.0)
        assert_equal(evaluation.primary_value, 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
