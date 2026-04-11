from std.testing import TestSuite, assert_equal

from kayak.eval import evaluate_task
from kayak.interop import load_mock_python_task
from kayak.runtime import ExactCpuBackend


def test_python_task_decoder_builds_judged_task() raises:
    var task = load_mock_python_task()
    var evaluation = evaluate_task(ExactCpuBackend(), task)

    assert_equal(task.family, "mock")
    assert_equal(task.slice_name, "python_bridge")
    assert_equal(task.vector_dim, 2)
    assert_equal(len(task.documents), 2)
    assert_equal(len(task.queries), 1)
    assert_equal(evaluation.mean_reciprocal_rank, 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
