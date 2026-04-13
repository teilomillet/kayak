from std.testing import TestSuite, assert_equal

from kayak import StageArtifactMaterialization, Stage2Result


def test_stage_artifact_materialization_rejects_empty_family() raises:
    var raised = False

    try:
        _ = StageArtifactMaterialization("", 1, 1, 1, 1, 16)
    except:
        raised = True

    assert_equal(raised, True)


def test_stage2_result_can_carry_explicit_materialized_artifacts() raises:
    var result = Stage2Result(
        [],
        [StageArtifactMaterialization("document_text", 1, 2, 7, 0, 48)],
        1,
        2,
        7,
        0,
        48,
    )

    assert_equal(len(result.materialized_artifacts), 1)
    assert_equal(result.materialized_artifacts[0].family, "document_text")
    assert_equal(result.materialized_artifacts[0].byte_size, 48)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
