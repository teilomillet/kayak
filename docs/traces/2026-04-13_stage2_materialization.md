# 2026-04-13: explicit stage-2 artifact materialization

## Why this step

The repository already had an explicit stage-2 operator contract:

- operator kind
- operator family
- required artifact families

But the execution path still only reported aggregate stage-2 counts:

- document count
- token count
- vector count
- byte size

That was enough to estimate cost, but not enough to answer the more precise
question:

- what candidate-window artifacts did stage 2 actually materialize?

That distinction matters because stage-2 requirements and stage-2
materialization are different facts.

Examples:

- `exact_late_interaction` requires late-interaction vectors and materializes a
  packed candidate index
- `clause_text` requires document text and materializes a bounded text window
- `noop_topk` requires nothing and should not pretend that it materialized
  anything

## What changed

New primitive:

- [kayak/planning/stage_artifact_materialization.mojo](../../kayak/planning/stage_artifact_materialization.mojo)

The primitive records, for one stage-local artifact window:

- `family`
- `segment_count`
- `document_count`
- `token_count`
- `vector_count`
- `byte_size`

It is now threaded through:

- [kayak/planning/stage2_result.mojo](../../kayak/planning/stage2_result.mojo)
- [kayak/planning/stage_profile.mojo](../../kayak/planning/stage_profile.mojo)
- [kayak/planning/exact_stage.mojo](../../kayak/planning/exact_stage.mojo)
- [kayak/planning/clause_text_stage.mojo](../../kayak/planning/clause_text_stage.mojo)
- [kayak/planning/explain.mojo](../../kayak/planning/explain.mojo)
- [kayak/planning/json.mojo](../../kayak/planning/json.mojo)

## Verified behavior

The current engine now reports stage-2 materialization explicitly:

- `noop_topk`
  - `materialized_artifacts = []`
- `exact_late_interaction`
  - one `late_interaction` materialization over the candidate window
- `clause_text`
  - one `document_text` materialization over the candidate window

The local Python SDK now mirrors the same stage-2 concept through:

- [python/kayak_bridge/stage_artifact_materialization.py](../../python/kayak_bridge/stage_artifact_materialization.py)
- [python/kayak_bridge/search_stage_profile.py](../../python/kayak_bridge/search_stage_profile.py)
- [python/kayak_bridge/planned_search.py](../../python/kayak_bridge/planned_search.py)

This is verified by direct tests:

- `pixi run mojo -I . tests/test_stage_artifact_materialization.mojo`
- `pixi run mojo -I . tests/test_service_json.mojo`
- `pixi run mojo -I . tests/test_collection_search_plan.mojo`
- `pixi run mojo -I . tests/test_service_contracts.mojo`
- `pixi run mojo -I . tests/test_service_runtime.mojo`
- `PYTHONPATH=python pixi run python python/tests/test_search_plan_api.py`
- `PYTHONPATH=python pixi run python python/tests/test_public_api_contract.py`
- `PYTHONPATH=python pixi run python python/tests/test_python_sdk_docs.py`

## Why this shape was chosen

I did not encode artifact family only in `Stage2Result` totals, because that
would still force callers to infer whether `token_count > 0` means text,
vectors, or both.

I also did not hide this under a stage-2-specific JSON-only field, because the
engine should own the contract first and serialization should mirror it.

The chosen shape keeps the facts separate:

- the plan says what stage 2 requires
- the result/profile says what stage 2 actually materialized

## What remains open

This step does **not** yet prove everything about the stage-2 primitive.

Still open:

- benchmark summaries do not yet emit stage-2 materialization families
- future multi-artifact stage-2 operators have not been implemented yet

So the contract is now explicit in the engine and the local Python SDK, but not
yet fully propagated through every reporting surface.
