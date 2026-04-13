# 2026-04-13: multi-artifact stage-2 operator

## Why this step

The repository already had:

- an explicit `Stage2Operator` contract
- explicit stage-2 artifact materialization reporting
- stage-2 reporting propagated through explain, service, benchmarks, and Python

But the built-in stage-2 operators were still all single-artifact operators:

- `noop_topk`
- `exact_late_interaction`
- `clause_text`

That left one important claim unverified:

- can the current stage-2 primitive actually represent a refinement stage that
  consumes more than one artifact family?

I did not want to answer that by inventing a synthetic contract-only operator.
The right verification target was a real operator grounded in paths the repo
already owns:

- exact late-interaction scoring over a bounded candidate window
- clause-text rescoring over that same candidate window

## Decision

Add one explicit hybrid operator:

- `exact_late_interaction_clause_text`

Properties:

- `family = "hybrid"`
- `required_artifact_families = ["late_interaction", "document_text"]`
- `requires_query_text = true`
- `is_exact_reference = false`

Why `is_exact_reference = false`:

- the operator executes an exact late-interaction sub-step
- but the final stage-2 ranking is the blended hybrid result, not the exact
  reference result

## Compatibility decision

I checked the legacy compatibility fields before implementing the operator:

- `exact_stage_kind`
- `reranker_kind`

The conservative choice was:

- `exact_stage_kind = "none"`
- `reranker_kind = "clause_text"`

Reason:

- `exact_stage_kind` names the whole legacy stage, not an internal sub-step
- setting it to `exact_late_interaction` would falsely suggest the final stage-2
  profile is itself the exact-reference profile
- `reranker_kind = "clause_text"` remains truthful because the final reranking
  step is clause-text rescoring

I also fixed one benchmark-reporting bug exposed by this operator:

- `CeilingComparisonSummary` had been emitting `reranker_kind = stage2_kind`
- that was only accidentally correct for single-kind text rerankers
- it now carries an explicit `reranker_kind` field

## What changed

Engine:

- [kayak/planning/stage2_operator.mojo](../../kayak/planning/stage2_operator.mojo)
- [kayak/planning/exact_late_interaction_clause_text_stage.mojo](../../kayak/planning/exact_late_interaction_clause_text_stage.mojo)
- [kayak/planning/execution_stage2.mojo](../../kayak/planning/execution_stage2.mojo)

Python SDK:

- [python/kayak_bridge/stage2_operator.py](../../python/kayak_bridge/stage2_operator.py)
- [python/kayak_bridge/clause_text.py](../../python/kayak_bridge/clause_text.py)
- [python/kayak_bridge/planned_search.py](../../python/kayak_bridge/planned_search.py)
- [python/kayak/README.md](../../python/kayak/README.md)

Benchmark/reporting:

- [kayak/benchmarks/ceiling_comparison_json.mojo](../../kayak/benchmarks/ceiling_comparison_json.mojo)

## Verified behavior

The new operator now:

- exact-scores the candidate window with late interaction
- materializes the candidate text window
- applies clause-text rescoring over the exact candidate scores
- reports both materialized artifact families in first-seen order

Current aggregate stage-2 totals for the hybrid operator are interpreted as:

- `document_count`: candidate-window document count, not a sum over artifact
  windows
- `token_count`: sum of materialized token-bearing artifacts
- `vector_count`: late-interaction vector count
- `byte_size`: sum of materialized artifact byte sizes

That keeps the stable stage count readable while the per-artifact breakdown
remains explicit in `materialized_artifacts`.

## Commands run

Mojo:

```bash
pixi run mojo -I . tests/test_collection_search_plan.mojo
pixi run mojo -I . tests/test_service_contracts.mojo
pixi run mojo -I . tests/test_service_runtime.mojo
pixi run mojo -I . tests/test_ceiling_comparison_json.mojo
```

Python:

```bash
PYTHONPATH=python pixi run python python/tests/test_search_plan_api.py
PYTHONPATH=python pixi run python python/tests/test_public_api_contract.py
PYTHONPATH=python pixi run python python/tests/test_python_sdk_docs.py
```

## What this proves

Verified:

- the stage-2 primitive is not limited to one artifact family per operator
- the service boundary accepts and validates a real multi-artifact stage-2
  operator
- the Python SDK can expose the same operator without hiding its artifact
  requirements
- benchmark JSON no longer assumes `reranker_kind == stage2_kind`

Not verified here:

- whether this hybrid operator is the right long-term ranking formula
- whether a richer exact sub-profile should eventually become first-class in
  explain output rather than staying implicit inside the hybrid operator

That second point remains open on purpose.
