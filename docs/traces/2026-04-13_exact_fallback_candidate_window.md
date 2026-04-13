# 2026-04-13: Exact Fallback Preserves Candidate Window

## Why This Step Exists

The repository already had:

- explicit stage-2 operators
- planned search requests with visible stage-1 selection
- planner benchmark and planner-evidence artifacts

But one planner path still violated the intended primitive contract:

- when the planner fell back to `exact_full_scan` because no non-exact stage-1
  family was available, it narrowed `candidate_k` down to `final_k`

That was unsound for stage-2-first planning because:

- exact fallback is still a valid stage-1 candidate generator
- text-family or other non-default stage-2 operators may need a wider exact
  candidate window than `final_k`
- silently shrinking the exact window makes later stage-2 overrides look worse
  for reasons unrelated to the stage-2 primitive itself

## What Changed

Updated:

- [kayak/planning/planner.mojo](../../kayak/planning/planner.mojo)

The planner fallback path now preserves the caller-requested:

- `final_k`
- `candidate_k`

instead of forcing:

- `candidate_k = final_k`

This keeps fallback behavior aligned with the explicit `CandidateBudget`
contract already used elsewhere in the planner.

## Documentation Update

Updated:

- [docs/architecture/service_api.md](../architecture/service_api.md)

The service architecture document now states explicitly that:

- `PlannedSearchRequest` may carry `query_text`
- `PlannedSearchRequest` may carry `stage2_operator_kind`
- exact planner fallback preserves the requested candidate window so stage-2
  overrides still rerank the intended exact set

## Validation

Focused planner validation:

```bash
pixi run mojo -I . tests/test_search_planner.mojo
```

Focused planned-stage2 regression:

```bash
pixi run mojo -I . tests/test_planned_stage2_override.mojo
```

Hosted runtime regression:

```bash
pixi run mojo -I . tests/test_service_runtime.mojo
```

Planner benchmark and evidence artifacts:

```bash
pixi run bench_browsecomp_plus_gold_planner_benchmark_raw
pixi run bench_browsecomp_plus_gold_planner_evidence_raw
```

Observed result:

- all listed tests passed locally
- gold planner benchmark artifact was written to:
  - `.cache/kayak/browsecomp_plus_gold_planner_benchmark.json`
- gold planner evidence artifact was written to:
  - `.cache/kayak/browsecomp_plus_gold_planner_evidence.json`

Artifact shape check:

- benchmark artifact is a list of `20` planner summaries
- evidence artifact is a list of `20` planner-evidence summaries
- evidence summaries include:
  - `stage2_kind`
  - `selected_is_undominated`
  - per-candidate measured summaries

## Epistemic Boundary

This step proves:

- exact planner fallback no longer weakens the requested candidate window
- planned stage-2 override behavior is covered by a focused regression
- the gold planner benchmark and planner-evidence entrypoints run and emit
  machine-readable artifacts

This step does **not** prove:

- that the current planner ordering is globally optimal
- that current stage-2 choices are the final frontier
- that exact fallback should be the long-term dominant primitive

It only proves that fallback now preserves the explicit search-budget contract,
which is the sounder baseline for later stage-1 and stage-2 comparisons.
