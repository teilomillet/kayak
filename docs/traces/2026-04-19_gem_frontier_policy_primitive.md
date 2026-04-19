# GEM Frontier Policy Primitive

Date: `2026-04-19`
Status: `implemented and verified against current code, tests, and probe output`

## Objective

Decide what to finish before wrapping the current GEM cycle and moving to the
next paper family.

The narrow question was not whether one more frontier heuristic could be added.
The narrower and more useful question was:

- should graph-frontier scheduling stay benchmark-local glue
- or should it become an explicit Kayak primitive that researchers can vary,
  inspect, and benchmark through the same planning and serving contract

## Decision

Promote graph-frontier scheduling to a first-class graph-family primitive.

Reason:

- the held-out GEM probes already showed that frontier policy changes recall and
  cost materially on the same graph
- keeping that choice only inside benchmark code would leave the measured
  result outside the main research surface
- centralizing the traversal runtime lets serving and benchmark paths exercise
  the same implementation, which is the sound way to freeze this GEM question

## Verified Implementation

Explicit policy contract:

- [`kayak/planning/graph_frontier_policy.mojo`](../../kayak/planning/graph_frontier_policy.mojo)
  now owns the supported policy kinds and validation

Shared traversal runtime:

- [`kayak/planning/graph_frontier_runtime.mojo`](../../kayak/planning/graph_frontier_runtime.mojo)
  now owns the graph-family traversal implementations used by GEM frontier
  experiments

Planning and execution surface:

- [`kayak/planning/candidate_generator.mojo`](../../kayak/planning/candidate_generator.mojo)
  carries `graph_frontier_policy_kind`
- [`kayak/planning/search_plan.mojo`](../../kayak/planning/search_plan.mojo)
  exposes the configured policy on GEM plans
- [`kayak/planning/planner.mojo`](../../kayak/planning/planner.mojo) and
  [`kayak/planning/planner_plan_factory.mojo`](../../kayak/planning/planner_plan_factory.mojo)
  thread the policy through planner requests and default plan construction
- [`kayak/planning/execution_graph_family.mojo`](../../kayak/planning/execution_graph_family.mojo)
  now calls the shared runtime instead of carrying a duplicate local traversal

JSON and service contract:

- [`kayak/planning/json.mojo`](../../kayak/planning/json.mojo)
- [`kayak/benchmarks/search_plan_semantics_json.mojo`](../../kayak/benchmarks/search_plan_semantics_json.mojo)
- [`kayak/service/json.mojo`](../../kayak/service/json.mojo)
- [`kayak/service/search_contracts.mojo`](../../kayak/service/search_contracts.mojo)

Benchmark alignment:

- [`kayak/benchmarks/gem_heldout_frontier_policy_probe.mojo`](../../kayak/benchmarks/gem_heldout_frontier_policy_probe.mojo)
  now imports the same shared graph-frontier runtime instead of duplicating the
  traversal logic locally

## Validation

Focused contract and execution tests:

```bash
pixi run mojo -I . tests/test_stage1_semantic_guardrails.mojo
pixi run mojo -I . tests/test_search_planner.mojo
pixi run mojo -I . tests/test_service_json.mojo
pixi run mojo -I . tests/test_service_contracts.mojo
pixi run mojo -I . tests/test_collection_search_plan.mojo
pixi run mojo -I . tests/test_gem_heldout_frontier_policy_probe.mojo
```

Observed result:

- all listed tests passed locally after the primitive promotion

Probe check:

```bash
pixi run mojo -I . benchmarks/synthetic_hard_recall_gem_heldout_frontier_policy_probe.mojo
```

Observed result:

- the refreshed run on the checked synthetic held-out slice showed:
- `local_per_entry` and `hybrid_best_head_fair_round` stayed strongest on the
  baseline profile at `candidate_k=64` with recall `1.0`
- `global_best_first` stayed worse on that same point with recall `0.5`
- the looser quota-`2` round collapsed back toward local behavior rather than
  opening a better quality-cost middle point

## Conclusion

The verified conclusion is narrow:

- the useful ingredient in the strongest current GEM baseline is root-fair
  scheduling, not just "more global ordering"
- simple frontier-ordering scalars now look mostly exhausted on the checked
  slice
- the right GEM wrap point is therefore to keep `local_per_entry` as the
  default policy, keep frontier policy explicit in the research contract, and
  move the next GEM step to either:
  - a stronger structural graph primitive
  - a more stateful diversity primitive

## What This Does Not Claim

- it does **not** claim GEM is now the best default stage-1 engine
- it does **not** claim all search-side work is exhausted
- it does **not** claim the held-out synthetic slice is the full GEM verdict

It only claims the narrower verified point:

- frontier scheduling is now part of Kayak's explicit primitive surface, and
  the current queue-ordering exploration has reached a sound stop point
