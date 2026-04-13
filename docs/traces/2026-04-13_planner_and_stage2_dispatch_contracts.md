## Goal

Remove two remaining centralized kind-switches from the planning/execution
surface:

- planner default plan construction by candidate-generator kind
- stage-2 execution dispatch by raw operator kind

without changing current behavior.

## Why this tranche

After the previous structural work:

- stage-1 already carried richer contract metadata
- exactness was already capability-driven
- centroid-family execution had its own explicit execution contract

Two seams were still more ad hoc than they needed to be:

- `planner.mojo` had a long `selected_plan_for_kind` chain that recreated the
  default search-plan semantics one kind at a time
- `execution_stage2.mojo` dispatched directly on `plan.stage2_operator.kind`
  even though `Stage2Operator` was already the canonical contract object

These were not broken, but they were the next obvious places where future
primitives or aliases would require repetitive edits.

## Changes

### Planner plan factory

Added `kayak/planning/planner_plan_factory.mojo` with three helpers:

- `planner_candidate_generator_for_kind(...)`
- `planner_default_search_plan_for_candidate_generator(...)`
- `planner_default_search_plan_for_kind(...)`

This moves planner default semantics into one place:

- exact candidate generators default to:
  - `exact_stage1_required`
  - `noop_topk`
- non-exact candidate generators default to:
  - requested faithfulness policy
  - `exact_late_interaction`
- `gem_graph` keeps its beam/cluster request parameters

`planner.mojo` now delegates default plan construction to that factory instead
of repeating per-kind constructor logic inline.

### Richer stage-2 operator contract

Extended `Stage2Operator` to carry:

- `execution_kind`
- `compatibility_exact_stage_kind`
- `compatibility_reranker_kind`

This makes the operator itself the source of truth for both:

- runtime stage-2 dispatch
- `SearchPlan` compatibility metadata

`execution_stage2.mojo` now dispatches on `execution_kind`, and
`search_plan.mojo` now reads compatibility values from the operator contract
instead of recomputing them from the raw operator kind.

### Package surface

Exported the planner plan-factory helpers through:

- `kayak/planning/__init__.mojo`
- `kayak/__init__.mojo`

so the new contract surface is first-class and inspectable.

## Verification

Executed:

- `pixi run mojo -I . tests/test_search_planner.mojo`
- `pixi run mojo -I . tests/test_stage2_operator_contract.mojo`
- `pixi run mojo -I . tests/test_service_contracts.mojo`
- `pixi run mojo -I . tests/test_collection_search_plan.mojo`
- `pixi run mojo -I . tests/test_service_runtime.mojo`

Observed:

- planner default semantics remained unchanged
- stage-2 identity, text, late-interaction, and hybrid operators kept their
  expected contract fields
- collection-search-plan and hosted runtime suites passed after the refactor,
  including hybrid stage-2 and planned-search flows

## Outcome

Kayak now treats planner defaults and stage-2 execution as explicit contracts
instead of long raw-kind switch blocks. That does not make the engine more
accurate by itself, but it materially lowers the cost of changing candidate
generators and stage-2 operators later without threading new branches through
the core planner and runtime files.
