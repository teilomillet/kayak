# 2026-04-13: Explicit Search Planner Layer

## Why This Step Exists

The repository already had:

- explicit `SearchPlan`
- explicit stage-1 families
- explicit explain payloads

But the hosted service still required callers to materialize a concrete
generator-specific plan up front.

That was too rigid for the next hosted-engine step, yet the repo's own product
position also ruled out a hidden auto-selector that silently changed search
behavior.

The sound next step was therefore:

- add an explicit planner contract above `SearchPlan`
- make the planner inspect what a snapshot actually contains
- keep the selected concrete plan visible in responses and explain output

## What Changed

### Snapshot artifact inventory

Added:

- [kayak/collections/snapshot_inventory.mojo](../../kayak/collections/snapshot_inventory.mojo)

This path loads:

- collection manifest
- snapshot manifest
- sealed segment manifests

and computes:

- search-artifact families available on any segment
- search-artifact families available on all segments

Reason:
- planner selection should key off actual sealed snapshot inventory
- it should not infer capability from collection build intent alone

### Explicit planner contracts

Added:

- [kayak/planning/planner.mojo](../../kayak/planning/planner.mojo)

New public planner-facing contracts:

- `SearchPlanSelectionRequest`
- `SearchPlanSelection`

New planner goals:

- `balanced`
- `exact_only`
- `latency_first`
- `native_multivector`

Current explicit guardrails:

- non-`match_all` filters force exact stage 1
- `exact_stage1_required` forces exact stage 1
- `oracle_full_recall_required` without `debug_mode` forces exact stage 1

### Hosted runtime path

Added:

- `PlannedSearchRequest`
- `PlannedSearchResponse`
- `PlannedDebugSearchResponse`
- `PlannedExplainResponse`
- `execute_planned_search(...)`
- `execute_planned_debug_search(...)`
- `execute_planned_explain(...)`

Important design choice:

- existing `SearchRequest` remains explicit and unchanged
- the planner path is a second typed contract, not a hidden mutation of the
  explicit one

## What The Default Order Does And Does Not Claim

The default goal order is intentionally conservative.

It **does** claim:

- `document_proxy` is still the best measured current default frontier point on
  the local public slices for a general balanced goal
- `centroid_postings_imputed_flat` is the preferred current WARP-shaped
  implementation direction for an explicitly native multi-vector goal
- exact fallback is always available and visible

It does **not** claim:

- that `gem_graph` is the current default winner
- that `centroid_postings_head_auto` should be silently promoted
- that `centroid_postings_blockmax` dominates plain centroid postings

Those exclusions are deliberate because the current local traces only justify:

- keeping GEM instrumentation and comparison
- keeping head and blockmax variants as explicit measurable options
- not promoting them as hidden defaults yet

## Validation

Targeted planner tests:

```bash
pixi run mojo -I . tests/test_search_planner.mojo
```

Typed service-contract validation:

```bash
pixi run mojo -I . tests/test_service_contracts.mojo
pixi run mojo -I . tests/test_service_json.mojo
```

Hosted runtime validation:

```bash
pixi run mojo -I . tests/test_service_runtime.mojo
```

Observed result:

- all listed tests passed locally after the planner layer landed

Notable runtime confirmations:

- balanced planned search selects `document_proxy` on the default sealed policy
- native-multivector planned search selects `centroid_postings_imputed_flat`
  when centroid postings are available
- filtered planned debug search falls back to `exact_full_scan`

## Epistemic Boundary

This step proves that `kayak` now has:

- an explicit planner contract
- real runtime execution through that planner
- visible selected-plan reporting

This step does **not** prove that the current default goal order is final.

It only proves that the repository now has a reversible, inspectable place to
host those choices while stage-1 engine work continues to evolve.
