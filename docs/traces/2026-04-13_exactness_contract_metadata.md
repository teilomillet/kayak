## Goal

Push exactness handling one layer deeper into the primitive contract so planner
and hosted-service logic no longer rely on the literal generator name
`exact_full_scan` when the real question is whether stage 1 is exact.

## Why this tranche

After the earlier stage-1 contract refactor, `CandidateGenerator` already
carried `is_exact` and required artifact families. Three important seams still
ignored that metadata:

- `SearchPlan` enforced `exact_stage1_required` by checking the generator name.
- `SearchRequest` enforced the oracle-debug guardrail by checking the generator
  name.
- planner availability special-cased `exact_full_scan` instead of simply using
  the declared required families.

That left the system logically narrower than its own contract surface.

## Changes

- `kayak/planning/search_plan.mojo`
  - `exact_stage1_required` now accepts any candidate generator whose
    `is_exact` flag is true.
- `kayak/service/search_contracts.mojo`
  - the `oracle_full_recall_required` debug guardrail now keys off
    `plan.candidate_generator.is_exact`.
- `kayak/planning/planner.mojo`
  - candidate-generator availability now derives uniformly from required search
    artifact families, with no name-based special case for exact search.

## Regression tests

Added tests that would fail if the code regressed back to name-based logic:

- `tests/test_collection_search_plan.mojo`
  - constructs a synthetic exact candidate generator with a nonstandard kind
    string and verifies `SearchPlan` still accepts
    `exact_stage1_required`.
  - verifies a synthetic non-exact generator is rejected under that same
    policy.
- `tests/test_service_contracts.mojo`
  - constructs a synthetic exact candidate generator with a nonstandard kind
    string and verifies `SearchRequest` allows
    `oracle_full_recall_required` without debug mode.

Also fixed the local test fixture for `CollectionSearchExplain` so it matches
the current constructor shape with explicit `exact_stage` and `final_hits`.

## Verification

Executed:

- `pixi run mojo -I . tests/test_search_planner.mojo`
- `pixi run mojo -I . tests/test_service_contracts.mojo`
- `pixi run mojo -I . tests/test_collection_search_plan.mojo`
- `pixi run mojo -I . tests/test_service_runtime.mojo`

Observed:

- all four suites passed
- the runtime suite exercised planned search, exact-filter guardrails, text
  stage-2 execution, gem-graph configuration, reclaim flows, and lifecycle
  persistence on top of the refactor

## Outcome

Kayak is now more internally consistent: "exact" is a declared semantic
property of the stage-1 operator, not an accidental synonym for one current
implementation label. That keeps future primitive swaps cheaper and reduces the
risk of hidden `exact_full_scan` assumptions in the hosted engine boundary.
