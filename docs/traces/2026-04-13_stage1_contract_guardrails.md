## Goal

Generalize the primitive-facing stage-1 contract so planner, runtime, and explain
paths depend on declared capabilities rather than generator-name branches.

## Why this tranche

- `Stage2Operator` already modeled required artifact families as a list, but
  `CandidateGenerator` still collapsed stage-1 requirements to a single string.
- Exactness in faithfulness assessment was hardcoded to the generator name
  `"exact_full_scan"` rather than derived from declared capabilities.
- `CollectionSearchExplain.exact_stage` incorrectly mirrored the observed stage-2
  profile instead of describing the exact-oracle ceiling.

Those behaviors made the system narrower than needed for future primitives and
hid a real instrumentation bug in the explain path.

## Changes

### Stage-1 contract propagation

- Added list-based artifact-family helpers to
  `kayak/planning/stage1_capabilities.mojo`.
- Added build-policy helpers to
  `kayak/collections/search_artifact_policy.mojo` so compatibility checks work
  from required families instead of generator-specific conditionals.
- Extended `CandidateGenerator` to carry:
  - `required_search_artifact_families`
  - `is_exact`
  - `supports_match_all_filter`
  - `supports_structured_filter`
- Updated runtime snapshot load requirements to reuse the plan-carried contract
  instead of recomputing it from the generator kind.

### Correctness guardrails

- Switched faithfulness exactness checks to capability-derived metadata rather
  than a hardcoded generator name.
- Replaced the explain-path `exact_stage` placeholder with a real exact-oracle
  stage profile built from the collection snapshot and exact final hits.
- Made centroid-family execution derive its artifact requirement from the plan’s
  declared artifact family instead of a fixed list of generator names.

### Introspection

- Added `candidate_generator_family` and
  `stage1_required_artifact_families` to explain JSON so emitted artifacts
  expose the generalized contract directly.

## Verification

Executed:

- `pixi run mojo -I . tests/test_stage1_capabilities.mojo`
- `pixi run mojo -I . tests/test_collection_search_plan.mojo`
- `pixi run mojo -I . tests/test_planner_registry.mojo`
- `pixi run mojo -I . tests/test_planner_benchmark_json.mojo`
- `pixi run mojo -I . tests/test_planner_evidence_json.mojo`
- `pixi run mojo -I . tests/test_collection_contracts.mojo`
- `pixi run mojo -I . tests/test_segment_builder_policy.mojo`
- `pixi run bench_real_subset_planner_benchmark_smoke_raw`

Observed:

- All targeted Mojo test suites passed.
- The planner smoke benchmark completed and wrote
  `.cache/kayak/public_planner_benchmark_smoke.json`.
- The smoke artifact remains a planner-summary list rather than the richer
  explain JSON shape, so explain-field verification is covered by the dedicated
  JSON tests above, not by the smoke artifact itself.

## Outcome

Kayak now treats stage-1 metadata as a first-class plan contract instead of a
thin generator label. That keeps the current centroid and graph families working
while making later primitive swaps materially cheaper and less error-prone.
