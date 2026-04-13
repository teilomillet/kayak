# Search Plan Semantics

This note records the semantic contract that the planner and explain surfaces
now expose explicitly.

## Problem

The codebase already had multiple native late-interaction stage-1 engines:

- `exact_full_scan`
- `centroid_postings_*`
- `gem_graph`

What was missing was not staged structure. What was missing was explicit
semantic naming. It was still too easy to read the code in implementation
buckets like `proxy`, `centroid`, or `graph`, rather than in retrieval terms
like:

- what the reference scoring semantics are
- whether stage 1 is proxy or interaction-native
- whether stage 2 actually realizes the reference semantics
- whether a verifier runs afterward

## Contract

`SearchPlan` now separates four concepts:

1. `candidate_generator`
2. `reference_scoring_semantics`
3. `stage2_reference_operator`
4. `stage3_verifier`

The legacy combined `stage2_operator` is still present as a compatibility view,
but it is no longer the primary semantic description.

## Stage meanings

- Stage 1:
  candidate generation, which may be proxy, approximate late interaction, or
  exact late interaction.
- Stage 2:
  reference scoring over the candidate window, or a no-op when stage 1 already
  realized the reference semantics.
- Stage 3:
  optional verifier or reranker over the reference-scored window.
- Exact oracle:
  a separate explain-time profile over the full snapshot, used for faithfulness
  and recall reasoning.

## Stage-1 provenance

`Stage1Capabilities`, `CandidateGenerator`, and `CandidateSet` now expose:

- `interaction_semantics`
- `alignment_granularity`
- `score_kind`

This makes it explicit that:

- `document_proxy` is a proxy stage
- `centroid_postings_*` are approximate late-interaction stages
- `gem_graph` is an experimental approximate late-interaction stage
- `exact_full_scan` is exact late interaction

## Planner defaults

The default planner order was adjusted to match the native late-interaction
thesis more honestly:

- `centroid_postings_imputed_flat` is preferred for `balanced`
- `document_proxy` remains the latency-first baseline
- `gem_graph` remains experimental

## Benchmark And Report Semantics

The benchmark layer now mirrors the same split explicitly instead of collapsing
everything back into one legacy `stage2_kind`.

Current benchmark and evidence JSON emit:

- stage-1 generator semantics from the real `CandidateGenerator`
- `reference_scoring_semantics_*`
- `stage2_reference_*`
- `stage3_verifier_*`
- exact-oracle storage and vector counts as a separate section where relevant

Compatibility fields are still emitted, but only under explicit
`compatibility_*` names. The benchmark surfaces no longer present those legacy
combined names as the primary contract.

`exact_stage_kind` and `reranker_kind` should now be understood as compatibility
projection fields, not as canonical stored planner state.

Reason:
- the actual semantic state lives in:
  - `reference_scoring_semantics`
  - `stage2_reference_operator`
  - `stage3_verifier`
- compatibility naming is now derived from those components when projected

This matters for comparing stage-1 engines like centroid-family WARP-style
paths and GEM:

- the comparison can now be phrased directly in retrieval semantics
- the reporting no longer implies that every difference is "just stage 2"
- exact-oracle counts are no longer aliased to the stage-2 reference stage

## Verification status

This contract was verified locally against:

- `tests/test_stage1_capabilities.mojo`
- `tests/test_faithfulness_frontier_json.mojo`
- `tests/test_stage_aware_benchmark_json.mojo`
- `tests/test_single_core_scale_fixture.mojo`
- `tests/test_synthetic_hard_recall_fixture.mojo`
- `tests/test_collection_search_plan.mojo`
- `tests/test_service_contracts.mojo`
- `tests/test_service_json.mojo`
- `tests/test_service_runtime.mojo`
- `tests/test_planner_registry.mojo`
- `tests/test_planner_benchmark_json.mojo`
- `tests/test_planner_evidence_json.mojo`
- `tests/test_ceiling_comparison_json.mojo`
- `tests/test_stage1_semantic_guardrails.mojo`
- `python/tests/test_search_plan_api.py`
- `python/tests/test_public_api_contract.py`
- `python/tests/test_python_sdk_docs.py`
- `python/tests/test_stage_semantic_guardrails.py`

The repo also now has a compatibility-boundary guardrail:

- legacy combined stage naming is only allowed in explicitly allowlisted
  compatibility modules
- new production files that introduce `Stage2Operator`,
  `stage2_operator_kind`,
  `exact_stage_kind`, or `reranker_kind` outside that allowlist should fail the
  guardrail test

The compatibility view is also now derived from one planner-owned surface:

- `SearchPlanCompatibilitySemantics`
- `search_plan_compatibility_semantics(...)`

Reason:
- service JSON, explain JSON, and equality checks should not each rediscover
  compatibility stage naming independently
- one helper reduces the chance that compatibility names drift from the real
  explicit stage components
- `SearchPlan` itself no longer stores cached `exact_stage_kind` or
  `reranker_kind` fields

The repo also now has a stage-1 contract guardrail:

- registered candidate generators are checked against an explicit semantic
  matrix for interaction semantics, alignment granularity, score kind, and
  filter support
- the shared search-plan semantics JSON helper must continue to emit the
  explicit stage-1 fields even when compatibility stage-2 fields are disabled

The intended fast semantic merge gate is now:

- `pixi run test_semantic_guardrails`

That guardrail is now token-specific rather than one broad file allowlist.

Reason:
- `stage2_operator_kind` in request-override parsing is a different ownership
  surface from `exact_stage_kind` in JSON projection
- helper names like `...for_stage2_operator_kind` should not be mistaken for a
  compatibility leak

## Next steps

- Move remaining downstream consumers from `compatibility_*` fields to the
  explicit semantic keys.
- Extend the same contract to future native stage-1 engines beyond the current
  centroid and GEM families.
- Keep `stage2_operator` compatibility until downstream consumers move to the
  explicit fields.
