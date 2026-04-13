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

## Verification status

This contract was verified locally against:

- `tests/test_stage1_capabilities.mojo`
- `tests/test_collection_search_plan.mojo`
- `tests/test_service_contracts.mojo`
- `tests/test_service_json.mojo`
- `tests/test_service_runtime.mojo`
- `tests/test_planner_registry.mojo`
- `tests/test_planner_evidence_json.mojo`
- `tests/test_ceiling_comparison_json.mojo`
- `python/tests/test_search_plan_api.py`
- `python/tests/test_public_api_contract.py`
- `python/tests/test_python_sdk_docs.py`

## Next steps

- Add semantic fields to planner benchmark summaries where only legacy
  `stage2_*` names are still emitted.
- Extend the same contract to future native stage-1 engines beyond the current
  centroid and GEM families.
- Keep `stage2_operator` compatibility until downstream consumers move to the
  explicit fields.
