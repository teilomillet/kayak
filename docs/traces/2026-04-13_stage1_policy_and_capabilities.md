# Trace: Stage-1 Policy And Capabilities

Date: `2026-04-13`

## Why This Change Exists

The hosted path had two hard-coded assumptions that were becoming architectural
debt:

- newly sealed segments always built the same stage-1 sidecars
- runtime loading and filter support were derived from ad hoc generator checks

That shape was acceptable for a narrow prototype, but it was not a sound
foundation for a search engine whose stage-1 family may evolve.

## What Was Implemented

### 1. Collection-scoped stage-1 build policy

Collection manifests now persist a `search_artifact_build_policy`.

Current policy shape is intentionally narrow:

- each entry is `family + root`
- duplicate families are rejected
- duplicate roots are rejected
- roots that collide with reserved segment directories are rejected

Current sealing support is intentionally explicit:

- `document_proxy`
- `centroid_postings`

Unsupported configured families fail early at collection-manifest validation,
instead of allowing a collection to be created and only failing later during
snapshot sealing.

### 2. Capability-driven stage-1 contract

Planning now has an explicit `Stage1Capabilities` contract for candidate
generator kinds.

It records:

- generator family
- required search-artifact families
- whether stage 1 is exact
- whether match-all filters are supported
- whether structured filters are supported

The current runtime uses that contract for:

- filter guardrails
- snapshot-load requirements

This replaces the older direct dependence on a single `artifact_family` string
for planning and runtime decisions.

### 3. Policy-driven segment sealing

`seal_single_segment()` now builds stage-1 sidecars from the collection policy
instead of hard-coding the default pair inside the seal helper itself.

The default behavior is preserved by the default collection policy, not by an
implicit storage law.

## What This Does Not Claim

This tranche does not claim:

- that current supported families are the final stage-1 set
- that richer family-specific config should live in the collection core today
- that GEM-style or future native engines should share the same config shape

The point is the seam, not the final family.

## Verified By

Focused tests added or updated:

- `tests/test_collection_contracts.mojo`
- `tests/test_collection_storage.mojo`
- `tests/test_service_contracts.mojo`
- `tests/test_service_json.mojo`
- `tests/test_snapshot_bundle.mojo`
- `tests/test_stage1_capabilities.mojo`
- `tests/test_segment_builder_policy.mojo`

Dependent paths re-run:

- `tests/test_collection_resolution.mojo`
- `tests/test_collection_reclaim.mojo`
- `tests/test_collection_compaction.mojo`
- `tests/test_collection_search_plan.mojo`
- `tests/test_service_runtime.mojo`

## Result

The engine is now materially easier to change:

- collection continuity owns default stage-1 build intent
- segment manifests still record only actual built sidecars
- runtime decisions flow from explicit capabilities
- future family work can extend policy and capabilities without rewriting the
  exact-storage boundary
