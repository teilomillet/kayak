# 2026-04-13 Document Representation Transform Contract

## Claim

`kayak` now has a first-class document-representation-transform contract at the
sealed segment boundary.

This is a contract and provenance step, not a claim that token pooling,
pruning, or other transforms are fully implemented as serving defaults.

## Why This Change Was Justified

Verified before implementation:

- the repo already separated:
  - model identity
  - search-native artifacts
  - stage-1 engine families
  - stage-2 reference semantics
- what remained implicit was how the exact packed representation for a sealed
  segment had been transformed before storage

Evidence checked before editing:

- [TODO.md](../../TODO.md)
- [docs/architecture/extensibility_wall.md](../architecture/extensibility_wall.md)
- [kayak/collections/segment.mojo](../../kayak/collections/segment.mojo)
- [kayak/collections/segment_store.mojo](../../kayak/collections/segment_store.mojo)

## What Changed

Added a new collection-level contract module:

- [kayak/collections/document_representation_transform.mojo](../../kayak/collections/document_representation_transform.mojo)

This module now defines:

- `DocumentRepresentationTransformManifest`
- `DocumentRepresentationTransformConfigEntry`
- ordered transform-chain helpers
- two explicit example transform kinds:
  - `token_pooling`
  - `prefix_pruning`

Threaded that contract through sealed segments:

- [kayak/collections/segment.mojo](../../kayak/collections/segment.mojo)
- [kayak/collections/segment_store.mojo](../../kayak/collections/segment_store.mojo)

The segment manifest now persists:

- `document_representation_transform_count`
- ordered transform kinds
- per-transform config entries

Backward compatibility decision:

- old segment manifests without any transform entries still load
- they resolve to an empty transform chain rather than failing

Reason:

- this keeps prior sealed manifests readable while making the seam explicit for
  future pooled or pruned representations

## What Was Verified

Focused test results:

- `pixi run mojo -I . tests/test_document_representation_transform.mojo`
- `pixi run mojo -I . tests/test_collection_storage.mojo`
- `pixi run mojo -I . tests/test_collection_contracts.mojo`
- `pixi run mojo -I . tests/test_snapshot_inventory.mojo`
- `pixi run mojo -I . tests/test_collection_search_plan.mojo`
- `pixi run mojo -I . tests/test_collection_compaction.mojo`

Observed outcome:

- all targeted tests passed locally after the change
- segment manifest roundtrips preserve transform chains
- old manifests without transform entries still load
- planner, snapshot inventory, and compaction paths were unaffected by the new
  metadata

## What This Does Not Yet Prove

This change does **not** prove:

- that token pooling is already implemented in the hosted engine
- that pruning or pooling improves recall, latency, or bytes/vector on any
  workload
- that transformed exact segments should replace the current untransformed
  exact reference path

Those claims still need:

- concrete transform execution code
- benchmark evidence against an untransformed reference
- storage and latency measurements on representative slices

## Resulting Next Step

The next sound step on this seam is:

1. implement one concrete transform build path on top of this contract
2. measure bytes/vector, vectors/document, latency, and quality
3. compare the transformed segment against an untransformed exact reference
