# 2026-04-13: atomic publication and draft baseline compaction

## Why this step

The hosted-engine substrate already had:

- collection manifests
- snapshot manifests
- draft mutation logging
- snapshot export/import
- live-snapshot compaction

But two engine-level lifecycle gaps were still real:

- publication was not modeled around a stable active snapshot pointer
- draft state stayed as an append-only mutation log even after sealing

Those gaps matter even before any heavier stage-1 work. If publication is not
explicit and draft state is not compacted, later performance work sits on a
weaker substrate than the engine contract suggests.

## What changed

### Atomic manifest and sidecar writes

New file:

- [kayak/storage/atomic_write.mojo](../../kayak/storage/atomic_write.mojo)

What changed:

- manifest writes now go through sibling temp-file writes plus atomic rename
- snapshot segment-id lists also use atomic text writes
- text and document-metadata sidecar payload files use atomic text writes
- draft delete-batch doc-id lists use atomic text writes

This does not create a transactional multi-file commit, but it does remove the
previous partial-write risk for individual text manifests and sidecar lists.

### Active snapshot becomes explicit

Updated files:

- [kayak/collections/collection.mojo](../../kayak/collections/collection.mojo)
- [kayak/collections/collection_store.mojo](../../kayak/collections/collection_store.mojo)
- [kayak/collections/publish.mojo](../../kayak/collections/publish.mojo)
- [kayak/service/metrics_runtime.mojo](../../kayak/service/metrics_runtime.mojo)
- [kayak/collections/snapshot_transfer.mojo](../../kayak/collections/snapshot_transfer.mojo)

What is now explicit:

- `CollectionManifest` carries `active_snapshot_id`
- snapshot publication writes the snapshot manifest first, then updates the
  collection manifest to point at that snapshot
- service metrics prefer the active snapshot pointer rather than inferring only
  from the highest generation
- snapshot bundle export/import preserves the active snapshot semantics

This gives the engine a safer publication boundary:

- if the snapshot manifest write succeeds but collection publication fails, the
  new snapshot exists but is not active
- if the collection manifest points at a snapshot, that snapshot manifest has
  already been written

### Draft state now compacts to a baseline plus mutations

Updated files:

- [kayak/service/draft_state.mojo](../../kayak/service/draft_state.mojo)
- [kayak/service/paths.mojo](../../kayak/service/paths.mojo)
- [kayak/service/runtime.mojo](../../kayak/service/runtime.mojo)

What is now explicit:

- after snapshot publication, the draft state is compacted into baseline
  sidecars under `draft/packed_index`, `draft/text_corpus`, and
  `draft/document_metadata`
- the mutation log is cleared back to `mutation_count = 0`
- later mutations replay on top of that baseline

Important bug found and fixed during verification:

- the first implementation incorrectly returned an empty draft when
  `mutation_count == 0`, even if a compacted baseline existed
- that made the next mutation compute its `document_count` against an empty
  state, which later broke replay validation
- the loader now returns the compacted baseline when present and only returns an
  empty draft when there is neither a baseline nor pending mutations

## Guardrails

What is now justified:

- individual manifest and sidecar text writes are atomic at the file level
- publication tracks a real active snapshot pointer
- live-snapshot metrics and bundle transfer respect that pointer
- the draft state after sealing is a compacted baseline rather than an
  ever-growing mutation log
- baseline-plus-mutation replay works across a post-snapshot mutation and a
  second snapshot

What is still not justified:

- atomic multi-file transactions across an entire snapshot publish
- crash-safe rollback across segment sealing plus publication
- garbage collection of superseded snapshots or draft artifacts
- any WARP/GEM/native-engine claim beyond the existing exact/hosted substrate

## Verification

Focused suites rerun after the change:

- `pixi run test_collection_storage`
- `pixi run test_collections`
- `pixi run test_service_contracts`
- `pixi run test_snapshot_bundle`
- `pixi run test_collection_compaction`
- `pixi run test_service_runtime`
- `pixi run test_collection_resolution`

All passed on this branch.

## Conclusion

This is the right substrate work before heavier native search work:

- publication semantics are more honest
- draft-state growth is bounded by compaction
- active snapshot continuity is explicit across runtime, compaction, and bundle
  transfer

The next sound step remains architecture-neutral engine work, not deeper
commitment to one current stage-1 generator.
