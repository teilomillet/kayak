# 2026-04-13: hosted-engine P0 mainline tranche

## Why this step

The hosted-engine grand plan identified five mainline blockers:

1. the runtime sealed and published snapshots inside one large function
2. draft mutations rewrote the whole draft state on every write
3. snapshot resolution loaded more search artifacts than a plan actually used
4. generic search-artifact support existed in storage contracts, but the main
   seal path was still using older dedicated-field construction style
5. correctness and soundness needed tests that proved the new boundaries

This step focused on concrete implementation claims, not on broader field
claims about late-interaction efficiency.

## What changed

### Capability-aware resolution

New file:

- [kayak/collections/resolution_requirements.mojo](../../kayak/collections/resolution_requirements.mojo)

Resolver changes:

- [kayak/collections/resolver.mojo](../../kayak/collections/resolver.mojo)

What is now explicit:

- exact-only snapshot loading
- selective search-artifact family loading
- optional text-corpus loading

Search and explain now resolve only the artifacts required by the chosen plan
instead of always loading all sidecars.

### Explicit seal and publish steps

New files:

- [kayak/collections/segment_builder.mojo](../../kayak/collections/segment_builder.mojo)
- [kayak/collections/publish.mojo](../../kayak/collections/publish.mojo)

Runtime changes:

- [kayak/service/runtime.mojo](../../kayak/service/runtime.mojo)

What is now explicit:

- `seal_single_segment(...)`
- `snapshot_manifest_for_segment(...)`
- `publish_snapshot_manifest(...)`
- `promote_collection_generation(...)`

The hosted runtime no longer hides sealing, snapshot construction, and
collection publication inside one monolithic `create_snapshot(...)` body.

### Append-friendly draft mutations

Updated file:

- [kayak/service/draft_state.mojo](../../kayak/service/draft_state.mojo)

Path additions:

- [kayak/service/paths.mojo](../../kayak/service/paths.mojo)

What is now explicit:

- append-only draft upsert batches
- append-only draft delete batches
- tombstone-style deletes in the draft log
- materialized draft loading by replaying the mutation log
- legacy draft-state compatibility on load

Important limit:

- this is not compaction yet
- it removes whole-draft rewrites from the mutation path, but the draft log can
  still grow until a later cleanup or compaction step

### Scalar-correct storage load

Updated file:

- [kayak/storage/packed_index_store.mojo](../../kayak/storage/packed_index_store.mojo)

What changed:

- `load_stored_packed_index(...)` now returns the scalar name recorded in the
  manifest instead of silently substituting the current build default

That does not broaden scalar support by itself, but it removes one correctness
edge that would have obscured later scalar modularity work.

## Guardrails

Tests rerun after the refactor:

- `pixi run test_collection_resolution`
- `pixi run test_collection_search_plan`
- `pixi run test_service_runtime`
- `pixi run test_storage`
- `pixi run test_storage_invariants`

All passed on this branch.

New tests added in this step:

- selective snapshot loading skips unrelated sidecars and text corpus
- hosted runtime persists append-style draft mutations instead of recreating a
  whole draft packed index

## Verified implementation claims

What is now justified:

- the mainline hosted runtime can resolve snapshots with explicit load
  requirements
- the mainline draft write path no longer rewrites one whole packed index plus
  one whole text corpus after each mutation
- the mainline seal path now constructs search-visible sidecars through the
  generic search-artifact list
- the mainline snapshot path now has explicit seal and publish boundaries

What is not justified by this step:

- atomic publish across crash boundaries
- compaction-safe replacement of long draft mutation logs
- stronger filter execution
- GEM execution in the mainline stage-1 engine
- any broad asymptotic or bytes/vector claim from the 2030 discussion

## Conclusion

This step completes the hosted-engine P0 tranche in the sense intended by the
grand plan:

- the hosted-engine substrate is materially easier to browse
- the mainline write and read paths are closer to a real engine shape
- the changes are backed by targeted tests

The next sound mainline moves are no longer "make the current blob faster."
They are:

- filter and metadata sidecars
- compaction and publish-safe replacement
- richer service metrics
- and then heavier native stage-1 work once the engine boundary is stable
