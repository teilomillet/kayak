# 2026-04-13: metadata sidecars and exact metadata filtering

## Why this step

The hosted runtime had an exact-only filter substrate, but it could only filter
on `doc_id`. That was honest, but incomplete for the hosted-engine direction:
document fields need to survive ingest, sealing, compaction, export/import, and
exact evaluation before metadata filtering can be claimed at all.

The user clarified the contract that matters here:

- metadata updates merge into existing document metadata
- an empty string deletes a key

This step implements that contract end to end without widening claims beyond
what the engine can actually verify.

## What changed

New files:

- [kayak/collections/document_metadata.mojo](../../kayak/collections/document_metadata.mojo)
- [kayak/collections/document_metadata_store.mojo](../../kayak/collections/document_metadata_store.mojo)

Updated areas:

- service request boundary:
  [kayak/service/document_requests.mojo](../../kayak/service/document_requests.mojo)
- draft mutation persistence:
  [kayak/service/draft_state.mojo](../../kayak/service/draft_state.mojo)
- sealing and snapshot continuity:
  [kayak/collections/segment_builder.mojo](../../kayak/collections/segment_builder.mojo)
  [kayak/collections/resolver.mojo](../../kayak/collections/resolver.mojo)
  [kayak/collections/compaction_runtime.mojo](../../kayak/collections/compaction_runtime.mojo)
  [kayak/collections/snapshot_transfer.mojo](../../kayak/collections/snapshot_transfer.mojo)
- exact filter evaluation:
  [kayak/filters/runtime.mojo](../../kayak/filters/runtime.mojo)
  [kayak/planning/execution_exact_family.mojo](../../kayak/planning/execution_exact_family.mojo)
  [kayak/planning/exact_stage.mojo](../../kayak/planning/exact_stage.mojo)

## Contract now implemented

Metadata ingest:

- `UpsertDocument` can now carry explicit metadata updates
- updates merge with prior metadata for that document
- `""` means remove the key from stored metadata

Metadata persistence:

- draft upsert batches store normalized post-merge metadata state
- sealed segments persist a `document_metadata` sidecar only when at least one
  document has metadata
- compaction preserves metadata sidecars
- snapshot export/import preserves metadata sidecars

Exact filtering:

- exact full-scan stage-1 can now filter on metadata fields as well as `doc_id`
- filter evaluation happens against the loaded document metadata aligned with the
  packed index

## Guardrails

What is now justified:

- merge semantics for document metadata updates
- empty-string deletion semantics
- exact metadata filtering on `exact_full_scan`
- metadata continuity across sealing, compaction, and bundle transfer

What is still not justified:

- metadata-aware approximate stage-1 filtering
- candidate-pushdown filtering for proxy, centroid, or graph generators
- typed or nested metadata values

The storage format is deliberately conservative in this tranche:

- keys and stored values must be inline strings
- tabs and newlines are rejected
- blank stored values are not allowed because blank is reserved for deletion at
  update time

## Verification

Focused suites rerun after the change:

- `pixi run test_filters`
- `pixi run test_service_contracts`
- `pixi run test_service_json`
- `pixi run test_service_runtime`
- `pixi run test_collection_resolution`
- `pixi run test_collection_storage`
- `pixi run test_snapshot_bundle`
- `pixi run test_collection_compaction`
- `pixi run test_collection_search_plan`

All passed on this branch.

## Conclusion

This makes the hosted-engine surface materially more useful without overstating
support:

- metadata is now a real first-class sidecar
- exact filtering can use it honestly
- approximate filtered retrieval remains explicitly out of scope until a real
  candidate-pushdown path exists
