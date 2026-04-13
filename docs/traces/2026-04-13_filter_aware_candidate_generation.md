# 2026-04-13: native metadata-aware candidate generation

## Claim

Kayak now keeps structured metadata filters native on the `document_proxy` and
centroid stage-1 paths instead of forcing blanket exact fallback.

## Why this change is justified

- The repo already had typed filter expressions and exact-only guardrails.
- The missing product piece was not filter parsing; it was native candidate
  generation that remained correct under realistic metadata filters.
- A native stage-1 that breaks under structured filters is not a feature-complete
  hosted search engine.

## Design

- Added a persisted `document_filter_index` sidecar that stores exact
  field/value -> document postings per segment.
- The sidecar is sealed for every new segment, even when it contains no
  postings, so empty-metadata segments still participate in native filtered
  search without forcing exact fallback.
- `document_proxy` and centroid generators now derive a segment-local allowlist
  from that sidecar and push it into candidate generation.
- Planner availability is request-aware:
  - if a metadata filter is requested and every segment has the sidecar, native
    proxy/centroid stage-1 remains eligible
  - if an older snapshot is missing the sidecar, the planner falls back to
    `exact_full_scan`
- `gem_graph` remains `match_all`-only in this tranche.

## Evidence

Verified with focused local tests:

- `pixi run mojo -I . tests/test_document_filter_index.mojo`
  - 3/3 passed
- `pixi run mojo -I . tests/test_stage1_capabilities.mojo`
  - 4/4 passed
- `pixi run mojo -I . tests/test_snapshot_inventory.mojo`
  - 3/3 passed
- `pixi run mojo -I . tests/test_snapshot_bundle.mojo`
  - 3/3 passed
- `pixi run mojo -I . tests/test_collection_search_plan.mojo`
  - 31/31 passed
- `pixi run mojo -I . tests/test_service_runtime.mojo`
  - 19/19 passed

Key behavioral checks covered by those tests:

- exact equivalence between `document_filter_index` allowlists and the existing
  exact metadata filter runtime
- native proxy/centroid execution still works for exact `doc_id` filters
- planned metadata-filtered search now stays on native centroid stage-1 for new
  snapshots
- planner still falls back to exact on snapshots that do not contain the new
  sidecar
- snapshot export/import remains compatible with older bundles that do not have
  `document_filter_index`

## Limits

- This tranche proves correctness and hosted continuity, not latency wins.
- No dedicated selectivity benchmark was added yet, so there is still no sound
  performance claim about the cost/benefit frontier under low-selectivity or
  high-selectivity filters.
- Tenant-aware selectivity on shared hosted layouts is still open work.
