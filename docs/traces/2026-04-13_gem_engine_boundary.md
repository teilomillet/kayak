# GEM Engine Boundary Refactor

Date: `2026-04-13`

This trace records what was changed, why it was justified, and what was
verified locally.

## Claim

`kayak` needs a stage-1 abstraction that can host both:
- WARP-family centroid/residual engines
- GEM-family graph engines

without forcing either family through hardcoded manifest fields or one monolithic
execution branch.

## Evidence Checked

### Local code inspection

Verified before the refactor:
- segment manifests hardcoded `centroid_postings_root`,
  `centroid_heads_root`, and `document_proxy_root`
- resolved snapshots hardcoded those same artifact families as dedicated loaded
  fields
- stage-1 execution was one large `if` chain over generator kinds

Touched files:
- [`kayak/collections/segment.mojo`](../../kayak/collections/segment.mojo)
- [`kayak/collections/segment_store.mojo`](../../kayak/collections/segment_store.mojo)
- [`kayak/collections/resolved_snapshot.mojo`](../../kayak/collections/resolved_snapshot.mojo)
- [`kayak/collections/resolver.mojo`](../../kayak/collections/resolver.mojo)
- [`kayak/planning/candidate_generator.mojo`](../../kayak/planning/candidate_generator.mojo)
- [`kayak/planning/execution.mojo`](../../kayak/planning/execution.mojo)

### Primary sources

Checked again during this step:
- WARP paper: https://arxiv.org/abs/2501.17788
- GEM paper: https://arxiv.org/abs/2603.20336

What matters for the abstraction boundary:
- WARP is centroid/residual-oriented
- GEM is graph-oriented
- both need stage-1 candidate generation, but they do not share the same search
  geometry

## Implemented Change

### 1. Generic search-artifact registry

Added:
- [`kayak/collections/search_artifact.mojo`](../../kayak/collections/search_artifact.mojo)

Changed:
- `SealedSegmentManifest` now stores `search_artifacts`
- legacy constructors remain so existing call sites continue to work
- manifest persistence now writes:
  - generic `search_artifact_*` entries
  - legacy shadow root fields for compatibility

### 2. Generic loaded search-artifact view

Added:
- `LoadedSearchArtifact`
- family-specific accessors for:
  - centroid postings
  - centroid heads
  - document proxy
  - GEM graph metadata

Reason:
- keep current callers explicit
- stop assuming the set of stage-1 families is closed

### 3. Family-aware candidate generators and execution

Changed:
- `CandidateGenerator` now carries:
  - `kind`
  - `family`
  - `artifact_family`
- stage-1 execution is now dispatched by family module:
  - exact
  - proxy
  - centroid
  - graph

### 4. GEM storage scaffold

Added:
- [`kayak/storage/gem_graph_store.mojo`](../../kayak/storage/gem_graph_store.mojo)
- `StoredGemGraphIndex` metadata contract

This is intentionally a metadata scaffold, not a finished graph engine.

It persists:
- `document_count`
- `cluster_count`
- `graph_edge_count`
- `shortcut_edge_count`
- `entry_point_count`
- `quantization_centroid_count`
- `artifact_byte_size`

### 5. Explicit non-implementation boundary

Added:
- `gem_graph` candidate generator
- `gem_graph_search_plan`

Current behavior:
- graph-family execution raises an explicit "registered but not implemented"
  error

Reason:
- the repo can now model GEM-family storage and planning honestly
- it still does not claim graph retrieval exists before it is built and
  benchmarked

## Local Verification

Commands run:

```bash
pixi run test_collection_storage
pixi run test_collection_resolution
pixi run test_collection_search_plan
pixi run mojo -I . tests/test_gem_graph_store.mojo
pixi run mojo -I . tests/test_collection_contracts.mojo
pixi run mojo -I . tests/test_snapshot_bundle.mojo
```

Observed results:
- `test_collection_storage`: passed
- `test_collection_resolution`: passed
- `test_collection_search_plan`: passed
- `test_gem_graph_store`: passed
- `test_collection_contracts`: passed
- `test_snapshot_bundle`: passed

Specific new coverage:
- generic search-artifact manifest roundtrip
- resolved snapshot loading of GEM graph metadata
- explicit `gem_graph` search-plan registration without fake execution
- snapshot bundle export/import after the manifest refactor

## Conclusion

The verified outcome is narrower than "GEM implemented":
- the repo now has the correct shared boundary for comparing WARP-family and
  GEM-family engines
- the repo does not yet have a graph retrieval implementation

That is the intended result of this step.
