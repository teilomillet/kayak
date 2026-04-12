# GEM Integration Plan

Status: `Phase 0 architecture plan`  
Date: `2026-04-13`

This note defines the implementation boundary for adding a GEM-family engine to
`kayak`.

It is intentionally epistemic:
- verified facts are tied to the local codebase or primary sources
- inferences are labeled as such
- non-goals are explicit so the repo does not silently drift into a misleading
  abstraction

## Verified Facts

These statements were checked against the current repo and primary sources.

### Repo facts

1. Stage-1 engine identity is currently a single string in
   [`kayak/planning/candidate_generator.mojo`](../../kayak/planning/candidate_generator.mojo).
2. Stage-1 execution is currently one monolithic branch chain in
   [`kayak/planning/execution.mojo`](../../kayak/planning/execution.mojo).
3. Search-visible segment manifests currently hardcode engine-specific sidecar
   roots:
   - `centroid_postings_root`
   - `centroid_heads_root`
   - `document_proxy_root`
   in [`kayak/collections/segment.mojo`](../../kayak/collections/segment.mojo).
4. Resolved snapshot loading also hardcodes those same families into
   per-field loaded views in
   [`kayak/collections/resolved_snapshot.mojo`](../../kayak/collections/resolved_snapshot.mojo)
   and [`kayak/collections/resolver.mojo`](../../kayak/collections/resolver.mojo).
5. Current non-exact engines in repo are all centroid- or proxy-oriented:
   - `document_proxy`
   - `centroid_postings`
   - `centroid_heads`
   - centroid-postings execution variants
6. Exact late interaction is already isolated as stage 2, which is the right
   stability boundary for comparing approximate stage-1 families.

### Primary-source facts

Sources checked:
- WARP paper: https://arxiv.org/abs/2501.17788
- GEM paper: https://arxiv.org/abs/2603.20336

What was directly verified:

1. WARP is a centroid/residual engine built around:
   - query-centroid scoring
   - centroid probing
   - implicit decompression of residuals
   - token-level max reduction
   - document-level sum reduction with missing-similarity imputation
2. GEM is not a centroid-postings variant. It is a native graph-based set-level
   index built around:
   - set-level clustering
   - dual graph structure
   - metric decoupling using EMD for graph construction and Chamfer/MaxSim-style
     scoring for retrieval
   - semantic shortcuts
   - multi-entry beam search
   - cluster-guided early pruning
3. GEM therefore needs graph-native metadata and query-time counters that WARP
   does not need.

## Decision

`kayak` should keep **late interaction** as the primitive and **stage-aware
search** as the systems contract.

The correct shared abstraction is **not** "WARP/GEM-style engine".

The correct shared abstraction is narrower:
- stage-1 engine family
- stage-1 search artifact registry
- stage-1 family dispatch boundary
- exact stage-2 rerank contract

Reason:
- WARP and GEM share the need for a non-exact stage-1 candidate generator
- they do **not** share the same artifact layout, traversal logic, or profiling
  counters

## What Must Stay Stable

These parts already match the repo's direction and should remain stable.

1. `EncodedQuery`, `EncodedDocument`, and `PackedIndex` remain the exact
   late-interaction substrate.
2. Exact reranking remains the correctness anchor.
3. `CandidateSet` remains the public benchmark/reporting surface for stage 1.
4. Search plans remain explicit; no hidden engine switching should be added.

## What Must Change

### 1. Segment manifests must stop naming engine families in top-level fields

The repo should move from:
- one field per sidecar family

to:
- one generic list of stage-1 search artifacts

Minimal required artifact metadata:
- `family`
- `root`

Why this is justified:
- WARP-family engines can continue to use centroid-oriented artifacts
- GEM-family engines can register a graph artifact without forcing new
  top-level manifest fields such as `gem_graph_root`

### 2. Loaded snapshots must stop carrying per-family booleans as the primary API

The repo should move from:
- `has_centroid_postings_index`
- `stored_centroid_postings_index`
- `has_document_proxy_index`
- `stored_document_proxy_index`

to:
- loaded search artifacts plus family-specific accessors

Why this is justified:
- the loader must remain explicit about what is available
- the loaded view should no longer assume the universe of engine families is
  closed

### 3. Candidate generators must carry family identity explicitly

The repo should stop treating generator identity as a single opaque string.

Minimal required contract:
- `kind`
- `family`
- `artifact_family`

Why this is justified:
- `centroid_postings_imputed_flat` and `centroid_heads` are different kinds in
  one centroid family
- `gem_graph` belongs to a different family entirely

### 4. Stage-1 execution must dispatch by family module

The repo should move from:
- one large `if/elif` chain in `execution.mojo`

to:
- family-level dispatch
- kind-specific handling within each family module

Why this is justified:
- WARP-family code will remain centroid/residual-oriented
- GEM-family code will remain graph-oriented
- comparing families becomes easier if their execution entry points are already
  separate

### 5. Stage profiling must support graph-native counters

The current profile/reporting surface is adequate for vector and byte budgets,
but not for graph traversal.

The next profile extension should support counters such as:
- visited vertices
- expanded edges
- visited clusters
- entry-point count
- beam width or search frontier size

This is a source-backed inference from GEM, not a currently implemented repo
fact.

## What We Should Not Do

These are explicit non-goals.

1. Do **not** model GEM as another centroid-postings variant.
2. Do **not** add `gem_graph_root` beside `centroid_postings_root` and
   `document_proxy_root`.
3. Do **not** overload centroid parameters such as `centroid_budget` or
   `posting_cap` to mean graph parameters.
4. Do **not** claim full GEM behavior before graph construction, shortcuts,
   multi-entry search, and graph-native pruning are implemented and measured.
5. Do **not** merge WARP and GEM into one "native engine" kind that hides which
   retrieval geometry is actually being used.

## Immediate Implementation Phases

### Phase 1

Refactor shared infrastructure only.

Deliverables:
- generic stage-1 search artifact registry in segment manifests
- loaded search-artifact boundary in resolved snapshots
- family-aware `CandidateGenerator`
- family-aware stage-1 dispatch

Exit criterion:
- all current exact, proxy, and centroid engines still work with unchanged
  behavior

### Phase 2

Add a GEM-family scaffold without claiming a production graph engine.

Deliverables:
- `gem_graph` candidate-generator family entry
- `StoredGemGraphIndex` storage contract
- manifest/load/save plumbing for GEM-family artifact metadata
- explicit "unsupported engine execution" error path until graph search is
  implemented

Exit criterion:
- the repo can represent a GEM-family artifact and a GEM-family plan without
  pretending that centroid code can execute it

### Phase 3

Implement and measure a first graph-native search path.

Deliverables:
- graph construction path
- graph search path
- graph-native profiling counters
- WARP-vs-GEM comparison traces on identical slices

Exit criterion:
- the comparison is backed by measured candidate recall, reranked quality,
  storage, and latency

## Why This Is The Right Comparison Boundary

This plan is meant to make WARP and GEM comparable without forcing either one
through the other's storage or execution model.

That is the important design point:
- exact rerank is shared
- stage-1 reporting is shared
- stage-1 infrastructure is shared
- the actual approximate search geometry remains family-specific

That keeps the repo honest.
