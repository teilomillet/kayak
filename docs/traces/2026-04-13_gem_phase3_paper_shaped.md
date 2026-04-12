# GEM Phase 3: Paper-Shaped Implementation

Date: `2026-04-13`
Status: `implemented and locally validated in worktree`

## Goal

Move `gem_graph` from a metadata-only scaffold to a paper-shaped native stage-1
engine that mirrors the GEM paper closely enough to compare honestly against
other stage-1 families.

## Verified Outcome

The repo now has a real `gem_graph` path with:

- a stored `GemGraphIndex` artifact
- two-stage clustering
  - fine quantization centroids
  - coarse index centroids
- per-document quantized code histograms
- TF-IDF-style coarse cluster profiles
- cluster memberships and deterministic per-cluster entry docs
- graph adjacency built over documents
- query-time cluster filtering
- multi-entry graph traversal
- cluster-guided pruning
- exact late-interaction rerank preserved as stage 2

## Files

Core index and storage:

- [`kayak/index/gem_graph.mojo`](../../kayak/index/gem_graph.mojo)
- [`kayak/storage/gem_graph_store.mojo`](../../kayak/storage/gem_graph_store.mojo)
- [`kayak/storage/metadata.mojo`](../../kayak/storage/metadata.mojo)

Execution and plan surface:

- [`kayak/planning/candidate_generator.mojo`](../../kayak/planning/candidate_generator.mojo)
- [`kayak/planning/search_plan.mojo`](../../kayak/planning/search_plan.mojo)
- [`kayak/planning/execution_graph_family.mojo`](../../kayak/planning/execution_graph_family.mojo)
- [`kayak/planning/json.mojo`](../../kayak/planning/json.mojo)
- [`kayak/service/json.mojo`](../../kayak/service/json.mojo)

Focused validation:

- [`tests/test_gem_graph_index.mojo`](../../tests/test_gem_graph_index.mojo)
- [`tests/test_gem_graph_store.mojo`](../../tests/test_gem_graph_store.mojo)
- [`tests/test_collection_resolution.mojo`](../../tests/test_collection_resolution.mojo)
- [`tests/test_collection_search_plan.mojo`](../../tests/test_collection_search_plan.mojo)

## What Mirrors The Paper

These pieces were implemented to match the paper's structure directly:

1. Two-stage clustering.
   The artifact stores fine quantization centroids and coarse index centroids.

2. TF-IDF-style cluster profiles.
   Each document is assigned to coarse clusters by quantized token support, then
   reduced to a top-`r` profile.

3. Native graph artifact.
   The graph is built over documents rather than pretending the stage is a
   centroid-postings variant.

4. Cluster-filtered multi-entry search.
   Query-time execution first finds relevant clusters, then starts traversal
   from one entry document per relevant cluster.

5. Cluster-guided pruning.
   Neighbors outside the relevant coarse-cluster profile are skipped before
   further expansion.

## Remaining Deviations

These are explicit and important.

1. qEMD is currently a greedy transport approximation.
   The paper uses optimal qEMD; this repo does not yet have a dedicated optimal
   transport solver.

2. Semantic shortcut injection is not implemented yet.
   The paper adds supervised shortcut edges from training pairs. The current
   path keeps `shortcut_edge_count = 0`.

3. Adaptive per-document cluster cutoff is not implemented yet.
   The paper uses a learned decision tree to predict `r`. The current repo uses
   the explicit `cluster_cutoff` supplied at build time.

4. Entry-point selection is deterministic, not random.
   The paper samples one entry per relevant cluster. The current repo uses a
   deterministic highest-profile document per cluster so tests remain stable.

5. Graph construction is pairwise within clusters, not incremental APG search.
   This keeps the implementation small and testable, but it is not yet the
   paper's full construction procedure.

## Validation

Commands run:

```bash
pixi run mojo -I . tests/test_gem_graph_index.mojo
pixi run mojo -I . tests/test_gem_graph_store.mojo
pixi run mojo -I . tests/test_collection_resolution.mojo
pixi run mojo -I . tests/test_collection_search_plan.mojo
```

Observed result:

- all listed tests passed locally in the Phase 3 worktree

## Next Steps

The next high-value steps are now narrower and better justified:

1. Replace greedy qEMD with an explicit optimal transport solver.
2. Add supervised shortcut injection from judged query-document pairs.
3. Add adaptive `r` prediction rather than fixed build-time cutoff.
4. Benchmark this graph family against centroid and proxy families on the same
   candidate budgets and exact rerank stage.
