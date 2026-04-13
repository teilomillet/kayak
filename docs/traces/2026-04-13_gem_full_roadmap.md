# GEM Full Implementation Roadmap

Date: 2026-04-13

## Goal

Bring Kayak's current `gem_graph` path from a paper-shaped approximation to a
reference-shaped implementation that is close enough to the public GEM code and
paper to support meaningful comparison work against WARP.

The guiding constraint is epistemic: only treat a feature as "GEM" when it is
verified in the paper or the public reference implementation.

## Verified Current Gaps

The current Kayak implementation already has:

- two-stage centroid hierarchy
- TF-IDF-style coarse cluster profiles
- multi-entry search over an explicit document graph
- qCH-style search-time proxy distance
- exact late-interaction rerank

The verified gaps, from direct inspection of the paper and public code, are:

1. `qEMD` is currently a greedy approximation.
Reason: `kayak/index/gem_graph.mojo` uses `greedy_quantized_emd_distance`,
while the reference uses `EMD_wrap_self(...)` over a full quantized cost matrix.

2. qCH/qEMD currently use squared L2 centroid distances rather than the
reference's quantized similarity table.
Reason: the reference computes distance as `1 - cluster_dis[...]`, where
`cluster_dis` is built from centroid dot products.

3. Cluster-graph construction is still pairwise and post-hoc.
Reason: the reference inserts documents cluster-by-cluster, using graph search
to find neighbors and a bridge-preserving merge path for shared documents.

4. Adaptive cluster cutoff is not implemented.
Reason: the paper uses a lightweight decision-tree classifier over top TF-IDF
scores and vector count to predict per-document `r`.

5. Shortcut injection is not implemented.
Reason: the paper explicitly augments the graph with supervised semantic
shortcuts derived from training pairs.

## Implementation Phases

### Phase 1: Exact Quantized Distance Kernels

Deliverables:

- exact transport-based `qEMD` over quantized code histograms
- reference-shaped `qCH` and cluster-distance table using `1 - dot`
- focused unit tests for transport and Chamfer behavior on toy inputs

Why first:

- both graph construction and query traversal depend on these distances
- the current greedy/L2 path is the largest algorithmic divergence

### Phase 2: Bridge-Preserving Graph Construction

Deliverables:

- cluster-by-cluster insertion path
- approximate proximity search during construction over the evolving graph
- bridge-aware neighbor merge that preserves at least one neighbor from each
  retained coarse cluster

Why:

- this is the structural core of GEM's "dual graph" design
- it closes the gap between a generic document graph and a cluster-woven graph

### Phase 3: Supervised GEM Enhancements

Deliverables:

- optional training-pair input for graph building
- adaptive per-document cluster cutoff using a lightweight decision tree
- semantic shortcut injection driven by missed positives

Why optional:

- these features require supervision that may not exist for every build
- making the supervision boundary explicit is more honest than hiding it

### Phase 4: Validation

Deliverables:

- updated unit coverage for the new build path
- existing search-plan regression coverage kept green
- a small trace document summarizing what matches the reference and what still
  differs

Why:

- the repo instructions require that behavior claims be validated or clearly
  marked as unresolved

## Known Remaining Divergence To Track

Even after these phases, one divergence may remain:

- the public reference builds on HNSW internals, while Kayak currently owns an
  explicit graph artifact and search loop

If Kayak keeps the explicit graph representation, the implementation can still
be reference-shaped in distance, bridge handling, cluster filtering, and
shortcut logic, but it should be documented as a different graph substrate.
