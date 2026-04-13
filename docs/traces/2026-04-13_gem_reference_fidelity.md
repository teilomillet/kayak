# GEM Reference Fidelity Trace

## Objective

Reduce the remaining deltas between Kayak's GEM path and the public GEM reference implementation, then verify what can and cannot be compared directly on this host.

## Verified alignment changes

- Cluster graph construction now applies an HNSW-style diversification heuristic when retaining neighbors.
  - Reference basis: `getNeighborsByHeuristic2Cluster(...)` in `/tmp/sigmod26gem-ref/hnswlib/hnswlib/hnswalg.h:2180`.
  - Kayak implementation: `select_neighbors_by_cluster_heuristic(...)` in [kayak/index/gem_graph.mojo](/private/tmp/kayak-gem-full-wt/kayak/index/gem_graph.mojo:1450).

- Cluster entry points now mirror the public reference's deterministic "first member of the cluster" policy.
  - Reference basis: `cluster_entries[i] = cluster_set[i][0]` in `/tmp/sigmod26gem-ref/hnswlib/examples/cpp/example_vecset_search_gem.cpp:151`.
  - Kayak implementation: entry selection in [kayak/index/gem_graph.mojo](/private/tmp/kayak-gem-full-wt/kayak/index/gem_graph.mojo:1933).

## Verified tests

- `tests/test_gem_graph_full.mojo`
  - covers bridge preservation
  - covers first-member cluster entry selection
  - covers the extracted HNSW diversification rule on a manual qEMD fixture
  - covers adaptive cutoff and shortcut injection

- `tests/test_gem_graph_index.mojo`
- `tests/test_gem_graph_store.mojo`
- `tests/test_gem_transport.mojo`
- `tests/test_collection_resolution.mojo`
- `tests/test_collection_search_plan.mojo`

## Public reference parity status

### What was attempted

A minimal compile against the public reference header:

```bash
clang++ -std=c++17 -I/tmp/sigmod26gem-ref/hnswlib/hnswlib /tmp/gem_ref_smoke.cpp -c -o /tmp/gem_ref_smoke.o
```

### Result

Compilation fails on this machine because the public reference hardcodes x86 intrinsics via `hnswlib.h`, while this host is Apple Silicon (`arm64-apple-darwin`).

Representative compiler evidence:

- `immintrin.h: "This header is only meant to be used on x86 and x64 architecture"`
- multiple `__builtin_ia32_*` errors from the forced x86 include path

### Conclusion

On this host, we can verify algorithmic parity against extracted public-reference logic, but we cannot run an executable side-by-side parity benchmark against the unmodified public C++ code.

To complete true executable parity testing, one of these is needed:

- an x86_64 Linux or macOS runner
- a reference fork patched to avoid forced x86 intrinsics on arm64
- a prebuilt x86 environment with the public GEM reference available
