# GEM Status Reconciliation

Date: `2026-04-19`
Status: `verified against current code and tests`

## Objective

Reconcile the current in-tree GEM implementation with older April 13 trace notes
that still describe transport, adaptive cutoff, and shortcut support as missing.

This is not a cosmetic note.

The contradiction matters because `docs/recent_paper_targets.md` uses GEM as the
next major paper-family target, so repo direction should be based on current
facts rather than historical intermediate states.

## Verified Facts

### Exact transport is implemented on the live path

Evidence:

- [`kayak/index/gem_transport.mojo`](../../kayak/index/gem_transport.mojo)
  implements `min_cost_transport_distance(...)`
- [`kayak/index/gem_graph.mojo`](../../kayak/index/gem_graph.mojo) defines
  `exact_quantized_emd_distance(...)`
- the current non-legacy GEM builder path is
  `build_gem_graph_index_with_config(...)`
- the only remaining greedy qEMD usage is inside
  `build_gem_graph_index_legacy(...)`, which is not referenced outside
  `gem_graph.mojo`

Verification command:

```bash
pixi run mojo -I . tests/test_gem_transport.mojo
```

Observed result:

- `1` test passed locally

### Adaptive cluster cutoff is implemented on the live path

Evidence:

- [`kayak/index/gem_graph.mojo`](../../kayak/index/gem_graph.mojo) carries:
  - `enable_adaptive_cluster_cutoff`
  - `adaptive_cluster_cutoff_max`
  - `build_adaptive_profile_limits(...)`
- the current builder applies adaptive profile limits before final document
  profile materialization

Verification command:

```bash
pixi run mojo -I . tests/test_gem_graph_full.mojo
```

Observed result:

- `5` tests passed locally
- this includes
  `test_adaptive_cutoff_can_keep_more_clusters_than_fixed_cutoff`

### Semantic shortcut injection is implemented on the live path

Evidence:

- [`kayak/index/gem_graph.mojo`](../../kayak/index/gem_graph.mojo) defines
  `inject_shortcuts(...)`
- the current builder conditionally applies shortcut injection when
  `enable_shortcuts` is enabled

Verification command:

```bash
pixi run mojo -I . tests/test_gem_graph_full.mojo
```

Observed result:

- the same local run passed
- this includes `test_shortcut_injection_adds_missing_semantic_edge`

### The stale statements are historical, not current

Verified stale notes:

- [`docs/traces/2026-04-13_gem_phase3_paper_shaped.md`](2026-04-13_gem_phase3_paper_shaped.md)
- [`docs/traces/2026-04-13_gem_full_roadmap.md`](2026-04-13_gem_full_roadmap.md)

Those notes still describe exact transport, adaptive cutoff, and shortcuts as
future work. That no longer matches the checked code or focused test surface.

## Current Narrow Conclusion

The next GEM step should **not** be:

- initial transport implementation
- initial adaptive cutoff implementation
- initial shortcut implementation

Those steps already exist in-tree on the active builder path.

The next GEM step should instead target the still-open question that remains
backed by existing benchmark traces:

- pruning quality and graph traversal efficiency on the measured frontier

That conclusion is consistent with:

- [`docs/traces/2026-04-13_gem_frontier_comparison.md`](2026-04-13_gem_frontier_comparison.md)

Specifically:

- the repo can already express and execute a paper-shaped GEM family
- the measured issue is not "missing GEM features"
- the measured issue is that current GEM still explores too much graph to earn
  a clear frontier advantage

## What This Note Does Not Claim

- it does **not** claim full public-reference parity on this host
- it does **not** claim GEM is now the preferred default stage-1 engine
- it does **not** claim the legacy builder should be removed immediately

It only claims the narrower verified point:

- repo planning should stop treating transport, adaptive cutoff, and shortcuts
  as the next missing GEM primitives
