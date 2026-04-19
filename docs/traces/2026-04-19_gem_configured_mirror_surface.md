# GEM Configured Mirror Surface

Date: `2026-04-19`
Status: `implemented and locally validated`

## Objective

Expose the full `GemGraphBuildConfig` through the collection-mirror setup seam
without changing the default frontier helpers to silently self-train on
evaluation labels.

This is the right scope because:

- the active GEM builder already supports adaptive cutoff and shortcuts
- mirror-based collections were still limited to the old
  `(fine_cluster_count, coarse_cluster_count, cluster_cutoff)` tuple
- the benchmark/setup primitive should be able to express the richer GEM build
  contract explicitly
- public frontier helpers should **not** automatically derive supervised GEM
  training pairs from the same judged queries they later evaluate on

## What Changed

Added full-config GEM mirror overloads in:

- [`kayak/collections/mirror.mojo`](../../kayak/collections/mirror.mojo)

The mirror surface can now build GEM sidecars from either:

- the historical tuple path
- a full `GemGraphBuildConfig`

The tuple path remains unchanged for existing callers.

Added a public benchmark dataset helper overload in:

- [`kayak/benchmarks/public_benchmark_dataset.mojo`](../../kayak/benchmarks/public_benchmark_dataset.mojo)

This allows future benchmark or experiment entry points to opt into an explicit
GEM build config when they have an epistemically valid source of supervision.

## Validation

Focused mirror test:

```bash
pixi run mojo -I . tests/test_collection_mirror.mojo
```

Observed result:

- `4` tests passed locally
- this includes configured GEM-sidecar coverage via
  `test_collection_mirror_can_build_configured_gem_graph_sidecar`

Benchmark-facing compile check:

```bash
pixi run mojo build benchmarks/public_small_window_frontier.mojo -I . -o /tmp/kayak-public-small-window-frontier
```

Observed result:

- compilation succeeded locally

## Current Boundary

What this step enables:

- mirror-backed experiments can now exercise adaptive cutoff and shortcut-aware
  GEM builds explicitly
- future benchmark code can opt into richer GEM builds without changing the
  default unsupervised frontier path

What this step intentionally does **not** do:

- it does not enable adaptive cutoff or shortcuts by default on the public
  frontier
- it does not derive GEM training pairs from the same judged task used for
  evaluation

That would be an evaluation-design choice, not a harmless plumbing change.

## Next Honest Question

If we want to benchmark supervised GEM features on public slices, the next
thing needed is not more mirror plumbing.

The next thing needed is a justified supervision source, for example:

- a separate training split
- an external judged source with no query leakage into the evaluation slice
- or a synthetic benchmark where the supervision boundary is deliberately part
  of the fixture design
