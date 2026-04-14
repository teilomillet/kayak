# Centroid Workspace Search Breakdown on BrowseComp-Plus Gold

## Claim

The earlier end-to-end check on `browsecomp_plus_gold` showed near-parity total
search time, but it still left one unresolved question:

- is the workspace optimization too small to matter end to end because stage-1
  is a small share of the search path, or because the earlier result was only
  noisy

This trace measures the boundary directly by benchmarking:

- stage-1 candidate generation
- stage-2 reranking from precomputed candidate windows
- whole search

for the same imputed centroid plans on the same slice.

## Benchmark surface

Added:

- `benchmarks/profile_centroid_workspace_search_breakdown.mojo`

Measured plans:

- `centroid_postings_imputed`
- `centroid_postings_imputed_flat`

Measured benchmark kinds:

- `stage1_fresh_workspace`
- `stage1_reused_workspace`
- `stage2_from_candidates`
- `search_fresh_workspace`
- `search_reused_workspace`

The benchmark verifies before timing that:

- fresh and reused candidate generation return the same candidate hit identities
- fresh and reused whole-search execution return the same final hit identities

So any timing difference is not coming from an obvious semantic change.

Artifacts:

- `.cache/kayak/profile_centroid_workspace_search_breakdown.tsv`
- `.cache/kayak/bench_quiet/20260414T202233Z`

## Commands run

```bash
pixi run mojo -I . benchmarks/profile_centroid_workspace_search_breakdown.mojo
bash scripts/run_bench_quiet.sh --timeout-seconds 20 --force --repeats 1 -- pixi run mojo -I . benchmarks/profile_centroid_workspace_search_breakdown.mojo
```

The quiet wrapper again timed out waiting for a quiet host and force-ran the
benchmark. Pre-run competing load samples ranged from roughly `379.30` to
`636.30` aggregate `%CPU` from other processes, so the wrapped run is still
host-contended.

## Wrapped result

Measured means:

- `centroid_postings_imputed`
  - stage1 fresh: `4.543463096309631e-04 s`
  - stage1 reused: `4.5109305058191584e-04 s`
  - stage2 from candidates: `2.2539228874106925e-03 s`
  - search fresh: `2.8110515071700323e-03 s`
  - search reused: `3.0684335622573906e-03 s`
- `centroid_postings_imputed_flat`
  - stage1 fresh: `4.544799245785271e-04 s`
  - stage1 reused: `4.4015754405286346e-04 s`
  - stage2 from candidates: `2.2942761834682363e-03 s`
  - search fresh: `2.6707811027985593e-03 s`
  - search reused: `2.7809116465863453e-03 s`

Derived ratios and shares:

- `centroid_postings_imputed`
  - stage1 reuse ratio: `0.992840`
  - search reuse ratio: `1.091561`
  - stage1 share of fresh search: `0.161629`
  - stage2 share of fresh search: `0.801808`
  - stage1 delta: `-0.003253 ms`
  - search delta: `+0.257382 ms`
- `centroid_postings_imputed_flat`
  - stage1 reuse ratio: `0.968486`
  - search reuse ratio: `1.041235`
  - stage1 share of fresh search: `0.170167`
  - stage2 share of fresh search: `0.859028`
  - stage1 delta: `-0.014322 ms`
  - search delta: `+0.110131 ms`

## Direct-run context

The direct run also showed stage-1 in the same rough band:

- around `0.47 ms` to `0.48 ms` for stage-1
- around `2.46 ms` to `2.50 ms` for stage2-from-candidates

That directional stability matters more than the sign of the end-to-end delta:

- the stage split is stable
- the whole-search latency sign is not

## Interpretation

What is verified:

- the reusable workspace still reduces stage-1 time on this slice
- stage-1 is only about `16%` to `17%` of fresh whole-search time here
- stage2-from-candidates alone is about `80%` to `86%` of fresh whole-search
  time here
- fresh and reused paths still match on candidate identities and final hit
  identities

What is **not** justified:

- claiming a reliable whole-search latency win from workspace reuse on this
  workload

The critical comparison is scale, not just sign:

- the measured stage-1 savings are on the order of `0.003 ms` to `0.014 ms`
- the whole-search movement under host contention is on the order of
  `0.110 ms` to `0.257 ms`

That means the local stage-1 improvement is too small relative to the dominant
stage-2 cost and normal run-to-run noise to cleanly move total search latency
on this slice.

## Bottom line

The most defensible combined reading is now:

- the reusable workspace is a correct stage-1 optimization with controlled
  benchmark support
- on `browsecomp_plus_gold`, stage-1 is not the dominant cost center
- stage2 late interaction dominates enough of the total path that the workspace
  optimization cannot be sold as an end-to-end latency breakthrough

So if the next goal is a visible whole-search win on heavier real slices, the
next target should move to stage2 exact late interaction rather than more
workspace tuning.
