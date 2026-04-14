# Centroid Workspace End-to-End Check on BrowseComp-Plus Gold

## Claim

After the controlled stage-1 scale benchmarks, the next question was narrower
and more operational:

- does the reusable centroid workspace produce a visible end-to-end search win
  on a heavier cached public slice

This trace checks that on `browsecomp_plus_gold_real_subset`, which is the
largest cached real slice in this workspace alongside `browsecomp_plus_real_subset`.

## Why this slice

I verified the cached real-slice sizes from
`.cache/kayak/public_real_slice_collection_storage.json`:

- `limit_small`: `46` docs, `7432` vectors
- `scifact_real_subset`: `55` docs, `9639` vectors
- `fiqa_real_subset`: `82` docs, `10159` vectors
- `browsecomp_plus_gold`: `90` docs, `15756` vectors
- `browsecomp_plus_real_subset`: `90` docs, `15756` vectors

That made `browsecomp_plus_gold` the sound next place to check whether the
stage-1 candidate-generation win survives a heavier end-to-end workload.

## Benchmark surface

Added:

- `benchmarks/profile_centroid_workspace_search_reuse.mojo`

The benchmark compares:

- `search_collection_for_plan(...)`
- `search_collection_for_plan_with_workspace(...)`

Measured plans:

- `centroid_postings_imputed`
- `centroid_postings_imputed_flat`

The benchmark verifies before timing that fresh and reused execution return the
same final hit identities for every query on the slice. The check compares hit
order plus `(segment_id, doc_id)` pairs.

Artifacts:

- `.cache/kayak/profile_centroid_workspace_search_reuse.tsv`
- `.cache/kayak/bench_quiet/20260414T195527Z`

## Commands run

```bash
pixi run mojo -I . benchmarks/profile_centroid_workspace_search_reuse.mojo
bash scripts/run_bench_quiet.sh --timeout-seconds 20 --force --repeats 1 -- pixi run mojo -I . benchmarks/profile_centroid_workspace_search_reuse.mojo
```

The direct run was used first as a compile-and-correctness check.

The quiet wrapper timed out waiting for a quiet host and force-ran the
benchmark. Pre-run competing load samples ranged from roughly `396.60` to
`536.10` aggregate `%CPU` from other processes, so the wrapped run remains
host-contended.

## Result

Wrapped means and reuse ratios (`reused / fresh`):

- `centroid_postings_imputed`
  - fresh: `0.0028279373907979036 s`
  - reused: `0.002857475294117647 s`
  - ratio: `1.010445`
- `centroid_postings_imputed_flat`
  - fresh: `0.0027905632352941173 s`
  - reused: `0.002757083116883117 s`
  - ratio: `0.988002`

Direct-run comparison for context:

- `centroid_postings_imputed`
  - direct ratio: `1.069624`
- `centroid_postings_imputed_flat`
  - direct ratio: `1.005721`

## Interpretation

What is verified:

- fresh and reused end-to-end execution return the same final hit identities on
  this slice for both measured imputed plans
- on the wrapped run, `centroid_postings_imputed` is slightly slower with
  workspace reuse
- on the same wrapped run, `centroid_postings_imputed_flat` is slightly faster
  with workspace reuse

What that means:

- the clear stage-1 candidate-generation win from the controlled scale
  benchmarks does **not** translate into a broad end-to-end latency win on this
  heavier public slice
- the end-to-end effect here is near parity and mixed in sign

The most likely reason is straightforward:

- stage-1 setup work got better
- but stage-2 exact late interaction and the rest of the search path still
  dominate enough of this slice that the saved candidate-generation work is too
  small to cleanly move total search time

## Bottom line

The sound combined reading across the three workspace traces is:

- controlled stage-1 candidate-generation benchmarks now support the reusable
  workspace on both exact and imputed centroid paths
- a heavier end-to-end public-slice check does **not** support claiming a broad
  whole-search speedup from workspace reuse alone

So the workspace change looks like a correct local optimization with measured
stage-1 benefit, not a headline end-to-end latency breakthrough by itself.
