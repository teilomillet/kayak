# 2026-04-15: `centroid_postings_imputed_flat` exact small-count prefix limit

## Claim

The small-count `centroid_postings_imputed_flat` path was still doing more
sorted shortlist maintenance than `finalize_imputed_centroid_selection()`
actually needs on the public small-window slices.

The retained claim is narrow:

- only the flat imputed stage gets the tighter bound
- the new bound is exact, not heuristic
- it only matters on the production stage path where the bound can be computed
  once and reused across all query tokens
- retrieval behavior must stay unchanged

## Why this target

The last planner step left `centroid_postings_imputed_flat` as the preferred
small-window native path. The next useful optimization therefore had to target
that exact regime rather than another planner reorder.

Direct inspection showed the remaining waste:

- `small_count_imputed_centroid_selection_limit()` used a worst-case
  `max(nprobe, t')` prefix bound
- on the public small-count slices that still kept the full `111–128` centroid
  prefix, because `t'` is much larger than `nprobe`
- but the actual centroid token counts are not all `1`, so the baseline walk
  can often be proven to finish much earlier

The sound next question was:

- can we derive a tighter exact prefix bound from the actual centroid token
  counts
- compute it once per segment
- and reuse it across all query tokens in the flat imputed stage

## Design

Files:

- [kayak/planning/centroid_postings_imputed_stage.mojo](../../kayak/planning/centroid_postings_imputed_stage.mojo)
- [kayak/planning/centroid_postings_imputed_flat_stage.mojo](../../kayak/planning/centroid_postings_imputed_flat_stage.mojo)
- [tests/test_imputed_high_centroid_probe.mojo](../../tests/test_imputed_high_centroid_probe.mojo)

What changed:

- kept the existing `small_count_imputed_centroid_selection_limit(...)`
  conservative helper for direct selector entry points
- added
  `exact_small_count_imputed_centroid_selection_limit(...)`
  for the production flat stage path
- that exact helper sorts centroid token counts ascending once, then finds the
  shortest prefix whose cumulative sum is guaranteed to cross `t'` while still
  respecting `nprobe`
- the flat stage now computes that exact prefix limit once per segment and
  passes it into the per-token selector helper

Why this is exact:

- the smallest centroid token counts define the worst-case score-ordered prefix
  for the `t'` walk
- if their cumulative sum crosses `t'` after `p` entries, then every
  score-ordered prefix of length `p` must also cross `t'`
- if any centroid has `token_count <= 0`, the proof fails, so the code falls
  back to the old full-bound behavior

Why the exact helper is not used everywhere:

- computing the tighter prefix needs sorting the centroid token counts
- that is worth doing once per segment and reusing across all query tokens
- it is not worth doing inside the direct per-token selector entry points,
  because those APIs do not amortize the setup cost

## Validation

Correctness:

- `pixi run bash -lc 'mojo -I . tests/test_imputed_high_centroid_probe.mojo'`
- `pixi run bash -lc 'mojo -I . tests/test_collection_search_plan.mojo'`
- `pixi run bash -lc 'mojo -I . tests/test_service_runtime.mojo'`

Observed results:

- `tests/test_imputed_high_centroid_probe.mojo`: `13/13` passed
- `tests/test_collection_search_plan.mojo`: `38/38` passed
- `tests/test_service_runtime.mojo`: `26/26` passed

New guardrails added:

- token-rich small-count fixture proves the exact helper can reduce the prefix
  below the full bound
- the flat selector still matches the nested selector on that token-rich slice
- the zero-token fallback still preserves the old behavior

## Measured outcome

### Exact helper proof point

The new token-rich small-count regression fixture uses:

- `centroid_count = 128`
- `token_weight = 8`
- `final_k = 10`

Measured invariants from the test:

- `bound = 128`
- `nprobe = 32`
- `t' = 90`
- conservative helper result: `90`
- exact helper result: `32`

Interpretation:

- the production flat stage can keep only the top `32` centroids in this
  regime and still preserve the exact same baseline-correction walk

### Selector microbenchmark

Source comparison:

- before: `/tmp/profile_imputed_selection_dim128_real_subset_before.tsv`
- after: `.cache/kayak/profile_imputed_selection_dim128_real_subset.tsv`

The direct selector benchmark is exploratory only for this slice, because it
still exercises the conservative per-token helper rather than the new hoisted
production path.

Even so, the final run stayed in the expected range:

- `SciFact`: `0.000425 s -> 0.000439 s` (`+3.29%`)
- `FIQA`: `0.00039433333333333334 s -> 0.0004091666666666667 s` (`+3.76%`)
- `LIMIT-small`: `0.0003565625 s -> 0.0003196875 s` (`-10.34%`)

Interpretation:

- the selector-only entry point is not the right decision artifact here
- the real claim is about the hoisted stage path below

### Public small-window frontier

Source comparison:

- before:
  `/tmp/public_small_window_frontier_before_imputed_limit.json`
- after:
  `.cache/kayak/public_small_window_frontier.json`

The post-change frontier kept `nDCG` and candidate recall identical on every
`centroid_postings_imputed_flat` row while reducing search latency on almost
every small-window slice:

- `scifact_real_subset`
  - `k = 10`: `0.0011558333333333334 -> 0.0009126666666666667`
    (`-21.04%`)
  - `k = 20`: `0.0012266666666666667 -> 0.0009835`
    (`-19.82%`)
  - `k = 40`: `0.0020203333333333336 -> 0.0011271666666666667`
    (`-44.21%`)
- `fiqa_real_subset`
  - `k = 10`: `0.0011675 -> 0.0009851666666666665` (`-15.62%`)
  - `k = 20`: `0.0012146666666666666 -> 0.0008878333333333334`
    (`-26.91%`)
  - `k = 40`: `0.0017448333333333333 -> 0.0009186666666666667`
    (`-47.35%`)
- `limit_small_real_subset`
  - `k = 10`: `0.00092678125 -> 0.000923375` (`-0.37%`)
  - `k = 20`: `0.00115034375 -> 0.00092846875` (`-19.29%`)
  - `k = 40`: `0.0015706875 -> 0.00112590625` (`-28.32%`)
- `browsecomp_plus_evidence_slice`
  - `k = 10`: `0.001099 -> 0.0011025` (`+0.32%`)
  - `k = 20`: `0.00125575 -> 0.0010535` (`-16.11%`)
  - `k = 40`: `0.0020075 -> 0.00118` (`-41.22%`)
- `browsecomp_plus_gold_slice`
  - `k = 10`: `0.001371 -> 0.00100225` (`-26.90%`)
  - `k = 20`: `0.00126275 -> 0.001046` (`-17.16%`)
  - `k = 40`: `0.0016645 -> 0.00119175` (`-28.40%`)

Behavior stayed identical on every one of those rows:

- `nDCG` unchanged
- candidate recall unchanged

I reran the frontier twice after the change.
The second post-change run stayed directionally consistent with the first:

- some rows moved a few percent run to run, which is normal for raw timing
- the large improvements at `k = 20` and `k = 40` persisted across datasets

## Conclusion

Keep the exact small-count prefix optimization.

Why this is justified:

- the proof is exact and guarded by a zero-token fallback
- the production flat stage amortizes the tighter bound across all query tokens
- all semantic and runtime suites still pass
- the public small-window frontier keeps quality / recall identical and shows
  substantial search-latency wins on the planner’s preferred small-window
  imputed-flat path

What this does **not** justify:

- claiming the direct selector API itself is universally faster
- changing the nested imputed selector at the same time
- generalizing the win to wide-window `flat`, which was already handled by the
  planner crossover step

The next justified target is still inside the imputed family:

- profile whether the remaining small-window cost now sits in exact rerank or
  in the imputed nested selector path
- then only optimize the next hotspot that survives measurement
