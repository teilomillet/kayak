# Imputed Flat Small-Count Prefix Cap

## Claim

The `centroid_postings_imputed_flat` small-count path does not need to keep a
fully sorted `111–128` centroid list when finalization only consumes a much
shorter prefix.

The retained claim is deliberately narrow:

- it applies only to the flat imputed selectors
  - `centroid_selection_for_flat_query_token_generic`
  - `centroid_selection_for_flat_query_token_dim128`
- it applies only when `centroid_count <= bound`
- it preserves current selection semantics exactly
- it falls back to the old full-bound path if any centroid has a zero token
  count

I explicitly did **not** keep the same optimization on the nested imputed
selector. The paired wrapped measurements did not justify broadening the change
beyond the flat path the user asked to optimize.

## Why This Is Sound

`finalize_imputed_centroid_selection()` does two things with the sorted
centroids:

1. it scans the sorted prefix until cumulative centroid token count reaches
   `t_prime`
2. it truncates the returned shortlist to `nprobe`

That means the full sorted tail is unnecessary if we can prove the scan will
finish inside a smaller exact prefix.

The proof used here is:

- if every centroid has `centroid_token_count > 0`, then after the top
  `t_prime` sorted centroids the cumulative token count must be at least
  `t_prime`
- therefore the finalization scan can only need the top
  `max(nprobe, t_prime)` centroids
- if any centroid has `centroid_token_count <= 0`, that proof fails, so the
  code falls back to the old full-bound behavior

This is why the implementation computes:

- `selection_limit = max(nprobe, t_prime)` when all centroid token counts are
  positive
- `selection_limit = bound` otherwise

That changes the amount of exact shortlist maintenance, not the retrieval
policy.

## Rejected Variant

I first tried a different small-count kernel: append all centroid scores and
heap-sort the entire `<= 128` list at the end.

That version passed correctness, but the wrapped median comparison regressed on
the public flat slices, so it was removed instead of rationalized away.

The surviving change in this trace is the smaller exact-prefix cap above, not
the heap-sort rewrite.

## Implementation

Files:

- `kayak/planning/centroid_postings_imputed_stage.mojo`
- `kayak/planning/centroid_postings_imputed_flat_stage.mojo`
- `tests/test_imputed_high_centroid_probe.mojo`
- `benchmarks/profile_imputed_selection_dim128_real_subset.mojo`

What changed:

- added `small_count_imputed_centroid_selection_limit(...)`
  - computes `max(nprobe, t_prime)`
  - falls back to `bound` if any centroid token count is non-positive
- applied that limit only to the flat small-count selector branches
- added a focused real-subset benchmark for the dim=128 flat/nested imputed
  selection kernels with explicit:
  - `centroid_count`
  - `final_k`
  - `query_count`
  - `total_query_vectors`
  - `max_query_vectors`
- added a zero-token small-count regression test so the fallback is checked
  mechanically

## Validation

Commands run:

```bash
pixi run mojo -I . tests/test_imputed_high_centroid_probe.mojo
pixi run mojo -I . tests/test_collection_search_plan.mojo
pixi run mojo -I . tests/test_service_runtime.mojo
```

Observed results:

- `tests/test_imputed_high_centroid_probe.mojo`: `11/11` passed
- `tests/test_collection_search_plan.mojo`: `38/38` passed
- `tests/test_service_runtime.mojo`: `26/26` passed

The new targeted regression covers the edge the proof depends on:

- flat small-count path with all-positive token counts still matches the nested
  selector
- flat small-count path with many zero-token centroids falls back to the nested
  full-bound result

## Wrapped Evidence

Benchmark command:

```bash
bash scripts/run_bench_quiet.sh --timeout-seconds 20 --force --repeats 3 -- \
  pixi run mojo -I . benchmarks/profile_imputed_selection_dim128_real_subset.mojo
```

Paired logs used for the final decision:

- baseline `fa0d9e4` worktree:
  `/tmp/kayak-smallcount-baseline/.cache/kayak/bench_quiet/20260415T050522Z`
- current worktree:
  `.cache/kayak/bench_quiet/20260415T050650Z`

Wrapped median deltas:

- `SciFact` flat:
  - before: `0.000364833333333 s`
  - after: `0.000352666666667 s`
  - delta: `-3.33%`
- `FIQA` flat:
  - before: `0.000370500000000 s`
  - after: `0.000365166666667 s`
  - delta: `-1.44%`
- `LIMIT-small` flat:
  - before: `0.000295343750000 s`
  - after: `0.000294812500000 s`
  - delta: `-0.18%`

Control lines from the unchanged nested path:

- `SciFact` nested: `-0.51%`
- `FIQA` nested: `+1.36%`
- `LIMIT-small` nested: `-1.56%`

Interpretation:

- the flat win is clearly above the apparent noise floor on `SciFact`
- `FIQA` is directionally positive but near the observed host-noise range
- `LIMIT-small` is effectively neutral
- this is a modest optimization, not a broad frontier shift

## Conclusion

This slice is worth keeping because it is:

- semantically justified by the existing finalization contract
- guarded by an explicit zero-token fallback
- validated end-to-end
- modestly beneficial on the targeted flat small-count public shape

What it does **not** justify:

- claiming a large public benchmark jump
- changing the nested imputed selector at the same time
