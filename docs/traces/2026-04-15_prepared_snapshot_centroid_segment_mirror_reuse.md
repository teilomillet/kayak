# Prepared Snapshot Centroid Segment-Mirror Reuse

## Claim

After adding the explicit prepared-snapshot mirror opt-in, the next question
was no longer "is the stage-2 kernel slightly faster in isolation." The real
serving question was:

- if one published snapshot is reused for many centroid queries on the same
  host, does the mirror opt-in improve prepared-snapshot latency or throughput
  enough to justify broader use

This trace records the new serving-level benchmark seam, the added batch
correctness coverage, and the measured outcome.

## Why this was the right next test

The earlier stage-2 work already established two things:

- building segment mirrors per query is wrong because the build cost is too
  large
- the only plausible production seam is the explicit prepared-snapshot path,
  where one loaded snapshot survives across many requests

That means the next sound benchmark is:

- same snapshot
- explicit centroid plans
- repeated direct prepared search
- repeated batch search on the same prepared snapshot
- compare baseline prepared search with the explicit mirror opt-in

## Implementation

Added focused batch correctness coverage:

- [test_service_prepared_exact_search_batch_segment_mirror.mojo](/Users/teilomillet/Code/kayak/tests/test_service_prepared_exact_search_batch_segment_mirror.mojo:1)

This test:

- builds a dim128 one-segment service fixture with centroid postings available
- prepares both baseline and mirror-loaded prepared snapshots
- checks that serial mirrored search matches serial baseline search for:
  - `centroid_postings_imputed`
  - `centroid_postings_imputed_flat`
- checks that batch execution on the mirrored prepared snapshot matches the
  same serial baseline under:
  - default scoring config
  - `parallel_work_item_count_override = 2`

Added the serving-level benchmark:

- [profile_prepared_snapshot_centroid_segment_mirror_reuse_browsecomp_gold.mojo](/Users/teilomillet/Code/kayak/benchmarks/profile_prepared_snapshot_centroid_segment_mirror_reuse_browsecomp_gold.mojo:1)

This benchmark:

- mirrors the BrowseComp-Plus gold slice into one service collection
- uses explicit centroid plans, not planner inference
- measures both:
  - `centroid_postings_imputed`
  - `centroid_postings_imputed_flat`
- measures:
  - prepared snapshot load without mirrors
  - prepared snapshot load with mirrors
  - repeated direct prepared search
  - repeated batch search with worker counts `1`, `2`, `4`, and `8`
- validates result equality before timing:
  - mirrored direct search vs baseline direct search
  - batch search vs serial search on the same prepared snapshot

Reason:

- this keeps correctness checks in the same executable seam as the timing
- it measures the only lifecycle where mirror amortization could matter

## Validation

Commands run:

```bash
pixi run mojo -I . tests/test_service_prepared_exact_search_batch_segment_mirror.mojo
pixi run mojo -I . tests/test_service_prepared_snapshot_runtime.mojo
pixi run mojo -I . tests/test_service_prepared_exact_search_batch.mojo
pixi run mojo -I . tests/test_service_prepared_exact_search_executor.mojo
pixi run mojo -I . benchmarks/profile_prepared_snapshot_centroid_segment_mirror_reuse_browsecomp_gold.mojo
bash scripts/run_bench_quiet.sh --timeout-seconds 30 --force --repeats 2 -- \
  pixi run mojo -I . benchmarks/profile_prepared_snapshot_centroid_segment_mirror_reuse_browsecomp_gold.mojo
```

Observed correctness results:

- `tests/test_service_prepared_exact_search_batch_segment_mirror.mojo`: `2/2`
  passed
- `tests/test_service_prepared_snapshot_runtime.mojo`: `2/2` passed
- `tests/test_service_prepared_exact_search_batch.mojo`: `2/2` passed
- `tests/test_service_prepared_exact_search_executor.mojo`: `2/2` passed

What is therefore verified:

- the mirrored prepared-snapshot path preserves search results for the measured
  centroid plans
- the batch kernel preserves those results as well

## Benchmark Setup

Benchmark:

- [profile_prepared_snapshot_centroid_segment_mirror_reuse_browsecomp_gold.mojo](/Users/teilomillet/Code/kayak/benchmarks/profile_prepared_snapshot_centroid_segment_mirror_reuse_browsecomp_gold.mojo:1)

Quiet-wrapper artifact directory:

- `.cache/kayak/bench_quiet/20260415T180435Z`

Measured slice:

- dataset: `Tevatron/browsecomp-plus/gold-slice`
- slice: `browsecomp_plus_gold_slice`
- documents: `90`
- distinct queries: `4`
- request pool: `32`
- nominal query vectors: `32`
- nominal document vectors: `175`
- vector dim: `128`
- host `parallelism_level`: `8`

## Exploratory Direct Run

The first direct benchmark run, before the quiet wrapper, suggested a modest
reuse win:

- prepared snapshot load:
  - baseline: `0.0075749 s`
  - with mirrors: `0.008464894736842105 s`
  - extra load cost: about `0.89 ms` (`+11.75%`)
- `centroid_postings_imputed` direct prepared search:
  - baseline: `0.0007194969325153374 s`
  - with mirrors: `0.0007073431952662722 s`
  - about `1.69%` faster
- `centroid_postings_imputed_flat` direct prepared search:
  - baseline: `0.0005632512562814071 s`
  - with mirrors: `0.0005589627906976744 s`
  - about `0.76%` faster

The same exploratory run also suggested small batch wins on most worker counts:

- imputed:
  - `w1`: about `0.91%` faster
  - `w2`: about `2.11%` faster
  - `w4`: about `4.77%` faster
  - `w8`: about `1.54%` faster
- imputed-flat:
  - `w1`: about `1.73%` faster
  - `w2`: about `3.57%` faster
  - `w4`: about `0.61%` slower
  - `w8`: about `1.22%` faster

That made the reuse hypothesis worth repeating, but not yet decision-grade.

## Quiet-Wrapper Reruns

The quiet wrapper never reached the requested quiet threshold and had to run
with `--force` on a heavily contended host. That matters for interpretation.

Quiet-wrapper run 1:

- mirror load cost remained similar:
  - baseline prepare: `0.0074492 s`
  - mirror prepare: `0.008398631578947368 s`
- some rows favored mirrors:
  - imputed `w1`: `0.022631363636363637 -> 0.022452 s`
  - imputed `w2`: `0.0122545 -> 0.012121076923076922 s`
  - imputed `w8`: `0.006854684210526316 -> 0.006819736842105263 s`
  - imputed-flat direct: `0.0005616826923076923 -> 0.0005521805555555555 s`
  - imputed-flat `w8`: `0.006471157894736843 -> 0.006289631578947369 s`
- some rows regressed:
  - imputed direct: `0.0007107791411042945 -> 0.0007137869822485207 s`
  - imputed `w4`: `0.0082528125 -> 0.0084483125 s`
  - imputed-flat `w2`: `0.011254 -> 0.011490357142857144 s`

Quiet-wrapper run 2 was noisier and more contradictory:

- mirror load cost again remained similar:
  - baseline prepare: `0.0074846999999999995 s`
  - mirror prepare: `0.008446473684210526 s`
- some rows still favored mirrors:
  - imputed direct: `0.0007132638036809816 -> 0.0007084082840236686 s`
  - imputed `w1`: `0.022668545454545455 -> 0.022378363636363637 s`
  - imputed `w4`: `0.0084138125 -> 0.0082805 s`
  - imputed-flat `w2`: `0.011934307692307692 -> 0.010952785714285715 s`
  - imputed-flat `w4`: `0.007934882352941177 -> 0.007548411764705882 s`
- but other rows regressed materially:
  - imputed `w8`: `0.007339526315789473 -> 0.010794400000000001 s`
  - imputed-flat direct:
    `0.0006178397790055248 -> 0.0006693377483443708 s`
  - imputed-flat `w1`: `0.020052818181818182 -> 0.02243209090909091 s`
  - imputed-flat `w8`: `0.006499842105263158 -> 0.006803526315789473 s`

## Interpretation

What the new serving-level evidence supports:

- the mirror opt-in is correct at the prepared-snapshot and batch-execution
  boundaries
- same-snapshot reuse is the right lifecycle to measure
- the steady extra prepare cost is real and small, roughly `0.95 ms`
- the potential serving-level win is also small, usually in the low
  single-digit-percent range when it appears

What the new serving-level evidence does **not** support:

- enabling the mirror path by default
- claiming a stable throughput improvement across worker counts
- naming a trustworthy break-even query count on this host

Reason:

- the measured win is small enough that host contention can erase it
- the forced quiet-wrapper reruns are materially mixed
- at least one high-worker row became a clear regression

## Conclusion

The sound current conclusion is:

- keep the centroid segment-mirror path as an explicit prepared-snapshot opt-in
- do not widen it into a default serving policy
- treat it as an available specialized kernel whose value still depends on the
  host, contention, and actual request mix

This is still useful progress because the repo now has:

- a real correctness-checked batch path for mirrored centroid prepared search
- a serving-level benchmark seam for repeated same-snapshot centroid queries
- concrete evidence that the effect exists but is too small and unstable, on
  this host, to justify default promotion
