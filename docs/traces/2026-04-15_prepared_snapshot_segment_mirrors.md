# Prepared Snapshot Segment Mirrors

## Claim

The measured stage-2 mirror experiment suggested one narrow production seam was
worth testing:

- build per-segment dim128 flat mirrors once
- reuse them only from an explicit prepared snapshot
- keep the default search/runtime path unchanged

Reason:

- the mirror build cost is substantial enough that per-query construction is
  obviously wrong
- only the same-snapshot multi-query case can plausibly amortize that cost
- the repo should keep this as an explicit opt-in seam, not a hidden policy

## Implementation

Added a new stage-2 helper module:

- [exact_stage_segment_mirror.mojo](/Users/teilomillet/Code/kayak/kayak/planning/exact_stage_segment_mirror.mojo:1)

It owns:

- dim128 mirror construction from loaded snapshot segments
- the tiled dim128 mirror scorer
- a stage-2 exact-rerank helper over prebuilt mirrors
- a narrow support predicate for the currently measured regime:
  - `ExactCpuBackend` dim128 fast path enabled
  - `query.vector_dim == 128`
  - `query.vector_count == 32`
  - `centroid_postings_imputed` or `centroid_postings_imputed_flat`
  - `stage2_reference_operator.kind == "exact_late_interaction"`

Integrated that helper into the prepared-snapshot seam:

- [prepared_snapshot_runtime.mojo](/Users/teilomillet/Code/kayak/kayak/service/prepared_snapshot_runtime.mojo:1)

Changes:

- `PreparedSearchSnapshot` now carries an explicit opt-in mirror payload
- `prepare_collection_search_snapshot(...)` and
  `prepare_service_search_snapshot(...)` accept
  `load_dim128_segment_mirrors: Bool = False`
- non-dim128 collections reject that opt-in at prepare time instead of
  silently doing nothing
- `ExactCpuBackend` prepared-search overloads use the mirror path only when the
  snapshot explicitly loaded mirrors and the request matches the measured
  centroid regime

The default remains unchanged:

- if callers do not request mirrors, behavior is identical to the old
  prepared-snapshot runtime
- even when mirrors are loaded, non-measured request shapes fall back to the
  existing exact stage

Also updated the benchmark seam to call the production helper rather than a
benchmark-only copy:

- [profile_centroid_exact_stage_segment_mirror_compare_browsecomp_gold.mojo](/Users/teilomillet/Code/kayak/benchmarks/profile_centroid_exact_stage_segment_mirror_compare_browsecomp_gold.mojo:1)

Added focused tests:

- [test_prepared_snapshot_segment_mirror.mojo](/Users/teilomillet/Code/kayak/tests/test_prepared_snapshot_segment_mirror.mojo:1)

## Validation

Commands run:

```bash
pixi run mojo -I . tests/test_prepared_snapshot_segment_mirror.mojo
pixi run mojo -I . tests/test_service_prepared_snapshot_runtime.mojo
pixi run mojo -I . tests/test_service_prepared_exact_search_executor.mojo
pixi run mojo -I . benchmarks/profile_centroid_exact_stage_segment_mirror_compare_browsecomp_gold.mojo
bash scripts/run_bench_quiet.sh --timeout-seconds 30 --force --repeats 2 -- pixi run mojo -I . benchmarks/profile_centroid_exact_stage_segment_mirror_compare_browsecomp_gold.mojo
```

Observed correctness results:

- `tests/test_prepared_snapshot_segment_mirror.mojo`: `2/2` passed
- `tests/test_service_prepared_snapshot_runtime.mojo`: `2/2` passed
- `tests/test_service_prepared_exact_search_executor.mojo`: `2/2` passed

What the new focused tests verify:

- prepared snapshots with loaded mirrors return identical hits and scores to the
  baseline prepared runtime for both imputed centroid plans
- non-dim128 collections reject the mirror opt-in explicitly

## Performance Evidence

The benchmark compares:

- current resolved-window scorer
- prebuilt mirror scorer
- build cost for segment mirrors
- full exact rerank with and without the prebuilt mirror path

Single direct run on BrowseComp-Plus gold shortlist:

- `centroid_postings_imputed`
  - `exact_rerank_current`: `0.0002867257142857143 s`
  - `exact_rerank_prebuilt_segment_mirror_reference`: `0.00028426704545454545 s`
- `centroid_postings_imputed_flat`
  - `exact_rerank_current`: `0.00028038547486033516 s`
  - `exact_rerank_prebuilt_segment_mirror_reference`: `0.00028304519774011296 s`

Repeated quiet-wrapper runs were still taken under high host load because the
machine never reached the requested quiet threshold and `--force` was required.
That means the runs are useful evidence, but not clean enough to justify a
strong speedup claim.

Quiet-wrapper run 1:

- `centroid_postings_imputed`
  - current exact rerank: `0.0002835480225988701 s`
  - mirror exact rerank: `0.00028361581920903955 s`
- `centroid_postings_imputed_flat`
  - current exact rerank: `0.00028497727272727274 s`
  - mirror exact rerank: `0.00027996648044692734 s`

Quiet-wrapper run 2:

- `centroid_postings_imputed`
  - current exact rerank: `0.00028423863636363634 s`
  - mirror exact rerank: `0.0002863314285714286 s`
- `centroid_postings_imputed_flat`
  - current exact rerank: `0.00028638285714285714 s`
  - mirror exact rerank: `0.00028599428571428575 s`

Mirror build cost remained roughly stable:

- imputed mirror build: about `0.736` to `0.762 ms`
- imputed-flat mirror build: about `0.751` to `0.755 ms`

## What Is Verified

Verified:

- the repo now has a real prepared-snapshot-only mirror kernel
- the optimization stays explicit and opt-in instead of altering generic search
  behavior
- correctness matches the baseline prepared runtime on the exercised centroid
  plans
- the production benchmark seam now measures the shipped helper rather than a
  benchmark-only copy

Not verified:

- a consistent stage-2 latency win large enough to justify enabling mirrors by
  default
- a trustworthy break-even query count on this host under low-noise conditions

## Conclusion

The sound current conclusion is:

- keep the mirror path available only as an explicit prepared-snapshot opt-in
- do not promote it to the default runtime path
- treat the current performance outcome as mixed / inconclusive rather than a
  proven win

That still leaves the change worthwhile:

- the module boundary is now real and benchmarkable
- future low-noise measurements can decide whether this path earns broader use
- the default engine behavior remains unchanged until the evidence is stronger
