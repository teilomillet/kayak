# Imputed Selector Shortlist Reuse on Wrapped Benchmarks

## Claim

Reusing the imputed selector's `bound` shortlist in place and moving each
selection into the per-query cache removes avoidable list allocation and copy
work from stage-1 without changing retrieval policy or observed behavior.

The verified claim is narrow:

- wrapped imputed selector microbenchmarks improve across three public subsets
- wrapped BrowseComp-Plus Gold stage-1 and fresh-search timings improve on both
  imputed centroid plans
- correctness remains unchanged on the targeted selector, plan, and service
  suites

I am **not** attributing the modest `stage2_from_candidates` movement in the
wrapped stage breakdown to this slice, because the code change does not target
exact rerank.

## Why This Was the Next Change

After the earlier workspace and exact-rerank work, the remaining measured
stage-1 hotspot was imputed centroid selection itself:

- wrapped baseline primitive source:
  `.cache/kayak/bench_quiet/20260414T224951Z/run_1.txt`
- exploratory rerun before this refactor again showed imputed selection far
  above accumulation on all three public subsets

Direct code inspection showed two avoidable costs that were still explicit in
the current implementation:

1. both imputed selectors built a `bound`-sized sorted shortlist and then
   allocated a second `nprobe`-sized shortlist just to return the prefix
2. both imputed stage wrappers deep-copied each `ScoredCentroidSelection` into
   the cached per-query list even though the local temporary was dead after
   baseline accumulation

That work is pure constant-factor overhead. It does not change policy, and it
does not hide a new retrieval rule inside the engine.

## What Changed

Files:

- [kayak/planning/centroid_postings_imputed_stage.mojo](../../kayak/planning/centroid_postings_imputed_stage.mojo)
- [kayak/planning/centroid_postings_imputed_flat_stage.mojo](../../kayak/planning/centroid_postings_imputed_flat_stage.mojo)
- [kayak/planning/centroid_postings_stage.mojo](../../kayak/planning/centroid_postings_stage.mojo)
- [tests/test_imputed_high_centroid_probe.mojo](../../tests/test_imputed_high_centroid_probe.mojo)

Implementation choices and reasons:

- Added `finalize_imputed_centroid_selection(...)` in
  `centroid_postings_imputed_stage.mojo`.
  Reason: the nested and flat imputed selectors shared the same "compute
  baseline from the full sorted shortlist, then keep only the top `nprobe`"
  finalization logic.
- The new helper computes baseline correction from the full sorted shortlist,
  then trims the same lists down to `nprobe` with `pop()`.
  Reason: this keeps the previous contract exactly while removing a second
  per-token allocation and append loop.
- Both stage wrappers now append each selection with `selection^` instead of
  `selection.copy()`.
  Reason: after `base_score += selection.baseline_correction`, the local
  temporary is no longer needed, so moving it avoids copying the contained
  lists.
- Kept the earlier O(1) tail-reject guard in
  `insert_descending_centroid_match(...)`.
  Reason: it is semantically free and still reduces loser scans once the
  shortlist is full, but the final performance claim below does not rely on
  that guard alone.

## Validation

Correctness commands:

- `pixi run mojo -I . tests/test_imputed_high_centroid_probe.mojo`
  - `4/4` passed
- `pixi run mojo -I . tests/test_collection_search_plan.mojo`
  - `38/38` passed
- `pixi run mojo -I . tests/test_service_runtime.mojo`
  - `26/26` passed

The targeted selector regressions now include:

- nested imputed selector vs a local reference implementation
- flat generic imputed selector vs the nested selector on the same token

## Wrapped Primitive Evidence

Sources:

- baseline: `.cache/kayak/bench_quiet/20260414T224951Z/run_1.txt`
- current: `.cache/kayak/bench_quiet/20260414T235004Z/run_1.txt`

Wrapped `profile_centroid_primitives_real_subset.mojo` results:

| dataset | nested before (s) | nested after (s) | nested delta | flat before (s) | flat after (s) | flat delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact | `4.044710081585082e-04` | `3.7030685121107264e-04` | `-34.164 us` (`-8.45%`) | `3.9224271468680415e-04` | `3.630037420709168e-04` | `-29.239 us` (`-7.45%`) |
| FIQA | `3.935989132031135e-04` | `3.762794468065224e-04` | `-17.319 us` (`-4.40%`) | `3.876162099463535e-04` | `3.690624221453287e-04` | `-18.554 us` (`-4.79%`) |
| LIMIT-small | `3.338294409937888e-04` | `3.0832055944055947e-04` | `-25.509 us` (`-7.64%`) | `3.290986677597868e-04` | `3.0269230465666927e-04` | `-26.406 us` (`-8.02%`) |

What this verifies:

- the selector hotspot itself moved down under the wrapped harness
- the improvement is consistent across three public subsets
- accumulation did not become the new dominant cost; it stayed in the same
  range as before

## Wrapped Stage-Level Evidence

Sources:

- baseline: `.cache/kayak/bench_quiet/20260414T221505Z/run_1.txt`
- current: `.cache/kayak/bench_quiet/20260415T000838Z/run_1.txt`

Wrapped `profile_centroid_workspace_search_breakdown.mojo` results on
BrowseComp-Plus Gold:

| plan | metric | before (s) | after (s) | delta |
| --- | --- | ---: | ---: | ---: |
| `centroid_postings_imputed` | `stage1 fresh` | `4.4192139232127097e-04` | `4.204602207643445e-04` | `-21.461 us` (`-4.86%`) |
| `centroid_postings_imputed` | `stage1 reused` | `4.44796556851979e-04` | `4.1843245374264084e-04` | `-26.364 us` (`-5.93%`) |
| `centroid_postings_imputed` | `search fresh` | `1.0415690058479532e-03` | `9.796333948139306e-04` | `-61.936 us` (`-5.95%`) |
| `centroid_postings_imputed` | `search reused` | `1.0147310588235294e-03` | `9.769426966292135e-04` | `-37.788 us` (`-3.72%`) |
| `centroid_postings_imputed_flat` | `stage1 fresh` | `4.275133426966292e-04` | `4.1438962506420133e-04` | `-13.124 us` (`-3.07%`) |
| `centroid_postings_imputed_flat` | `stage1 reused` | `4.3580412705272254e-04` | `4.11833421386306e-04` | `-23.971 us` (`-5.50%`) |
| `centroid_postings_imputed_flat` | `search fresh` | `1.000669799426934e-03` | `9.74779042253521e-04` | `-25.891 us` (`-2.59%`) |
| `centroid_postings_imputed_flat` | `search reused` | `9.963423121387284e-04` | `9.71864649859944e-04` | `-24.478 us` (`-2.46%`) |

What this verifies:

- the selector win survives into the explicit plan benchmark
- the biggest stable effect is still in stage-1, which matches the code change
- end-to-end fresh search improves on both imputed centroid plans, though the
  total win is naturally smaller than the selector-only microbenchmark because
  stage 2 still remains a large part of whole-search time

## Conclusion

This slice is worth keeping.

The evidence supports the original systems claim:

- keep retrieval policy explicit
- remove avoidable kernel work underneath it

Here, that meant:

- no new planner policy
- no hidden pruning rule
- no behavior change in tests
- fewer list allocations and fewer list copies on the imputed selector hot path

The next justified target remains the same general area:

- keep reducing imputed selector/container overhead with shape-specialized
  kernels where the measurements warrant it
- avoid claiming algorithmic wins where we only earned constant-factor ones
