# Document Proxy Stage-2 Persistent Flat Experiment

## Claim

The next stage-2 question after the current exact-stage `tiled4` rollout was:

- on the real resolved-candidate-window seam, does a persistent flat dim128
  sidecar justify production integration for exact stage-2 scoring

The concrete experiment here was:

- keep the existing planner and exact-stage contracts unchanged
- add a benchmark-only scorer that reads `HybridFlatDim128Index.token_values`
  directly on the same BrowseComp-Plus gold `document_proxy` workload
- compare it against the current production stage-2 scorer, both with and
  without flat-query build cost
- only productionize the sidecar if the win is clear enough to justify the
  extra index and snapshot complexity

## Why this was worth testing

Code inspection showed that the current exact stage already has most of the
high-value Mojo-only improvements:

- [kayak/planning/exact_stage.mojo](/Users/teilomillet/Code/kayak/kayak/planning/exact_stage.mojo:414)
  already routes measured dim128 shapes through a flat-query `tiled4` scorer
- [kayak/scoring/maxsim.mojo](/Users/teilomillet/Code/kayak/kayak/scoring/maxsim.mojo:241)
  already uses the current `tiled4` SIMD microkernel for the nested
  `PackedIndex` layout

So the persistent-flat idea was narrower:

- remove `List[List[VectorScalar]]` traffic from the document-token hot path
- replace per-token `unsafe_ptr()` loads with contiguous flat-token addressing
- measure whether that storage/layout change alone is big enough to matter

The existing flat sidecar format already exists in:

- [kayak/index/hybrid_flat_dim128.mojo](/Users/teilomillet/Code/kayak/kayak/index/hybrid_flat_dim128.mojo:12)

## Benchmark seam

Added benchmark:

- [benchmarks/profile_document_proxy_stage2_persistent_flat_compare_browsecomp_gold.mojo](/Users/teilomillet/Code/kayak/benchmarks/profile_document_proxy_stage2_persistent_flat_compare_browsecomp_gold.mojo:1)

It validates exact score equivalence for:

- current production scorer
- persistent-flat query-first scorer
- persistent-flat `tiled4` scorer

on the real BrowseComp-Plus gold resolved candidate windows.

Measured operating point:

- dataset: `BrowseComp-Plus Gold`
- slice: `browsecomp_plus_gold_slice`
- query vectors: `32`
- document vectors: `175`
- vector dim: `128`
- `candidate_k = 40`

## Important benchmark fix

The first draft used open-ended `std.benchmark.run[...]()` calls and was
terminated before writing results. That was not decision-quality behavior.

The benchmark was then corrected to use explicit bounded settings:

- stage-2 sections: `min_runtime_secs=0.05`, `max_runtime_secs=0.25`,
  `max_iters=128`
- sidecar build/load sections: `min_runtime_secs=0.01`,
  `max_runtime_secs=0.10`, `max_iters=20`

Reason:

- this repo requires measurable, reproducible runs
- bounded timings are preferable to a single open-ended sample that can be
  interrupted or distorted by host load

## Bounded run 1

Artifact:

- `.cache/kayak/profile_document_proxy_stage2_persistent_flat_compare_browsecomp_gold.run1.tsv`

Measured means:

- current scorer:
  `0.00038125 s`
- `build_flat_query_dim128`:
  `2.5639710784062356e-06 s`
- `build_hybrid_flat_dim128_index`:
  `0.0009019 s`
- `load_stored_hybrid_flat_dim128_index`:
  `0.0019735 s`
- prebuilt persistent-flat query-first:
  `0.0006899375 s`
- `build_flat_query + persistent-flat query-first`:
  `0.0007238828125 s`
- prebuilt persistent-flat `tiled4`:
  `0.00036381884057971013 s`
- `build_flat_query + persistent-flat tiled4`:
  `0.0003771127819548872 s`

Ratios versus current:

- prebuilt persistent-flat query-first: `1.8097x`
- build-inclusive persistent-flat query-first: `1.8987x`
- prebuilt persistent-flat `tiled4`: `0.9543x`
- build-inclusive persistent-flat `tiled4`: `0.9891x`

Interpretation:

- the plain query-first persistent-flat path clearly loses
- the persistent-flat `tiled4` path wins only narrowly

## Bounded run 2

Artifact:

- `.cache/kayak/profile_document_proxy_stage2_persistent_flat_compare_browsecomp_gold.tsv`

Measured means:

- current scorer:
  `0.0002980833333333333 s`
- `build_flat_query_dim128`:
  `2.1444930519814718e-06 s`
- `build_hybrid_flat_dim128_index`:
  `0.00081555 s`
- `load_stored_hybrid_flat_dim128_index`:
  `0.0018272 s`
- prebuilt persistent-flat query-first:
  `0.000578671875 s`
- `build_flat_query + persistent-flat query-first`:
  `0.0005853984375 s`
- prebuilt persistent-flat `tiled4`:
  `0.0002856931818181818 s`
- `build_flat_query + persistent-flat tiled4`:
  `0.0002912267441860465 s`

Ratios versus current:

- prebuilt persistent-flat query-first: `1.9413x`
- build-inclusive persistent-flat query-first: `1.9639x`
- prebuilt persistent-flat `tiled4`: `0.9584x`
- build-inclusive persistent-flat `tiled4`: `0.9770x`

Interpretation:

- the query-first persistent-flat path loses again, by a wide margin
- the persistent-flat `tiled4` path again wins only narrowly

## Quiet-wrapper attempt

Command:

```bash
bash scripts/run_bench_quiet.sh -- pixi run mojo -I . \
  benchmarks/profile_document_proxy_stage2_persistent_flat_compare_browsecomp_gold.mojo
```

Artifact:

- `.cache/kayak/bench_quiet/20260415T122313Z/`

Observed host state:

- the quiet wrapper never obtained a quiet host
- competing host CPU stayed around `400%` to `800%`
- the wrapper timed out after `120s` without starting the benchmark

This matters because:

- the narrow `tiled4` win is small enough that quieter reruns would be required
  before a production decision
- on this machine, that quieter rerun was not available during this session

## What the code and data establish together

Verified:

- the benchmark-only persistent-flat scorers are exact on this seam
- the current production exact stage is already getting most of the benefit
  from flat-query `tiled4`
- storage flattening alone is not a large win here

Supported by code inspection:

- current `tiled4` already hoists flat-query build into
  [kayak/planning/exact_stage.mojo](/Users/teilomillet/Code/kayak/kayak/planning/exact_stage.mojo:423)
- the remaining storage delta is mainly the difference between:
  - nested token access through
    [kayak/scoring/maxsim.mojo](/Users/teilomillet/Code/kayak/kayak/scoring/maxsim.mojo:281)
  - contiguous token access through
    [kayak/scoring/hybrid_flat_dim128.mojo](/Users/teilomillet/Code/kayak/kayak/scoring/hybrid_flat_dim128.mojo:144)

That narrower delta is consistent with the measured result:

- replacing the storage layout without the `tiled4` kernel is much worse
- pairing the storage layout with `tiled4` buys only about `4%` prebuilt and
  about `1%` to `2%` build-inclusive on the two bounded runs

## Decision

Do **not** productionize the persistent flat stage-2 sidecar from this
evidence.

Reason:

- the required production work is non-trivial because the sidecar would need to
  be built, stored, loaded, and threaded through resolved snapshots
- the measured benefit is small and not yet verified under a quiet host
- the alternative query-first persistent-flat path is decisively worse

That tradeoff is not strong enough to justify changing the production storage
and snapshot seam.

## Current conclusion

- keep the benchmark because it makes future re-checks cheap
- keep production exact stage on the existing nested `tiled4` scorer
- treat persistent flat dim128 stage-2 as a measured negative result for now

## Follow-on idea worth testing

The data does suggest one narrower follow-on that is more plausible than a
stored sidecar:

- build a transient in-memory flat mirror once per loaded segment, instead of
  storing and loading a second index format on disk

Reason:

- prebuilt persistent-flat `tiled4` saves about `12.4` to `17.4` microseconds
  per query on the two bounded runs
- rebuilding the flat sidecar costs about `0.82` to `0.90` milliseconds
- that implies a rough one-segment break-even around `52` to `66` queries
- loading the stored sidecar costs about `1.83` to `1.97` milliseconds, which
  pushes break-even out to about `113` to `147` queries

So the current evidence does **not** support persistent storage integration, but
it does leave room for a separate experiment around:

- eager or lazy in-memory flat mirrors attached to loaded segments
- only if the serving/query lifecycle is long-lived enough to amortize the
  one-time build cost

That would still need its own memory-cost and serving-lifecycle measurements.

## Best next stage-2 direction

If stage-2 Mojo work continues, the stronger next target is likely not a new
sidecar layout. The measured result suggests the remaining opportunity is closer
to the current production seam itself:

- specialize the existing resolved-window scorer further, instead of adding a
  second index layout
- or measure stage-2 orchestration overhead around the scorer to see whether
  boundary construction, segment dispatch, or result materialization is now the
  bigger remaining tax

That should be benchmarked before any further production change.
