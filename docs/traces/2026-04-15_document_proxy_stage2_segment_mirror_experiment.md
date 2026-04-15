# Document Proxy Stage-2 Segment Mirror Experiment

## Claim

After the persistent-flat sidecar experiment, the narrower follow-on question
was:

- if a loaded snapshot builds an in-memory flat dim128 mirror once per segment,
  can exact stage-2 scoring reuse that mirror cheaply enough to justify further
  production work

This is different from the stored sidecar question:

- no extra on-disk format
- no extra snapshot manifest plumbing
- one build per loaded snapshot instead of one build or load per query

## Why this experiment was the right next check

The previous persistent-flat experiment established two things:

- the plain hybrid-flat query-first path is much slower than current stage-2
- the hybrid-flat `tiled4` path only has a small possible win

That meant the only remaining plausible version was:

- build the flat mirror once
- reuse it over many exact stage-2 requests against the same loaded snapshot

Before testing that, I verified two service/runtime constraints:

- the hosted HTTP transport is currently
  [HTTPServer](/Users/teilomillet/Code/kayak/python/kayak_engine/server.py:32),
  not `ThreadingHTTPServer`
- each search request currently calls
  [load_resolved_collection_snapshot(...)](/Users/teilomillet/Code/kayak/kayak/service/runtime.mojo:535)
  inside
  [execute_search(...)](/Users/teilomillet/Code/kayak/kayak/service/runtime.mojo:511)

So the current hosted path does **not** yet reuse loaded snapshots across
requests. That matters because:

- a segment mirror only helps if the same loaded snapshot survives across
  multiple queries
- otherwise the mirror build cost is paid every request and should be treated as
  part of the request latency

## Search-concurrency facts verified from code

### Hosted HTTP transport

Current hosted transport:

- [python/kayak_engine/server.py](/Users/teilomillet/Code/kayak/python/kayak_engine/server.py:7)
  imports `HTTPServer`
- [python/kayak_engine/server.py](/Users/teilomillet/Code/kayak/python/kayak_engine/server.py:32)
  subclasses `HTTPServer`

That means:

- the current hosted server is single-process and single-request-at-a-time at
  the HTTP transport layer
- if one request is still running, another request to that same process waits
  for the handler loop to become free

### Search contract

Current search requests are snapshot-pinned:

- [kayak/service/search_contracts.mojo](/Users/teilomillet/Code/kayak/kayak/service/search_contracts.mojo:26)
  includes `snapshot_id` as part of `SearchRequest`

That means:

- search semantics are anchored to a specific published snapshot
- upserts and deletes affect draft state, not already-published snapshots
- creating a new snapshot publishes a new immutable search target

The relevant mutation path is:

- [kayak/service/runtime.mojo](/Users/teilomillet/Code/kayak/kayak/service/runtime.mojo:204)
  `upsert_documents(...)`
- [kayak/service/runtime.mojo](/Users/teilomillet/Code/kayak/kayak/service/runtime.mojo:268)
  `delete_documents(...)`
- [kayak/service/runtime.mojo](/Users/teilomillet/Code/kayak/kayak/service/runtime.mojo:407)
  `create_snapshot(...)`

So, for the current hosted service:

- overlapping search requests are serialized by the transport
- a search against `snapshot-0001` is not changed by concurrent draft-state
  writes because it explicitly loads `snapshot-0001`

### Within-request parallelism

Although the hosted transport is single-request, a single request still uses
parallel CPU work inside the scorer:

- [kayak/planning/exact_stage.mojo](/Users/teilomillet/Code/kayak/kayak/planning/exact_stage.mojo:363)
  partitions resolved candidate windows
- [kayak/planning/exact_stage.mojo](/Users/teilomillet/Code/kayak/kayak/planning/exact_stage.mojo:410)
  runs those partitions with `sync_parallelize`

So the current model is:

- one request at a time at the hosted-server boundary
- multiple cores within a request when the exact-stage shape warrants it

## Benchmark seam

Added:

- [benchmarks/profile_document_proxy_stage2_segment_mirror_compare_browsecomp_gold.mojo](/Users/teilomillet/Code/kayak/benchmarks/profile_document_proxy_stage2_segment_mirror_compare_browsecomp_gold.mojo:1)

This benchmark:

- loads the real BrowseComp-Plus gold `document_proxy` slice through a resolved
  snapshot
- builds one in-memory `HybridFlatDim128Index` mirror per loaded segment
- validates exact score equivalence against the current stage-2 scorer
- measures both prebuilt-mirror and build-per-request cases

Measured operating point:

- dataset: `BrowseComp-Plus Gold`
- slice: `browsecomp_plus_gold_slice`
- segments: `1`
- documents: `90`
- query vectors: `32`
- document vectors: `175`
- `candidate_k = 40`

The benchmark uses bounded timings from the start:

- stage-2 sections:
  `min_runtime_secs=0.05`, `max_runtime_secs=0.25`, `max_iters=128`
- build sections:
  `min_runtime_secs=0.01`, `max_runtime_secs=0.10`, `max_iters=20`

Reason:

- this repo requires benchmark evidence that is reproducible enough to compare
- open-ended benchmark loops were already shown to be fragile under this host

## Direct bounded run 1

Artifact:

- `.cache/kayak/profile_document_proxy_stage2_segment_mirror_compare_browsecomp_gold.run1.tsv`

Measured means:

- current scorer:
  `0.0002999940119760479 s`
- `build_flat_query_dim128`:
  `2.125127507650459e-06 s`
- `build_segment_hybrid_flat_dim128_mirrors`:
  `0.0008449 s`
- prebuilt segment-mirror `tiled4`:
  `0.0002780388888888889 s`
- `build_flat_query + prebuilt segment-mirror tiled4`:
  `0.00028207865168539326 s`
- `build_segment_mirrors + score`:
  `0.001133296875 s`
- `build_segment_mirrors + build_flat_query + score`:
  `0.0011125390625 s`

Ratios versus current:

- prebuilt segment mirror: `0.9268x`
- build-inclusive prebuilt segment mirror: `0.9403x`
- build-mirrors per query: `3.7777x`
- build-mirrors + flat-query per query: `3.7085x`

Interpretation:

- if the mirror is already built, this run shows a modest win
- if the mirror is built inside each request, it is a large loss

## Direct bounded run 2

Artifact:

- `.cache/kayak/profile_document_proxy_stage2_segment_mirror_compare_browsecomp_gold.tsv`

Measured means:

- current scorer:
  `0.00028606285714285714 s`
- `build_flat_query_dim128`:
  `2.10158876933423e-06 s`
- `build_segment_hybrid_flat_dim128_mirrors`:
  `0.0008169 s`
- prebuilt segment-mirror `tiled4`:
  `0.00029514619883040937 s`
- `build_flat_query + prebuilt segment-mirror tiled4`:
  `0.00029035838150289015 s`
- `build_segment_mirrors + score`:
  `0.001100515625 s`
- `build_segment_mirrors + build_flat_query + score`:
  `0.0011028828125 s`

Ratios versus current:

- prebuilt segment mirror: `1.0318x`
- build-inclusive prebuilt segment mirror: `1.0150x`
- build-mirrors per query: `3.8478x`
- build-mirrors + flat-query per query: `3.8561x`

Interpretation:

- this run does **not** confirm the prebuilt-mirror win
- the build-per-request loss remains clear

## Quiet-wrapper forced rerun

Command:

```bash
bash scripts/run_bench_quiet.sh --repeats 2 --timeout-seconds 20 --force -- \
  pixi run mojo -I . \
  benchmarks/profile_document_proxy_stage2_segment_mirror_compare_browsecomp_gold.mojo
```

Artifact:

- `.cache/kayak/bench_quiet/20260415T123713Z/`

Host note:

- the quiet wrapper never obtained a quiet host
- both runs were force-started after timeout
- competing host CPU stayed around `400%` to `857%`
- an unrelated external pytest process was active during run 2

That means these runs are still host-contended. They are useful for direction,
not for fine-grained promotion decisions.

### Quiet-wrapper run 1

Measured means:

- current scorer:
  `0.0003106832298136646 s`
- `build_segment_hybrid_flat_dim128_mirrors`:
  `0.0008031999999999999 s`
- prebuilt segment-mirror `tiled4`:
  `0.0002953588235294118 s`
- `build_flat_query + prebuilt segment-mirror tiled4`:
  `0.0002959058823529412 s`
- `build_segment_mirrors + score`:
  `0.0011220703125 s`
- `build_segment_mirrors + build_flat_query + score`:
  `0.0011172421875 s`

Ratios versus current:

- prebuilt segment mirror: `0.9507x`
- build-inclusive prebuilt segment mirror: `0.9525x`
- build-mirrors per query: `3.6113x`
- build-mirrors + flat-query per query: `3.5958x`

### Quiet-wrapper run 2

Measured means:

- current scorer:
  `0.000305109756097561 s`
- `build_segment_hybrid_flat_dim128_mirrors`:
  `0.0008250999999999999 s`
- prebuilt segment-mirror `tiled4`:
  `0.00030201807228915663 s`
- `build_flat_query + prebuilt segment-mirror tiled4`:
  `0.0003041939393939394 s`
- `build_segment_mirrors + score`:
  `0.001131859375 s`
- `build_segment_mirrors + build_flat_query + score`:
  `0.0011468515625 s`

Ratios versus current:

- prebuilt segment mirror: `0.9899x`
- build-inclusive prebuilt segment mirror: `0.9970x`
- build-mirrors per query: `3.7098x`
- build-mirrors + flat-query per query: `3.7589x`

## What the evidence establishes

Verified:

- the segment-mirror `tiled4` scorer is exact on this seam
- building mirrors inside each request is unequivocally bad on this seam
- prebuilt mirror wins are narrow and unstable

Consistent across all four measured runs:

- `build_segment_hybrid_flat_dim128_mirrors + score` loses badly to current
- the loss is around `3.6x` to `3.85x`

Mixed across runs:

- prebuilt mirror scoring ranges from about `7.3%` faster to about `3.2%`
  slower on the two direct runs
- forced quiet-wrapper runs narrow that spread, but still only show about
  `1.0%` to `4.9%` wins under host contention

## Why productionization is not justified yet

The current hosted path would still need two other changes before this
optimization could pay off reliably:

- concurrent or at least reusable service-side snapshot lifetime
- an explicit snapshot cache or loaded-snapshot reuse policy

Without those, the mirror build cost is paid inside the request, and that case
is clearly worse.

Even if snapshot reuse were added later, the current measured prebuilt win is
too small and too unstable to justify immediately threading a new optional
mirror through production snapshot objects.

## Decision

Do **not** productionize transient segment mirrors from this evidence.

Reason:

- per-request build is a large regression
- prebuilt wins are not stable enough yet
- the current hosted service does not reuse loaded snapshots anyway

## Best next move

If serving-path optimization is the goal, the next stronger target is not a
mirror kernel by itself. It is the serving seam:

- introduce a measured, explicit loaded-snapshot cache or service object
- only then re-benchmark optional per-segment flat mirrors on top of that cache

That would answer the real product question:

- repeated requests against the same published snapshot in a long-lived process

Without that serving seam, the mirror remains a microbenchmark curiosity rather
than a justified engine optimization.
