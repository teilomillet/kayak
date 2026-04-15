# Service Prepared Snapshot Reuse

## Claim

The next question after landing the explicit prepared-snapshot seam was:

- does service-side reuse of one loaded published snapshot remove a real request-time cost
- can we verify the effect without hiding planner policy or changing snapshot semantics
- what would concurrent requests on the same hosted engine actually do today

The sound target was not "add a hidden cache." It was:

- keep `snapshot_id` explicit in the request surface
- expose one explicit prepared-snapshot seam for repeated search on the same
  published snapshot
- measure the reuse benefit separately from the hosted HTTP transport

## Code inspection that motivated the check

The current stateless service path still reloads snapshot state on every
request:

- [execute_search(...)](/Users/teilomillet/Code/kayak/kayak/service/runtime.mojo:511)
  calls
  [load_resolved_collection_snapshot(...)](/Users/teilomillet/Code/kayak/kayak/service/runtime.mojo:535)
- [execute_explain(...)](/Users/teilomillet/Code/kayak/kayak/service/runtime.mojo:683)
  does the same at
  [runtime.mojo:707](/Users/teilomillet/Code/kayak/kayak/service/runtime.mojo:707)
- planned search still reloads planning availability through
  [select_search_plan_for_request(...)](/Users/teilomillet/Code/kayak/kayak/service/runtime.mojo:600)
  at
  [runtime.mojo:622](/Users/teilomillet/Code/kayak/kayak/service/runtime.mojo:622)

That matters because
[load_resolved_collection_snapshot(...)](/Users/teilomillet/Code/kayak/kayak/collections/resolver.mojo:528)
is not a metadata lookup. It:

- loads the collection and snapshot manifests
- walks every sealed segment in the published snapshot
- loads the packed index for each segment
- loads configured search artifacts that the request requires
- optionally loads text corpora

The reusable seam is explicit in
[prepared_snapshot_runtime.mojo](/Users/teilomillet/Code/kayak/kayak/service/prepared_snapshot_runtime.mojo:1):

- [PreparedSearchSnapshot](/Users/teilomillet/Code/kayak/kayak/service/prepared_snapshot_runtime.mojo:51)
  owns the loaded snapshot plus planning availability
- [prepare_service_search_snapshot(...)](/Users/teilomillet/Code/kayak/kayak/service/prepared_snapshot_runtime.mojo:95)
  materializes that state once
- [execute_search_with_prepared_snapshot(...)](/Users/teilomillet/Code/kayak/kayak/service/prepared_snapshot_runtime.mojo:153)
  and
  [execute_planned_search_with_prepared_snapshot(...)](/Users/teilomillet/Code/kayak/kayak/service/prepared_snapshot_runtime.mojo:248)
  reuse it while still validating collection, tenant, namespace, snapshot, and
  query shape

This is why the prepared-snapshot seam was the right thing to measure:

- it removes repeated snapshot loading work that is visible in the code
- it does not make retrieval policy implicit
- it keeps search pinned to one published snapshot

## Validation

Commands run:

```bash
pixi run mojo -I . tests/test_service_prepared_snapshot_runtime.mojo
pixi run mojo -I . tests/test_service_runtime.mojo
pixi run mojo -I . benchmarks/profile_service_prepared_snapshot_reuse_browsecomp_gold.mojo
bash scripts/run_bench_quiet.sh --repeats 3 --max-other-cpu 40 --timeout-seconds 10 --force -- \
  pixi run mojo -I . benchmarks/profile_service_prepared_snapshot_reuse_browsecomp_gold.mojo
```

Correctness checks that passed:

- `tests/test_service_prepared_snapshot_runtime.mojo`: `2/2`
- `tests/test_service_runtime.mojo`: `26/26`

The targeted prepared-snapshot tests verify two properties that matter for
reuse:

- [test_prepared_snapshot_stays_pinned_after_new_snapshot_publish()](/Users/teilomillet/Code/kayak/tests/test_service_prepared_snapshot_runtime.mojo:92)
  shows a prepared snapshot for `snapshot-0001` still returns the old document
  after a new draft mutation and publication of `snapshot-0002`
- [test_planned_search_with_prepared_snapshot_matches_stateless_runtime()](/Users/teilomillet/Code/kayak/tests/test_service_prepared_snapshot_runtime.mojo:186)
  shows the planned prepared path matches the current stateless planned path

Benchmark surface:

- benchmark:
  [profile_service_prepared_snapshot_reuse_browsecomp_gold.mojo](/Users/teilomillet/Code/kayak/benchmarks/profile_service_prepared_snapshot_reuse_browsecomp_gold.mojo:1)
- dataset: `Tevatron/browsecomp-plus/gold-slice`
- slice: `browsecomp_plus_gold_slice`
- queries: `4`
- documents: `90`
- nominal query vectors: `32`
- nominal document vectors: `175`
- vector dim: `128`
- service shape: one-segment mirrored collection rooted at
  `.cache/kayak/profile_service_prepared_snapshot_reuse_browsecomp_gold`

I forced the planned benchmark to prefer `exact_full_scan` so the measurement
isolates service-side snapshot reuse instead of mixing in heavier stage-1 or
stage-2 planner variation.

The benchmark now uses bounded iteration counts in
[profile_service_prepared_snapshot_reuse_browsecomp_gold.mojo](/Users/teilomillet/Code/kayak/benchmarks/profile_service_prepared_snapshot_reuse_browsecomp_gold.mojo:31)
through
[benchmark.run(...)](/Users/teilomillet/Code/kayak/benchmarks/profile_service_prepared_snapshot_reuse_browsecomp_gold.mojo:207)
call sites. Reason:

- the repo asks for comparable evidence, not one noisy exploratory run
- this host did not become quiet enough for the wrapper threshold, so bounded
  runs keep the forced reruns practical and comparable

Quiet-wrapper artifact directory:

- `.cache/kayak/bench_quiet/20260415T131136Z`

The wrapper metadata confirms the repeated command and thresholds:

- `.cache/kayak/bench_quiet/20260415T131136Z/meta.txt`

Host contention remained high during quiet checks. One sample from the wrapper
log already showed large competing load from `Zed`, `Helium`, and several
`modular-crashpad-handler` processes:

- `.cache/kayak/bench_quiet/20260415T131136Z/quiet_check_run_1_sample_0.txt`

So these are not decision-quality absolute wall-clock numbers. They are useful
because the reuse gap stayed stable across all three forced reruns.

## Result

Per-run means from the quiet-wrapper reruns:

- run 1
  - `prepare_snapshot`: `0.020051348837209302 s`
  - `exact_search_stateless`: `0.019197441860465115 s`
  - `exact_search_prepared`: `0.0010685652173913042 s`
  - `planned_search_stateless_exact_preferred`: `0.019226232558139534 s`
  - `planned_search_prepared_exact_preferred`: `0.0011504786324786325 s`
- run 2
  - `prepare_snapshot`: `0.02015860465116279 s`
  - `exact_search_stateless`: `0.0193003488372093 s`
  - `exact_search_prepared`: `0.0010713589743589743 s`
  - `planned_search_stateless_exact_preferred`: `0.019291627906976746 s`
  - `planned_search_prepared_exact_preferred`: `0.0010893879310344828 s`
- run 3
  - `prepare_snapshot`: `0.019773651162790697 s`
  - `exact_search_stateless`: `0.018755209302325583 s`
  - `exact_search_prepared`: `0.0010767094017094018 s`
  - `planned_search_stateless_exact_preferred`: `0.01894474418604651 s`
  - `planned_search_prepared_exact_preferred`: `0.0010876923076923077 s`

Median-style summary across the three runs:

- `prepare_snapshot`: `0.020051 s`
- `exact_search_stateless`: `0.019197 s`
- `exact_search_prepared`: `0.001071 s`
- `planned_search_stateless_exact_preferred`: `0.019226 s`
- `planned_search_prepared_exact_preferred`: `0.001089 s`

Derived ratios:

- exact prepared reuse speedup: `17.92x`
- planned exact-preferred prepared reuse speedup: `17.65x`

Interpretation:

- on this repeated-search slice, request-time snapshot loading is a real cost
- once the snapshot is already prepared, search latency drops by roughly an
  order of magnitude plus another half
- the one-time prepare step is about the same cost as one stateless query on
  this slice, so the seam is only justified when the same published snapshot is
  reused across multiple requests

## Hosted concurrency facts verified from code

The hosted Python transport is still single-request at the HTTP boundary:

- [server.py](/Users/teilomillet/Code/kayak/python/kayak_engine/server.py:7)
  imports `HTTPServer`
- [KayakEngineHttpServer](/Users/teilomillet/Code/kayak/python/kayak_engine/server.py:32)
  subclasses `HTTPServer`
- [do_POST(...)](/Users/teilomillet/Code/kayak/python/kayak_engine/server.py:80)
  dispatches one handler directly and writes the response synchronously

The server object currently stores only:

- `service_root`
- `engine_module`

Those are the only fields set in
[KayakEngineHttpServer.__init__](/Users/teilomillet/Code/kayak/python/kayak_engine/server.py:35).
There is no prepared-snapshot cache or lease state in the current hosted
transport.

That means, today:

- if one HTTP request is still running, another request to the same hosted
  process waits in the server loop
- repeated hosted requests still take the stateless path unless the caller is
  using the explicit prepared-snapshot runtime seam directly

Snapshot consistency is still explicit and stable:

- search requests carry a `snapshot_id`
- prepared snapshots validate that `collection_id`, `tenant_id`,
  `namespace_id`, and `snapshot_id` still match the request
- the pinned-snapshot test above verified that publishing `snapshot-0002` does
  not change a prepared `snapshot-0001`

So the current behavior for "simultaneous or overlapping requests on the same
Kayak service" is:

- transport-level serialization across requests in one hosted server process
- immutable published snapshot semantics for the request being executed
- no automatic cross-request snapshot reuse yet

## Bottom line

What is verified:

- the explicit prepared-snapshot seam is the right abstraction for repeated
  search on one published snapshot
- the measured reuse benefit is large and stable on the BrowseComp+ gold slice
- the benefit comes from removing a real request-time snapshot-loading tax that
  is visible in the current stateless runtime

What is not yet true:

- the hosted Python HTTP server does not automatically exploit this seam
- current hosted concurrency is not multi-request parallelism; it is one
  request at a time per server process

## Next step

The next justified systems step is not another micro-optimization inside the
prepared seam. It is an explicit hosted prepared-snapshot cache with clear
policy and invalidation rules, then measure it under a request loop that
actually reuses one published snapshot.

That future cache should stay explicit about at least:

- cache key: collection, tenant, namespace, snapshot, and text-corpus loading
  mode
- invalidation on reclaim or any operation that removes referenced snapshot
  artifacts
- reuse scope: per-process and measurable, not hidden behind planner policy
