# Prepared Exact Snapshot Requirements And Lane RSS

## Claim

Two storage-facing questions were still open for repeated same-snapshot exact
search:

- does the Python prepared exact-session path still load optional stage-1 search
  artifacts that exact requests do not need
- when multiple same-snapshot runtime lanes are enabled, how much resident
  memory do those extra lanes actually duplicate

The justified target was not "hide a cache" or "assume the OS will fix it." It
was:

- keep the generic prepared-snapshot contract unchanged for planned search
- narrow the prepared exact-session path to exact-only snapshot requirements
- expose an explicit worker-ready seam so runtime prepare cost can be measured
- measure per-lane resident memory instead of inferring it

## Why this scope was justified

The generic prepared-snapshot API is shared by more than exact search:

- [prepare_collection_search_snapshot(...)](/Users/teilomillet/Code/kayak/kayak/service/prepared_snapshot_runtime.mojo:130)
- [prepare_service_search_snapshot(...)](/Users/teilomillet/Code/kayak/kayak/service/prepared_snapshot_runtime.mojo:144)

Changing that contract globally would have risked planned-search regressions.
The narrower exact-only seam was safer because exact search already has its own
load requirements in the stateless runtime and only needs packed index payloads
plus optional text when the request asks for it.

This also matters on ordinary service collections, not only synthetic mirrors.
The default build policy still materializes optional stage-1 artifacts:

- [default_search_artifact_build_policy(...)](/Users/teilomillet/Code/kayak/kayak/collections/search_artifact_policy.mojo:306)
  includes `document_proxy` and `centroid_postings`

So a prepared exact path that loads "all snapshot search artifacts" is broader
than necessary on the common service surface.

## Implementation

Added an internal helper that prepares a `PreparedSearchSnapshot` from explicit
requirements and then kept the old generic wrapper on top:

- [prepare_collection_search_snapshot(...)](/Users/teilomillet/Code/kayak/kayak/service/prepared_snapshot_runtime.mojo:101)

Added exact-only prepared snapshot wrappers:

- [prepare_collection_exact_search_snapshot(...)](/Users/teilomillet/Code/kayak/kayak/service/prepared_snapshot_runtime.mojo:174)
- [prepare_service_exact_search_snapshot(...)](/Users/teilomillet/Code/kayak/kayak/service/prepared_snapshot_runtime.mojo:186)

These now use:

- [exact_only_snapshot_requirements(...)](/Users/teilomillet/Code/kayak/kayak/collections/resolution_requirements.mojo:52)

Then rerouted the Python prepared exact-session binding through the exact-only
service wrapper:

- [prepare_exact_search_session(...)](/Users/teilomillet/Code/kayak/python/kayak_engine/_mojo_service_bindings.mojo:972)

For measurement, added an explicit worker-ready signal and public ready/pid
inspection on the process runtime:

- worker ready envelope emitted after per-worker session preparation in
  [prepared_exact_process_runtime.py](/Users/teilomillet/Code/kayak/python/kayak_engine/prepared_exact_process_runtime.py:96)
- runtime startup bookkeeping and `wait_until_ready(...)` in
  [prepared_exact_process_runtime.py](/Users/teilomillet/Code/kayak/python/kayak_engine/prepared_exact_process_runtime.py:223)
  and
  [prepared_exact_process_runtime.py](/Users/teilomillet/Code/kayak/python/kayak_engine/prepared_exact_process_runtime.py:308)
- ready envelopes handled in the listener at
  [prepared_exact_process_runtime.py](/Users/teilomillet/Code/kayak/python/kayak_engine/prepared_exact_process_runtime.py:530)

Extended the Python runtime benchmark to record:

- single-session prepare time
- runtime `prepare_ready_seconds`
- resident RSS of the parent process and worker processes once ready

See:

- [bench_prepared_exact_runtime.py](/Users/teilomillet/Code/kayak/python/scripts/bench_prepared_exact_runtime.py:39)
- RSS sampling helper at
  [bench_prepared_exact_runtime.py](/Users/teilomillet/Code/kayak/python/scripts/bench_prepared_exact_runtime.py:302)
- ready/RSS measurement at
  [bench_prepared_exact_runtime.py](/Users/teilomillet/Code/kayak/python/scripts/bench_prepared_exact_runtime.py:376)

## Validation

Commands run:

```bash
pixi run mojo -I . tests/test_service_prepared_snapshot_runtime.mojo
PYTHONPATH=python pixi run pytest \
  python/tests/test_prepared_exact_search_session.py \
  python/tests/test_prepared_exact_search_scheduler.py -q
bash scripts/run_bench_quiet.sh --repeats 3 --max-other-cpu 40 --timeout-seconds 30 --force -- \
  pixi run python python/scripts/bench_prepared_exact_runtime.py \
    --task .cache/kayak/browsecomp_plus_real_subset/python_task_gold.json \
    --lane-counts 1,2,4 \
    --worker-counts 2 \
    --scoring-modes default_auto \
    --warmup-iterations 2 \
    --measurement-iterations 6 \
    --request-pool-count 32 \
    --max-batch-size 32 \
    --max-batch-wait-ms 5
```

Correctness checks that passed:

- [tests/test_service_prepared_snapshot_runtime.mojo](/Users/teilomillet/Code/kayak/tests/test_service_prepared_snapshot_runtime.mojo:187)
  now verifies the exact-only prepared snapshot still matches stateless exact
  search while loading zero optional search artifacts
- `python/tests/test_prepared_exact_search_session.py`: `4 passed`
- `python/tests/test_prepared_exact_search_scheduler.py`: `7 passed`

The Python scheduler test now also verifies the new ready seam:

- [test_runtime_wait_until_ready_exposes_worker_pids(...)](/Users/teilomillet/Code/kayak/python/tests/test_prepared_exact_search_scheduler.py:181)

Quiet-wrapper artifact directory:

- `.cache/kayak/bench_quiet/20260415T191255Z`

Relevant benchmark logs:

- [.cache/kayak/bench_quiet/20260415T191255Z/run_1.txt](/Users/teilomillet/Code/kayak/.cache/kayak/bench_quiet/20260415T191255Z/run_1.txt:11)
- [.cache/kayak/bench_quiet/20260415T191255Z/run_2.txt](/Users/teilomillet/Code/kayak/.cache/kayak/bench_quiet/20260415T191255Z/run_2.txt:11)
- [.cache/kayak/bench_quiet/20260415T191255Z/run_3.txt](/Users/teilomillet/Code/kayak/.cache/kayak/bench_quiet/20260415T191255Z/run_3.txt:11)
- wrapper summary:
  [.cache/kayak/bench_quiet/20260415T191255Z/summary.txt](/Users/teilomillet/Code/kayak/.cache/kayak/bench_quiet/20260415T191255Z/summary.txt:1)

## Result

### Exact-only load contract

The new regression test on a normal service-created snapshot confirmed:

- generic prepared snapshots still load optional search artifacts
- exact-only prepared snapshots load zero optional search artifacts for exact
  search

That is mechanically verified in:

- [test_prepared_exact_snapshot_loads_exact_only_artifacts(...)](/Users/teilomillet/Code/kayak/tests/test_service_prepared_snapshot_runtime.mojo:187)

### Lane RSS and prepare cost

Median across the 3 repeated quiet-wrapper runs for
`worker_count=2`, `default_auto`, request pool `32`:

| lanes | median `prepare_ready_seconds` | median worker RSS KiB | median total RSS KiB | median throughput q/s |
| --- | ---: | ---: | ---: | ---: |
| `1` | `0.613231` | `240336` | `676208` | `801.117` |
| `2` | `0.721234` | `479344` | `915472` | `763.309` |
| `4` | `0.798511` | `958672` | `1396016` | `811.326` |

Single-session exact-only prepare time in the same benchmark harness:

- median `session_prepare_seconds`: `0.008388 s`

Derived worker-RSS ratios:

- `2 lanes` vs `1 lane`: `1.99x`
- `4 lanes` vs `1 lane`: `3.99x`

Derived throughput ratios on this noisy host:

- `2 lanes` vs `1 lane`: `0.95x`
- `4 lanes` vs `1 lane`: `1.01x`

## Interpretation

What is clearly true:

- the prepared exact-session path is now narrower than the generic prepared
  snapshot path
- extra runtime lanes duplicate prepared worker memory almost linearly
- on this host and slice, lane count raised resident memory much more reliably
  than it raised throughput

This is the storage-facing answer for same-machine concurrent same-snapshot
search:

- prepared reuse removes per-query snapshot reloads
- but each extra process lane still pays its own prepare-time load and keeps its
  own resident copy
- so the main cost of "many simultaneous searches on the same snapshot" shifts
  from repeated storage reads per query to duplicated resident memory per lane

## Important Uncertainty

These are not clean quiet-host absolute latency numbers.

All three quiet-wrapper runs timed out and proceeded under `--force`. The wrapper
logs show substantial competing host load, including editor activity and other
Python/Mojo processes.

One two-lane run was an obvious outlier in both `prepare_ready_seconds` and RSS,
which is why the medians above are more trustworthy than any single run.

## What this does not prove

- physical disk I/O bytes after OS page cache effects
- networked remote-storage behavior when search and storage are on different
  machines
- hosted HTTP transport throughput, because the current HTTP server is still
  single-request at the transport boundary

## Conclusion

The current same-machine multi-search guidance is now better grounded:

- use prepared exact sessions/runtimes when the same published snapshot is
  reused, because they remove the per-query snapshot load tax
- do not assume extra lanes are free; they duplicate resident prepared-snapshot
  memory almost linearly
- treat lane count as a memory-budget decision first and a throughput decision
  second, then measure on the target host

The next storage-facing step, if needed, is not another blind concurrency knob.
It is to measure the same lane sweep on a larger collection and then compare the
same lane counts under truly concurrent hosted transport rather than the current
single-threaded HTTP server.
