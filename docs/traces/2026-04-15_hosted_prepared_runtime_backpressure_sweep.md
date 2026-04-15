# Hosted Prepared Runtime Backpressure Sweep

## Claim

After adding explicit overload admission for hosted prepared exact runtimes, the
next open question was narrower than "more optimization":

- is the current derived default `max_outstanding_request_count` too deep
- does a smaller cap materially improve burst latency enough to justify changing
  the default
- can we answer that at the real hosted HTTP seam rather than by inspecting
  runtime counters in isolation

The justified target was:

- add one reproducible hosted burst benchmark
- sweep admission caps against concurrent prepared-search traffic
- keep the default only if smaller caps do not clearly dominate

## Why this scope was justified

The repo already had evidence for three adjacent questions:

- hosted prepared exact-runtime routes can overlap across HTTP requests
- runtime overload now returns explicit `429`
- runtime stats expose queue depth, batch size, and rejection counts

What was still missing was the decision boundary between those facts:

- we had a contract
- we did **not** yet have evidence that the current default depth was right or
  wrong under burst load

Changing the default without that measurement would have been speculative,
because smaller caps trade one failure mode for another:

- lower queue depth and lower accepted latency
- but also more immediate rejection and lower accepted throughput

## Implementation

Added the benchmark script:

- [bench_hosted_prepared_exact_runtime_backpressure.py](/Users/teilomillet/Code/kayak/python/scripts/bench_hosted_prepared_exact_runtime_backpressure.py:1)

What it does:

- stages one encoded task into a temporary hosted service root
- starts the real hosted HTTP server
- prepares hosted exact runtimes with explicit:
  - `concurrency_lane_count`
  - `worker_count`
  - `max_batch_size`
  - `max_batch_wait_ms`
  - `max_outstanding_request_count`
- computes stateless exact-search references once
- fires one synchronized burst of concurrent
  `POST /v1/prepared-exact-search` requests
- records:
  - accepted and rejected counts
  - accepted and rejected latency percentiles
  - runtime batch and queue stats

Added a smoke test for the benchmark contract:

- [test_prepared_runtime_benchmark_smoke.py](/Users/teilomillet/Code/kayak/python/tests/test_prepared_runtime_benchmark_smoke.py:1)

That test only verifies:

- the script can stage a tiny hosted runtime
- the script emits the expected JSON summary shape

Reason:

- benchmark tooling should be mechanically checked
- but the smoke test should not pretend to validate performance claims

## Validation

Commands run:

```bash
python3 -m py_compile \
  python/scripts/bench_hosted_prepared_exact_runtime_backpressure.py \
  python/tests/test_prepared_runtime_benchmark_smoke.py
PYTHONPATH=python pixi run pytest \
  python/tests/test_prepared_runtime_benchmark_smoke.py -q
```

Observed:

- `python/tests/test_prepared_runtime_benchmark_smoke.py`: `1 passed`

## Measured Setup

Task:

- dataset: `Tevatron/browsecomp-plus/gold-slice`
- task json:
  [.cache/kayak/browsecomp_plus_real_subset/python_task_gold.json](/Users/teilomillet/Code/kayak/.cache/kayak/browsecomp_plus_real_subset/python_task_gold.json:1)
- documents: `90`
- distinct queries in source slice: `4`
- request pool: `32`
- burst size: `128`

Common runtime settings:

- `concurrency_lane_count=1`
- `worker_count=2`
- `max_batch_wait_ms=25`
- `repeats=5`

Measured artifacts:

- batch-8 sweep:
  [.cache/kayak/hosted_prepared_runtime_backpressure_browsecomp_gold_batch8.json](/Users/teilomillet/Code/kayak/.cache/kayak/hosted_prepared_runtime_backpressure_browsecomp_gold_batch8.json:1)
- batch-32 sweep:
  [.cache/kayak/hosted_prepared_runtime_backpressure_browsecomp_gold_batch32.json](/Users/teilomillet/Code/kayak/.cache/kayak/hosted_prepared_runtime_backpressure_browsecomp_gold_batch32.json:1)

Commands:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 40 --timeout-seconds 30 --force -- \
  pixi run python python/scripts/bench_hosted_prepared_exact_runtime_backpressure.py \
    --task .cache/kayak/browsecomp_plus_real_subset/python_task_gold.json \
    --request-pool-count 32 \
    --burst-sizes 128 \
    --lane-counts 1 \
    --worker-counts 2 \
    --outstanding-counts 0,8,16,32,64,128 \
    --max-batch-size 8 \
    --max-batch-wait-ms 25 \
    --repeats 5 \
    --output .cache/kayak/hosted_prepared_runtime_backpressure_browsecomp_gold_batch8.json

bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 40 --timeout-seconds 30 --force -- \
  pixi run python python/scripts/bench_hosted_prepared_exact_runtime_backpressure.py \
    --task .cache/kayak/browsecomp_plus_real_subset/python_task_gold.json \
    --request-pool-count 32 \
    --burst-sizes 128 \
    --lane-counts 1 \
    --worker-counts 2 \
    --outstanding-counts 0,32,64,128 \
    --max-batch-size 32 \
    --max-batch-wait-ms 25 \
    --repeats 5 \
    --output .cache/kayak/hosted_prepared_runtime_backpressure_browsecomp_gold_batch32.json
```

## Important Uncertainty

Both quiet-wrapper runs timed out waiting for a quiet host and proceeded under
`--force`.

Observed competing load included:

- active editor CPU usage
- an unrelated long-running biology benchmark process
- persistent `modular-crashpad-handler` activity

The strongest sign that this host was still noisy is that two rows with the
same resolved cap sometimes diverged materially:

- `requested=0` and `requested=128` both resolve to `128`
- yet their medians were not identical, especially in the batch-32 sweep

That means these measurements are decision-quality enough to debunk obviously
bad claims, but **not** clean enough to justify a universal retune from small
differences alone.

## Median Results

### `max_batch_size=8`

Median across 5 internal repeats:

| requested cap | resolved cap | accepted | rejected | median wall ms | median accepted q/s | median accepted p95 ms | median queue-wait ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `128` | `128` | `0` | `387.186` | `330.590` | `288.492` | `69.993` |
| `8` | `8` | `52` | `76` | `326.667` | `159.184` | `227.193` | `13.122` |
| `16` | `16` | `69` | `59` | `319.239` | `208.300` | `234.271` | `18.519` |
| `32` | `32` | `88` | `40` | `329.194` | `277.583` | `240.504` | `31.861` |
| `64` | `64` | `111` | `17` | `368.107` | `296.618` | `293.828` | `64.173` |
| `128` | `128` | `128` | `0` | `392.076` | `326.467` | `284.051` | `68.712` |

What this says:

- smaller caps clearly reduce queue depth and queue wait
- but every smaller cap also rejects real work
- no smaller cap dominated the resolved-`128` case on both accepted throughput
  and accepted tail latency

One instructive comparison:

- `cap=32` cut median accepted p95 from `288.492 ms` to `240.504 ms`
- but it also rejected a median `40` of `128` requests

That is a real tradeoff, not a free win.

### `max_batch_size=32`

Median across 5 internal repeats:

| requested cap | resolved cap | accepted | rejected | median wall ms | median accepted q/s | median accepted p95 ms | median queue-wait ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `128` | `128` | `0` | `389.486` | `328.638` | `305.352` | `63.439` |
| `32` | `32` | `65` | `63` | `346.162` | `195.904` | `251.217` | `33.051` |
| `64` | `64` | `105` | `23` | `385.455` | `272.405` | `311.939` | `48.956` |
| `128` | `128` | `128` | `0` | `492.082` | `260.119` | `362.058` | `82.806` |

What this says:

- `cap=32` again lowers queue wait and accepted p95
- but it rejects nearly half the burst
- `cap=64` preserved more accepts, yet its median accepted p95 was slightly
  worse than the derived-default row
- the two resolved-`128` rows diverged enough to expose host noise directly

So this slice does **not** support a clean "smaller cap is simply better"
conclusion.

## Interpretation

What is verified:

- the benchmark can measure hosted burst admission at the real HTTP seam
- lower caps do exactly what they should:
  - cap queue depth
  - lower queue-wait time
  - reject excess work quickly
- but lower caps also reduce the number of accepted requests in the same burst

What the data does **not** show:

- a smaller cap that dominates the current default on the combination of:
  - accepted throughput
  - accepted tail latency
  - rejection rate

This matters because the current default is a service policy, not only a
microbenchmark knob. If a smaller cap only helps by rejecting much more work,
that is not automatically an improvement.

## Conclusion

The sound outcome of this sweep is:

- keep `max_outstanding_request_count` explicit
- keep the current derived default unchanged for now
- use the new benchmark when tuning for a specific host or workload

Why no default change landed:

- the current measurements do not show a smaller cap clearly dominating
- repeated noisy-host runs produced meaningful variance even between
  identically resolved caps
- changing the default now would overfit to ambiguous local data

So the repository now has the missing evidence tool:

- a real hosted backpressure benchmark
- a smoke-tested JSON output contract
- measured guidance that separates:
  - "smaller cap lowers queue depth"
  - from
  - "smaller cap is globally better"

Only the first claim is verified here.
