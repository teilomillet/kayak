# Orchestration performance — 2026-09-25

This pass measures validation, result assembly, serialization, and the Python
client/service around inference. It changes `kayak/decisions.py` and
`kayak/client.py`. Model loading, tensor operations, batching, and the input
recipe are outside the change; the ongoing Mac model experiments remain separate.

## Why these changes

A CPU profile of 1,000 wide SDK calls put response validation first, followed by
request validation and HTTPX handling. Profiles include instrumentation overhead;
their timings identify work to investigate, not normal request latency.

- **Response checks:** dictionary views compare candidate IDs without allocating
  two sets. Built-in iteration checks finite values and probability bounds without
  per-element Python generator execution. The same winner, normalization,
  softmax, and strict type checks remain active. Score computation is unchanged.
- **Blank text checks:** length constraints already reject empty strings.
  `isspace()` checks the remaining strings without making stripped copies. Text
  remains unchanged, including Unicode whitespace and edge spaces.
- **SDK encoding:** HTTP consumes UTF-8 bytes. One reusable typed serializer emits
  those bytes directly, avoiding the previous JSON-to-text-to-bytes conversion.
  The public [Pydantic TypeAdapter API](https://docs.pydantic.dev/latest/concepts/type_adapter/)
  supports byte output and recommends reusing adapters. Requests still validate
  and snapshot caller-owned dictionaries before serialization.

A native numeric-field-constraint candidate improved successful responses but
made some malformed 256-candidate responses slower to reject. It was rejected.
The retained iteration changes keep the previous error rules and messages.

## Measurement boundaries

The production baseline is `6d5a8dd`. Concurrent installation/docs (`5bfc501`)
and BANKING77 (`57efcad`) additions were preserved in the final branch; neither
changes the measured modules. Measurements use Linux x86-64, Python 3.13.15, Pydantic
2.13.5, HTTPX 0.28.1, FastAPI 0.141.1, and pyperf 2.10.0. Both source snapshots use
the same expanded benchmark harness. Timing runs use CPU affinity 1, five worker
processes, five measured values per process, two warmups, and a 50 ms minimum
value duration. Profiles, allocation checks, and correctness tests run separately.

The small fixture has one question/two candidates; the wide fixture has
32 questions/256 candidates. The padded stress fixture has four 64,000-character
Unicode texts with edge spaces and a 1,024,092-byte serialized request. It is
within request character/byte limits; no claim is made about its tokenizer limits
or usefulness to a real model.

The SDK uses MockTransport. HTTP measurements include TestClient scheduling and
a fixed result. They exclude network and model latency. Full 8B and hardware
performance must be measured through the [inference evaluation tools](../inference-evals.md).

## Observed latency

Means and sample standard deviations from the complete 30-workload comparison:

| Workload | Before | After |
| --- | ---: | ---: |
| Wide response JSON validation | 432 ± 29 µs | 342 ± 10 µs |
| Wide result assembly | 979 ± 66 µs | 817 ± 58 µs |
| Wide SDK call | 1.01 ± 0.09 ms | 0.912 ± 0.052 ms |
| Padded Python request validation | 145 ± 11 µs | 78.6 ± 6.0 µs |
| Padded SDK call | 3.06 ± 0.47 ms | 1.34 ± 0.13 ms |
| Small SDK call | 296 ± 19 µs | 289 ± 17 µs |
| Wide HTTP call | 1.45 ± 0.11 ms | 1.48 ± 0.08 ms |
| Padded HTTP call | 6.69 ± 0.58 ms | 6.47 ± 0.44 ms |

Repeating the five main improvements with the optimized code measured first
confirmed the direction: response validation took 17–21% less time, wide result
assembly 16–19% less, wide SDK calls 9–10% less, padded Python validation 44–46%
less, and padded SDK calls about 56% less. These ranges span the two comparisons;
they are not confidence intervals.

`pyperf compare_to` found no statistically significant slowdowns in the complete
30-workload comparison. Some runs have stability warnings and substantial
variation, retained in the raw reports. Small changes in untouched operations
are not evidence of an improvement. The large Unicode case is a stress case,
not an estimate of the average user's gain.

Separate controls found no startup regression (fresh-process import: 301 ms
before, 292 ms after). Wide HTTP calls with JSON diagnostics took
1.770 ± 0.135 ms before and 1.811 ± 0.049 ms after; with an unavailable log sink,
1.807 ± 0.120 ms before and 1.811 ± 0.138 ms after. Neither diagnostics change was
statistically significant. Diagnostics controls used three processes and five
values per process, with the same warmups and minimum duration.

## Observed allocations

A separate `tracemalloc` probe records incremental Python allocations for 21
warmed calls per operation, collecting garbage before each call and keeping its
result alive through measurement. Fixture construction is excluded. These are
median peak traced bytes per call, not total process memory or native allocations:

| Workload | Before | After |
| --- | ---: | ---: |
| Small SDK call | 13,949 B | 12,917 B |
| Wide SDK call | 127,780 B | 105,464 B |
| Padded SDK call | 6,146,702 B | 1,035,819 B |
| Padded Python request validation | 257,780 B | 2,372 B |
| Wide response JSON validation | 51,000 B | 50,680 B |

Direct byte serialization reduced the padded SDK peak by about 83%; the exact
1,024,092-byte request remained unchanged. Allocation tests compare returned
values too. Small and padded result assembly/decoding each use 40 more peak
traced bytes because the validator retains a dictionary-values view. This
bounded increase remains in the measurements; memory did not improve in every
individual operation. Request JSON validation peaks were unchanged.

## Preserved behavior

- Generated tests check unchanged request bytes, content length, Unicode blank
  rules, text preservation, numeric boundaries, candidate order, ties, and
  dictionary ownership. Validation and revalidation remain enabled.
- A seeded comparison of 1,500 generated answers in both Python and JSON produced
  962 accepted and 2,038 rejected validations. Accepted output bytes and rejected
  error types, locations, and messages matched the baseline exactly.
- A separate rejection check covered invalid scores and probabilities with
  1×2, 32×8, and 1×256 question/candidate layouts. Five alternating pairs of
  300 calls kept the same error counts; no median rejection time increased.
  This small diagnostic is not a statistical performance guarantee.
- Python 3.13: 287 tests passed; one MPS test was skipped because this machine has
  no MPS device. This run includes a tiny actual model and the cached released
  heads, but not the full 8B encoder.
- Python 3.11: 259 surrounding-code tests passed. Python 3.14: 257 passed, with
  two benchmark-source tests skipped because pyperf was absent. Both runs exclude
  the 29 inference tests. Existing Starlette deprecation warnings remain visible.
- All four independently installed baseline/current client-server pairings passed.
- Ruff lint/format and strict mypy passed. Source archive and wheel builds,
  strict metadata checks, and the installed base-only SDK/CLI check passed.

## Benchmark source verification

The investigation exposed a harness bug: pyperf workers launched as a script
could import an editable installation from another checkout. Hashing nearby
source files did not detect this. A reverse-order comparison was discarded for
that reason and retained with an explicit invalid label in the local evidence.

Workers now launch the benchmark as a module from its source root. Every process
checks the actual imported package directory and hashes that package. Tests run
the benchmark from a copied checkout and challenge an unrelated installation.
The timing results for the final change use this corrected harness.

## Reproduce

From each source root, using the same environment and benchmark harness:

```sh
uv run --extra bench -m benchmarks.bench_overhead \
  -p 5 -n 5 -w 2 --min-time 0.05 --affinity 1 -o .benchmarks/timing.json
uv run --extra bench pyperf check .benchmarks/timing.json
```

Choose an available CPU on another machine, omit `--affinity` where unsupported,
and use fresh output paths. Compare with
`pyperf compare_to`, retain variation and warnings, and reverse run order before
attributing a small difference to a change. Reports include the imported source
directory, source hashes, versions, and workload sizes.

Local evidence is retained in `.benchmarks/optimization-20260924/`: source
snapshots, profiles, raw timing samples, allocation samples, generated contract
comparisons, and an `attempts.txt` identifying rejected or invalid experiments.
The final complete timing pair is `isolated-before.json` / `isolated-after.json`;
the reversed pair is `repeat-after.json` / `repeat-before.json`. Allocation results
are `allocations-before.json` / `allocations-accepted.json`, with the retained
`allocations.py` probe. Earlier candidate reports are not final-change evidence.
