# Historical code measurements

These observations were recorded during development on 2026-09-24. Test counts,
source sizes, experimental candidates, and timings refer to those snapshots.
They are not current release results or production capacity guarantees. Use the
[development guide](../development.md) to reproduce measurements on the current code.

## Initial observations — 2026-09-24

Measured on this Linux x86-64 VM with Python 3.13.15, Pydantic 2.13.5, HTTPX
0.28.1, FastAPI 0.141.1, and pyperf 2.10.0. Each timing used three worker processes,
five measured values per worker, two warmups, and a 0.03-second minimum value
duration. The command was:

```sh
uv run --extra bench -m benchmarks.bench_overhead -p 3 -n 5 -w 2 --min-time 0.03 \
  -o .benchmarks/before.json
```

The baseline already used the new package layout. The changes removed an
intermediate dictionary conversion of typed requests, used direct Pydantic JSON
encoding in the client/server, and validated the owned HTTP bytearray directly.
The body limit is checked before copying the next chunk. Input limits, strict
validation, result meaning, ordering, and model computation were preserved.

| Measurement | Before mean | After mean |
| --- | ---: | ---: |
| Small typed request validation | 10.9 µs | 7.22 µs |
| Wide typed request validation | 245 µs | 163 µs |
| Wide client call | 1.21 ms | 1.06 ms |
| Wide HTTP call | 1.94 ms | 1.48 ms |

The side-by-side wide response encoding experiment measured 365 µs through a
dictionary versus 88.9 µs through direct JSON. The small client comparison was
not statistically significant. Several runs triggered pyperf stability warnings;
the wide HTTP baseline had 12% standard deviation. These are observed local
improvements, not hardware-independent speed guarantees. The short CPU profile
also showed that response contract validation remains a substantial cost; its
checks were retained.

Raw local evidence is `.benchmarks/before.json`, `.benchmarks/after.json`, their
logs, `.benchmarks/client-before.prof`, and `.benchmarks/client-memory.json`.
The baseline source files are retained in `.benchmarks/before-source/`. Source
hash recording and untimed transport checks were added to the harness after
the initial timing pair; the measured operations and fixtures are unchanged.

The expanded suite passed 59 tests with the released heads enabled on Python
3.13.15. The lightweight Pixi environment passed all 48 non-inference tests on
Python 3.14.7, with 11 explicitly deselected. Full 8B validation remains separate
as described in [the model validation instructions](../validation.md).

## Speed reassessment after the readability changes — 2026-09-24

The new baseline used the same Python 3.13.15 environment as the initial review,
five worker processes, five values per worker, two warmups, and a 0.05-second
minimum value duration. It measures the current code; previous smoke timings
are not used as a performance baseline.

| Current operation | Small: 1 question, 2 candidates | Wide: 32 questions, 256 candidates |
| --- | ---: | ---: |
| Typed request validation | 6.91 ± 0.56 µs | 162 ± 19 µs |
| JSON request validation | 5.86 ± 0.39 µs | 181 ± 12 µs |
| Text preparation | 0.626 ± 0.047 µs | 10.1 ± 0.9 µs |
| Response validation | 11.5 ± 1.1 µs | 436 ± 43 µs |
| Direct JSON response encoding | 6.12 ± 0.54 µs | 93.3 ± 8.5 µs |
| Client through MockTransport | 302 ± 17 µs | 1.05 ± 0.08 ms |
| HTTP through TestClient | 1.13 ± 0.09 ms | 1.50 ± 0.09 ms |

Values are means ± sample standard deviations. Pyperf reported stability
warnings; these VM measurements do not resolve small differences reliably.
The raw baseline and log are `.benchmarks/speed-review-before.{json,log}`.
The exact starting source is in `.benchmarks/speed-review-before-source/`.

Separate profiles of 1,000 wide calls identified response validation as the
largest client component. Profile timings are instrumented and are not latency
measurements. A further profile exposed work missing from the original harness:
constructing each `ChoiceAnswer`, then constructing `DecisionResult`, runs **64
distribution validators and 96 softmax computations per wide result**. The
client/HTTP fixtures return a prebuilt result, so they never measured that work.

The harness now measures current result assembly and a candidate that builds
fresh field dictionaries and validates the complete result once. It retains
score-count, unique-ID, finite-score, and full result checks. The candidate runs
32 distribution validators and 64 softmax computations for the same wide result.
No validation setting or library implementation was changed.

| Result assembly | Current | Candidate | Reduction in elapsed time |
| --- | ---: | ---: | ---: |
| Small, first order | 25.25 ± 3.05 µs | 16.41 ± 1.29 µs | 35% |
| Small, reversed order | 24.07 ± 3.17 µs | 17.32 ± 2.58 µs | 28% |
| Wide, first order | 978 ± 134 µs | 622 ± 102 µs | 36% |
| Wide, reversed order | 977 ± 85 µs | 571 ± 46 µs | 42% |

These comparisons used the same five-process/five-value settings, pinned to CPU
1, and repeated with workload order reversed. Both rounds retain all samples in
`.benchmarks/speed-review-building.json` and
`.benchmarks/speed-review-building-reversed.json`, with adjacent logs. Variation
remains substantial (8–16% standard deviation). The consistent direction supports
this candidate within the measured fixtures, not a precise universal speedup.
Source hashes now include the benchmark itself as well as the measured modules.

A separate 500-example Hypothesis comparison found identical serialized results
and independent ownership after input mutation. Fixed checks also covered ties,
extreme finite scores, both 32×8 and 1×256 candidate layouts, NaN/infinities, and
score-count mismatches. Every benchmark worker checks both constructions against
the fixture before timing. These checks support the experiment; they do not
replace real-model evaluation.

The priorities from this review are:

1. **Result assembly:** test the candidate in the real-model path. Its observed
   saving is 0.36–0.41 ms per wide result. This cannot be subtracted from the
   prebuilt-result client or HTTP timings above.
2. **Response validation:** still the largest measured client component. The
   earlier experiment moving numeric constraints into Pydantic remains deferred
   for real-model evaluation; validation and caller-data snapshots stay enabled.
3. **Preparation and tiny allocation changes:** low priority against these
   larger costs. Preparation takes about 10 µs even for the wide fixture.

`benchmarks/bench_overhead.py` owns the temporary assembly comparison. After
real-model evaluation accepts or rejects it, remove the prototype or replace it
with the accepted production operation. The library code and numerical recipe
remain unchanged. Profiles are saved as `.benchmarks/speed-review-*.prof`, with
readable `*-profile.txt` summaries. To repeat one construction measurement:

```sh
uv run --extra bench -m benchmarks.bench_overhead --workload wide.build_result \
  -p 5 -n 5 -w 2 --min-time 0.05 --affinity 1 -o .benchmarks/build-current.json
uv run --extra bench -m benchmarks.bench_overhead --workload wide.build_result_once \
  -p 5 -n 5 -w 2 --min-time 0.05 --affinity 1 -o .benchmarks/build-candidate.json
```

## SDK interface overhead check

After adding model inspection and sharing client transport handling, `wide.client`
was compared with the pushed `4c26ed8` baseline on Python 3.13. The same workload
used three processes, five measured values, two warmups, a 50 ms minimum, and CPU
affinity 1. Before: **1.039 ± 0.038 ms**; after: **1.089 ± 0.145 ms** (mean ±
standard deviation). Pyperf reported the difference as not statistically
significant. The increased variation remains visible; this is support within
one sampled client workload, not proof of unchanged latency or a speedup claim.

Raw results are `.benchmarks/interface-client-before.json` and
`.benchmarks/interface-client-after.json`. All 20 workload smoke checks completed
in `.benchmarks/interface-smoke.json`. Runtime loading, encoder execution, and
projection code remained byte-for-byte unchanged from that baseline.

Choose an available CPU on another machine. Keep latency runs separate from
profiles and other checks; reverse the workload order for a repeat comparison.

## Optional diagnostics overhead — 2026-09-24

The HTTP harness supports `--diagnostics off` (default), `json` (the CLI's
buffered writer targeting the null device), and `unavailable` (the same writer
with a sink that raises on every write). These are benchmark modes; the service
CLI accepts only `off` and `json`. For example:

```sh
uv run --extra bench -m benchmarks.bench_overhead --workload wide.http --diagnostics json \
  -p 3 -n 5 -w 2 --min-time 0.05 --affinity 1 -o .benchmarks/diagnostics-json.json
```

Compared with `4ff3ab1` in the same Python 3.13.15 environment, using three
worker processes, five values per process, two warmups, and a 50 ms minimum:

| HTTP workload | Before request IDs/logging | Diagnostics off | JSON writer | Unavailable sink |
| --- | ---: | ---: | ---: | ---: |
| Small | 1.11 ± 0.06 ms | 1.11 ± 0.06 ms | 1.50 ± 0.11 ms | 1.50 ± 0.07 ms |
| Wide | 1.51 ± 0.13 ms | 1.55 ± 0.06 ms | 1.86 ± 0.19 ms | 1.92 ± 0.23 ms |

Values are means ± sample standard deviations. Pyperf found neither off-mode
comparison statistically significant. Enabled logging adds about 0.35–0.41 ms
against the corresponding baseline means in this fixture. Stability warnings,
including 10–12% variation in the wide enabled runs, limit precision. This is
TestClient scheduling, JSON events, and a fixed model; it excludes network and
model execution and is not a production latency/capacity claim.

Separate `--tracemalloc --loops 100 -p 2 -n 3 -w 1` runs of `wide.http` measured
traced Python allocation peaks of 3,400 KiB before, 3,400 KiB off, 3,412 KiB with
JSON, and 3,382 KiB with an unavailable sink. These include the harness and are
neither allocations per request nor resident memory; the lower failed-sink value
does not establish a memory improvement. A deterministic blocked-sink test
separately verifies the queue bound, dropped-record count, and bounded shutdown.

Raw timings, allocation reports, and logs are retained locally in
`.benchmarks/diagnostics-*.{json,log}`; the original source is
`/tmp/kayak-diagnostics-before`. Reports hash the measured diagnostic modules
alongside the existing source and environment metadata.
