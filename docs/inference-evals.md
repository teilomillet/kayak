# Local inference evaluation and optimization

For a packaged dataset benchmark, use [`kayak.eval` with BANKING77](evaluation.md).
It measures classification quality across fixed dataset splits. This inference
loop instead uses repeated process trials to investigate runtime changes on a
small, fixed set of interactive requests.

Use this loop when changing the runtime or choosing a serving configuration.
It targets **interactive decisions**: one request in flight, a resident model,
and equal weight for each fixed workload. It runs manually, outside CI/CD, and
saves local artifacts under `.benchmarks/inference/`. It sends no telemetry and
does not change serving defaults or install a background monitor.

The older [overhead benchmarks](development.md#timing-measurements) isolate
Python and transport costs. The [hardware validator](validation.md) checks the
released heads and full-model plumbing. This suite measures actual inference
and separately checks a small set of labeled decisions.

## What is measured

- Model loading time, separate from request timing. The filesystem/artifact cache
  may be warm; this is not a cold-disk benchmark.
- Every case's first observed call, excluded warmups, and all warm measurements.
  HTTP startup performs its own readiness inference before the case warmups.
- Per-case median, range, coefficient of variation, requests/s, and prepared
  input tokens/s. p95 is shown only with at least 20 measured requests per case;
  even then it is a descriptive sample estimate, not a production latency SLO.
- Every returned choice and score, token counts, and error. Failures and missing
  samples stay in the quality denominator. Labels measure task accuracy; matching
  a baseline's answer measures numerical/behavioral preservation, not accuracy.
- Peak process RSS, CUDA peak tensor allocation, or MPS tensor/driver observations.
  MPS records the maximum of **post-request snapshots**, not a true allocation
  peak. Counters overlap and must not be added. OS memory/swap snapshots are saved
  before and after each run. Memory sampling and JSON writes are outside timers.
- Exact requests, labels, corpus/hash, harness hash, runtime source hash, Git
  state, package versions, hardware, precision, thread settings, relevant
  allocator settings, and deterministic case order seed.

Local accelerator work is synchronized before and after each timed call. HTTP
uses the real public client and a loopback socket with the same resident model.
It includes serialization, dispatch, and response parsing, but not another
machine's network. Setup, warming, checks, and reporting are outside the measured
region. Both paths are sequential; requests/s here is single-client service
rate, not saturated multi-client capacity. CPU/GPU profiling should be a separate
experiment, since instrumentation can change latency.

At batch size 1, repeated ordered candidate token rows reuse the resident model's
last candidate embedding block. Warm measurements with shared candidates therefore
measure reuse; changed candidate blocks and each process's first call include
candidate encoding. Retain both first-call and warm evidence, and include changing
candidates in the workload when that reflects the application. Larger batch sizes
do not reuse candidate embeddings, so a batch sweep also changes this behavior.

## Fixed starter corpus

[`interactive-v1.jsonl`](../benchmarks/data/interactive-v1.jsonl) contains eight
search cases and four confirmation cases. The cases and labels were authored
before running the new evaluator. They include short routing, French/Unicode,
negation, longer context, two questions, up to eight candidates, and a near-tie
knowledge question. This is a **synthetic regression corpus**, not independently
reviewed application data or a claim of general model quality.

Each line has `id`, `split` (`dev` or `holdout`), `tags`, a normal Kayak `request`,
and `expected` mapping every question ID to a candidate ID. IDs must be unique
and every question must be labeled. Supply your own immutable JSONL with
`--suite PATH`; prefer representative, independently labeled requests. Version
changes to inputs, labels, and split membership. Do not rewrite labels to match
the model or remove difficult cases after observing results.

## Establish a baseline

Run from a source checkout or source archive; uv supplies the `serve` and `bench`
extras. Use the appropriate PyTorch build and cached model artifacts. On the tested local Mac:

```sh
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9

uv run --extra serve --extra bench -m benchmarks.inference run \
  --device mps --dtype bfloat16 --batch-size 1 \
  --cache-dir validation/model-cache --local-files-only \
  --warmups 2 --repeats 20 --max-memory-gib 17 \
  --output .benchmarks/inference/baseline-local
```

Keep unused apps closed and run one model process at a time. These process-local
allocator ratios are the tested Mac setup, not portable defaults. The 17 GiB
budget stops further requests after an observed driver allocation exceeds the
budget; it cannot prevent a transient allocation/OOM. On macOS, observed critical
system memory pressure also stops the run. Choose a budget for your own machine.

Use a new output directory each time; existing results are never overwritten.
`run.json` contains raw evidence; `summary.json` contains readable per-case
metrics. The run checkpoints after each round and saves failures. `status=passed`
means execution and exact within-configuration repeatability passed; inspect
`correct / labels` separately for task quality. Repeated labels are correlated
observations of the same cases, not additional independent quality examples.

Add `--transport http` for a real local service measurement. Compare local with
local and HTTP with HTTP; the comparator rejects mixed transports. CPU/CUDA and
compatible local model bundles use the same interface. CUDA behavior of this
new harness has not been established by the Mac runs.

## Bounded hill climbing

Batch size is the first existing runtime lever: it controls how many prepared
state/candidate sequences share an encoder forward pass. This is **within one
request**, not request concurrency. The hypothesis is that fewer encoder calls
can lower latency, at the cost of padding, memory, and possible numerical drift.
The runner measures that tradeoff; it does not assume batching will help.

```sh
uv run --extra serve --extra bench -m benchmarks.inference sweep \
  --device mps --dtype bfloat16 \
  --cache-dir validation/model-cache --local-files-only \
  --batch-sizes 1 2 4 8 --trials 3 --warmups 2 --repeats 10 \
  --max-memory-gib 17 \
  --output .benchmarks/inference/batch-search-001
```

The search is finite and serial. Each trial starts a fresh process; candidate
order is shuffled across rounds. Every process is given a timeout (default 900
seconds, `--trial-timeout`). Crashes and missing reports disqualify the affected
configuration. Worker logs and attempt/exit records are kept beside the reports.

The default comparison gates are conservative and explicit:

- Same corpus, cases, labels, protocol, hardware/environment, model fingerprint,
  precision, and token accounting. Source/configuration must stay fixed within
  each group of trials. Runtime source may differ between baseline and candidate
  groups to allow implementation experiments; the harness must remain unchanged.
- No failed/missing observations or changed choices, including warmups. Maximum
  absolute score drift is **zero** by default (`--score-atol`). Any nonzero budget
  must be justified by the application's numerical requirements before a search,
  not increased afterward merely to admit the fastest result.
- At least three fresh process trials per configuration. The equal-case geometric
  mean speedup must clear a 5% improvement at the lower end of a seeded 95%
  bootstrap interval. Bootstrap units are entire process trials, preserving
  correlated case timings. Three trials is a floor; shared host load, temperature,
  autocorrelation, and selection effects can still bias small experiments.
- No case median may regress more than 10%; observed memory may grow at most 10%.
  Thresholds are configurable and saved in the report. Warm first-call latency,
  load time, and descriptive tails are reported but are not selection objectives.

The best passing search candidate is frozen, then tested in new processes on
holdout cases against batch size 1. Only a second pass produces `confirmed`.
If confirmation fails, the runner does not try the runner-up on the same holdout.
Nothing is automatically installed, committed, or promoted to a serving default.
Even confirmed results apply only to these workloads and this environment.
Repeated reuse of the holdout makes it development data: refresh application
holdouts periodically and make the final deployment check on fresh labeled data.

A batch sweep is a scaling diagnostic. Inspect per-case latency, token rate,
memory, and variation across the batch axis. With fewer than six meaningful
points or a changing padding/work mix, a Universal Scalability Law fit would be
inconclusive; this tool retains raw measurements rather than inventing a fit.
Use evidence from the sweep to select the next change, and profile separately
before claiming a specific kernel, bandwidth, or dispatch bottleneck.

## Monitor and compare over time

```sh
uv run --extra serve --extra bench -m benchmarks.inference history --root .benchmarks/inference

# Each path contains trial directories with run.json files.
uv run --extra serve --extra bench -m benchmarks.inference compare \
  .benchmarks/inference/batch-search-001/search/batch-1 \
  .benchmarks/inference/batch-search-001/search/batch-4 \
  --output .benchmarks/inference/comparison-001.json
```

`history` prints a TSV table of all runs, including failures, with quality,
latency, memory, and report paths. It is an inventory, not a comparison across
incompatible machines or suites. `compare` returns nonzero if any gate fails;
a single run per configuration can show exploratory numbers but cannot pass.
A `sweep` exits successfully when it completes, even if it finds no winner;
inspect `sweep.json` for `confirmed`, `no_winner`, or `confirmation_failed`.

For each optimization: write a hypothesis; freeze workload/thresholds; collect
baseline trials; make one change; collect candidate trials under the same
conditions; compare; confirm on reserved cases; retain both artifacts. Change
one lever at a time. Keep a faster-but-rejected candidate's measurements to
understand score drift or memory tradeoffs, without calling it an accepted win.
