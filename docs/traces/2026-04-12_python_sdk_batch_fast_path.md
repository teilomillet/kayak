# Python SDK Batch Fast Path Trace

Date: `2026-04-12`

## Scope

This trace records the step where the Python SDK gained a measured batch
fast path for exact Mojo scoring without changing the explicit late-interaction
model.

What changed:
- `LateQueryBatch` remained the public batch abstraction
- `mojo_exact_cpu` batch dispatch now reuses one loaded Mojo module and one
  precomputed index payload across all queries in the batch
- the repo gained a reproducible Python microbenchmark for that path
- the Ordeal SDK smoke path now exercises query batches directly

## Validation

Commands:

```bash
PYTHONPATH=python pixi run python -m unittest python.tests.test_batch_api -v
pixi run test_python_api
pixi run test_python_sdk_ordeal
pixi run test_python_sdk_ordeal_mojo
```

All passed.

Verified:
- batch scoring still matches individual scoring on the NumPy reference path
- Mojo batch scoring now matches NumPy for both packed and
  `hybrid_flat_dim128` layouts
- the Ordeal smoke path finds no SDK batch/layout/backend regressions in either
  the NumPy-only or Mojo-enabled run

## Benchmark Commands

Raw:

```bash
pixi run bench_python_batch_maxsim_naive_raw
pixi run bench_python_batch_maxsim_shared_raw
```

Quiet-wrapper:

```bash
pixi run bench_python_batch_maxsim_naive
pixi run bench_python_batch_maxsim_shared
```

Actual verified quiet-wrapper invocations on this host used:

```bash
bash scripts/run_bench_quiet.sh --timeout-seconds 5 --force -- pixi run bench_python_batch_maxsim_naive_raw
bash scripts/run_bench_quiet.sh --timeout-seconds 5 --force -- pixi run bench_python_batch_maxsim_shared_raw
```

Reason:
- competing host CPU never dropped below the default threshold during the run
- forcing the wrapper preserved the contention snapshots and still gave a
  better comparison than a single uncaptured raw run

Quiet-wrapper log dirs:
- `.cache/kayak/bench_quiet/20260412T151153Z`
- `.cache/kayak/bench_quiet/20260412T151250Z`

## Benchmark Shape

| Parameter | Value |
| --- | ---: |
| `query_layout` | `flat_dim128` |
| `index_layout` | `hybrid_flat_dim128` |
| `batch_size` | `24` |
| `query_vector_count` | `6` |
| `document_count` | `192` |
| `document_vector_count` | `10` |
| `total_index_vectors` | `1920` |

## Result

### Raw single-process benchmark means

| Mode | Mean (s) |
| --- | ---: |
| naive per-query loop | `0.43316446244134565` |
| shared batch dispatch | `0.33616508534178136` |

### Forced quiet-wrapper median run means

| Mode | Median run mean (s) |
| --- | ---: |
| naive per-query loop | `0.4388567123998655` |
| shared batch dispatch | `0.32399859379511325` |

Derived comparison from the forced quiet-wrapper medians:
- speedup: `1.3545x`
- time reduction: `26.17%`

## Interpretation

What the evidence supports:
- reusing the loaded Mojo module and index payload materially reduces
  Python-to-Mojo dispatch overhead for batched exact scoring on this workload
- the optimization preserves score agreement against the NumPy reference path
- the public batch API stays explicit about query counts, layouts, and backend
  choice

What the evidence does not support:
- a claim about end-to-end service latency
- a claim that every layout sees the same speedup
- a claim that the measured absolute times are quiet-host quality

Important caveat:
- the quiet wrapper had to run with `--force` because competing CPU stayed well
  above the default threshold
- that means the medians are the best available controlled comparison from this
  session, but they are still host-contended measurements

Current conclusion:
- keep the batch fast path explicit and opt-in through `query_batch` plus
  `maxsim_batch`
- keep NumPy as the correctness oracle
- treat the current measured win as justified evidence for the shared-payload
  optimization, not as a finished performance story for every future backend
