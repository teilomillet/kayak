# Search-Layer Optimization Scorecard

Status: current scorecard
Date: `2026-04-25`

This scorecard tracks optimization work under the narrowed Kayak direction:

- optimize the late-interaction search layer
- keep vector counts explicit
- require measurements before claiming a performance win
- distinguish exact throughput, candidate recall, storage efficiency, service
  trust, and integration-boundary work

The matching product-direction note is
[docs/product_direction.md](product_direction.md).

## Scorecard Axes

Every search-layer optimization should report the axes it touches.

| Axis | Required evidence |
| --- | --- |
| Exact throughput | query vectors, document vectors, document count, backend, layout, latency |
| Competitive speed | matched shapes against FastPlaid, recall vs Kayak exact, build/update time, index bytes, CPU/GPU device |
| Candidate recall per cost | candidate recall against exact full scan, final quality, candidate count, vector budgets |
| Storage/vector budget | bytes per vector, bytes per document, vector-count distribution, quality before/after |
| Operational trust | lifecycle behavior, overload/backpressure, metrics, recovery tests |
| Integration boundary | runnable example, explicit parser/encoder/store boundary, no hidden dispatch |

Reason:

- late-interaction cost depends on vector count and layout, not only document
  count
- candidate-stage speedups are not useful if stage 1 misses the documents that
  exact reranking needs
- hosted-engine changes need operational evidence, not only local kernel timing

## FastPlaid Competitive Speed Track

FastPlaid is the closest current open-source speed reference for Kayak's
late-interaction search lane. LightOn describes it as a Rust engine for
late-interaction retrieval with GPU optimization and mutable indexes; the
FastPlaid README exposes K-means/PQ controls, CPU/GPU device selection, and an
`update(...)` API. Kayak should therefore track FastPlaid on its own turf, but
always report recall against Kayak exact.

Harness:

- `python/scripts/bench_fastplaid_speed_track.py`
- `python/scripts/bench_fastplaid_cpu_pareto.py`
- `pixi run bench_fastplaid_speed_track_kayak_smoke_raw`
- `pixi run bench_fastplaid_speed_track_raw`
- `pixi run bench_fastplaid_speed_track`
- `pixi run bench_fastplaid_cpu_pareto_smoke_raw`
- `pixi run bench_fastplaid_cpu_pareto_raw`
- `pixi run bench_fastplaid_cpu_pareto`
- `pixi run bench_fastplaid_cpu_long_token`

Kayak approximation status:

- `kayak_plaid` is an opt-in benchmark lane, not the default exact path
- current implementation is Mojo-backed: Python only prepares API inputs, while
  sampled centroid assignment, candidate scoring, and exact MaxSim rerank run
  inside the Mojo bridge
- the production target is still to graduate this explicit parameter from a
  benchmark lane into the public search API after larger recall/latency checks

Source references:

- <https://lighton.ai/lighton-blogs/fastplaid>
- <https://github.com/lightonai/fast-plaid>

### CPU Pareto Coverage

Command:

```bash
pixi run bench_fastplaid_cpu_pareto
```

Artifact:

- `.cache/kayak/fastplaid_cpu_pareto_scorecard/summary.json`

Pareto definition:

- scope: per explicit vector-count shape
- maximize: `recall_at_k_vs_kayak_exact`, `query_qps`
- minimize: `index_bytes`

Measured coverage:

| Shape | Total doc vectors | Total query vectors | FastPlaid recall / qps / bytes | Kayak dominating config | Kayak recall / qps / bytes | Kayak qps vs FastPlaid |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `small_128d_16dv_4q_8qv` | `2048` | `32` | `0.7249999999999999` / `245.97031442805334` / `1499796` | `recall` (`candidate_k=160`, `centroids_per_query_vector=32`, `centroid_count=128`) | `1.0` / `3315.603154127818` / `1066968` | `13.479688237328478` |
| `medium_256d_32dv_8q_16qv` | `8192` | `128` | `0.6375` / `99.05178469180652` / `5716542` | `recall` (`candidate_k=160`, `centroids_per_query_vector=32`, `centroid_count=128`) | `0.7500000000000001` / `942.5719203689406` / `4255712` | `9.515950906908893` |
| `token_128d_300dv_4q_50qv` | `38400` | `200` | `0.575` / `56.37008369349361` / `26006643` | `recall` (`candidate_k=160`, `centroids_per_query_vector=32`, `centroid_count=128`) | `1.0` / `57.985721015479534` / `19777248` | `1.028661254625251` |

Interpretation:

- in this scorecard run, FastPlaid is dominated by a Kayak Mojo approximation
  point on all three Pareto axes for every measured CPU shape
- the `token_128d_300dv_4q_50qv` shape is the tightest CPU case; Kayak still
  has higher recall and lower bytes, but the speed margin is only `1.029x`

### Long-Token CPU Candidate Sweep

Command:

```bash
pixi run bench_fastplaid_cpu_long_token
```

Artifact:

- `.cache/kayak/fastplaid_cpu_long_token/summary.json`

Measured shape:

| Field | Value |
| --- | ---: |
| documents | `128` |
| document vectors/document | `300` |
| total document vectors | `38400` |
| queries | `4` |
| query vectors/query | `50` |
| total query vectors | `200` |
| vector dim | `128` |
| top_k | `10` |
| warmup iterations | `1` |
| measurement iterations | `3` |

Result:

| System/config | Candidate k | Query batch mean s | Query q/s | Index bytes | Recall@10 vs Kayak exact | QPS vs FastPlaid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kayak `candidate_96` | `96` | `0.055075652666952614` | `72.62737355448799` | `19777248` | `0.675` | `1.2860867897866648` |
| Kayak `candidate_128` | `128` | `0.06960272233239569` | `57.46901652635865` | `19777248` | `1.0` | `1.0176623407857508` |
| Kayak `candidate_160` | `160` | `0.06955270866577241` | `57.5103411029115` | `19777248` | `1.0` | `1.0183941171036466` |
| Kayak `candidate_192` | `192` | `0.06865026433297317` | `58.266345204424425` | `19777248` | `1.0` | `1.0317814508374057` |
| FastPlaid CPU | n/a | `0.07083206933384645` | `56.47159595390554` | `26006107` | `0.6` | `1.0` |

Interpretation:

- `candidate_96` is the CPU Pareto winner in this long-token sweep: it has
  higher recall than FastPlaid, higher QPS, and lower index bytes
- if requiring `1.0` recall against Kayak exact on this synthetic shape,
  Kayak is now faster than FastPlaid on CPU: the standard task's fastest
  measured full-window run is `1.032x` FastPlaid QPS with lower index bytes
- `candidate_128`, `candidate_160`, and `candidate_192` all cover the full
  document set for this shape, so they use the same exact-rerank fast path;
  small QPS differences between those rows should be treated as measurement
  noise, not algorithmic differences
- a longer confirmation run with `2` warmups and `7` measurements also kept the
  exact-recall point ahead: `candidate_128` reached `58.41599318065928` QPS
  versus FastPlaid `57.187065195684156` QPS (`1.0214896144918426x`)
- a repeated paired confirmation with `2` warmups and `9` measurements per run
  kept the exact-recall point ahead in all `3` runs: `candidate_128` QPS ratios
  versus FastPlaid were `1.1040264918308047`, `1.1116468296064463`, and
  `1.0012816949582095`

Matched-shape CPU smoke:

```bash
pixi run bench_fastplaid_speed_track_raw
```

Artifact:

- `.cache/kayak/fastplaid_speed_track_scorecard/summary.json`

Measured shape:

| Field | Value |
| --- | ---: |
| documents | `256` |
| document vectors/document | `32` |
| total document vectors | `8192` |
| queries | `8` |
| query vectors/query | `16` |
| total query vectors | `128` |
| vector dim | `128` |
| top_k | `10` |
| FastPlaid version | `1.4.6.2110` |
| FastPlaid device | `cpu` |
| FastPlaid nbits | `4` |
| Kayak PLAID candidate_k | `160` |
| Kayak PLAID centroids/query vector | `32` |
| Kayak PLAID centroid count | `128` |

Result:

| System | Query batch mean s | Query q/s | Build s | Index bytes | Recall@10 vs Kayak exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kayak `mojo_exact_cpu` exact | `0.3453409026672792` | `23.165515402928246` | `0.0016379580010834616` | `4196360` | `1.0` |
| Kayak PLAID-style Mojo probe | `0.008783846998994704` | `910.7626761845448` | `0.04264354199767695` | `4255712` | `0.7500000000000001` |
| FastPlaid CPU PLAID/PQ | `0.08756494433206778` | `91.36076155843867` | `0.531085582999367` | `5716494` | `0.6` |

Pairwise:

- Kayak PLAID-style query batch latency ratio vs Kayak exact:
  `0.02543529286902211`
- Kayak PLAID-style QPS ratio vs Kayak exact: `39.3154505886547`
- Kayak PLAID-style index bytes ratio vs Kayak exact:
  `1.0141436864330038`
- FastPlaid query batch latency ratio vs Kayak exact: `0.2535608833351338`
- FastPlaid QPS ratio vs Kayak exact: `3.9438259831201585`
- FastPlaid index bytes ratio vs Kayak exact: `1.3622506172015747`

Interpretation:

- on this synthetic CPU smoke, Kayak's opt-in Mojo PLAID-style probe is faster
  than FastPlaid and has higher recall against Kayak exact
- Kayak's current exact path remains the correctness reference; the fast lane is
  optional and approximate
- the Kayak approximation stores the exact token values needed for rerank, so
  its reported index bytes include exact vectors plus centroid metadata

FastPlaid-token-shape exploratory smoke:

- artifact: `.cache/kayak/fastplaid_speed_track_turf_smoke/summary.json`
- shape: `128` documents, `300` document vectors/document, `4` queries, `50`
  query vectors/query, vector dim `128`, `top_k=10`
- command used one raw measurement iteration without the quiet wrapper, so this
  is a quick falsification check rather than a final benchmark claim
- Kayak PLAID-style Mojo probe: `0.05640716599737061 s` query batch mean,
  `70.91297584754493` QPS, recall@10 vs Kayak exact `0.675`
- FastPlaid CPU PLAID/PQ: `0.07875895899996976 s` query batch mean,
  `50.787872907278214` QPS, recall@10 vs Kayak exact `0.5499999999999999`

Mutable-index smoke:

- artifact: `.cache/kayak/fastplaid_speed_track_fastplaid_update_smoke/summary.json`
- shape: `80` documents, `16` update documents, `16` document vectors/document,
  `4` queries, `8` query vectors/query, `top_k=10`
- FastPlaid update time: `0.5240594580027391 s`
- FastPlaid QPS ratio vs Kayak exact after update: `3.3684699169176335`
- FastPlaid recall@10 vs Kayak exact after update: `0.75`

Caveat:

- these are local synthetic CPU smoke results, not final decision-quality
  claims
- the quiet-wrapper run required `--force` because the host did not stay below
  the default competing-CPU threshold
- the next FastPlaid track run should use larger shapes, real encoded tasks,
  and GPU when available

## Current Baselines

These measurements were collected on `2026-04-25` on the local development
host. Host-load caveat: quiet-wrapper runs used `--force` after short quiet
waits because competing host CPU did not consistently stay below the default
threshold.

### Local Exact Batch MaxSim

Command shape:

```bash
bash scripts/run_bench_quiet.sh --timeout-seconds 5 --force -- \
  env PYTHONPATH=python pixi run python python/scripts/bench_batch_maxsim.py \
    --mode shared_batch \
    --document-count 2000 \
    --document-vector-count 32 \
    --batch-size 4 \
    --query-vector-count 6 \
    --repeats 3 \
    --warmup-runs 1 \
    [--clear-index-payload-cache-per-run]
```

Measured shape:

| Field | Value |
| --- | ---: |
| query layout | `flat_dim128` |
| index layout | `hybrid_flat_dim128` |
| batch size | `4` |
| query vectors/query | `6` |
| documents | `2000` |
| document vectors/document | `32` |
| total index vectors | `64000` |

Result:

| Mode | Quiet-wrapper log | Median run mean (s) |
| --- | --- | ---: |
| forced uncached payload rebuild | `.cache/kayak/bench_quiet/20260425T093215Z` | `1.9359409443325906` |
| cached payload reuse | `.cache/kayak/bench_quiet/20260425T093320Z` | `1.7000958473339172` |

Derived comparison:

- speedup: `1.139x`
- time reduction: `12.18%`

Interpretation:

- bounded Mojo index-payload caching is justified for repeated exact search over
  larger `hybrid_flat_dim128` indexes
- the default small benchmark shape was effectively neutral under the quiet
  wrapper, so the claim should stay scoped to larger repeated-query shapes

### Store-Loaded Exact Search

Command:

```bash
pixi run bench_python_store_search_raw
```

Measured shape:

| Field | Value |
| --- | ---: |
| store | `directory` |
| backend | `mojo_exact_cpu` |
| layout | `packed` |
| documents | `1000` |
| vectors/document | `16` |
| total vectors | `16000` |
| query vectors | `16` |
| candidate window | `50` |

Result:

| Measurement | Mean seconds |
| --- | ---: |
| full exact search | `0.0010641249999025603` |
| candidate-window exact search | `0.0002877167000406189` |

Interpretation:

- this is an exploratory raw baseline, not a quiet-host claim
- it confirms the scorecard needs to distinguish full-index exact throughput
  from candidate-window exact reranking

### Planner Smoke Candidate Stage

Command:

```bash
pixi run bench_real_subset_planner_benchmark_smoke_raw
```

Output:

- `.cache/kayak/public_planner_benchmark_smoke.json`

Measured slice:

| Field | Value |
| --- | ---: |
| dataset | `beir/scifact/test` |
| slice | `scifact_real_subset` |
| queries | `6` |
| documents | `55` |
| vectors | `9639` |
| vector dim | `128` |
| final_k | `10` |
| candidate_k | `10` |

Selected rows:

| Plan goal | Generator | Mean search s | Candidate recall at final_k | Mean recall at k | Success at k |
| --- | --- | ---: | ---: | ---: | ---: |
| `exact_only` | `exact_full_scan` | `0.0008985` | `1.0` | `0.8333333333333334` | `0.8333333333333334` |
| `latency_first` | `document_proxy` | `0.00027833333333333334` | `0.5333333333333333` | `0.8333333333333334` | `0.8333333333333334` |
| `balanced` | `centroid_postings_imputed_flat` | `0.0005823333333333334` | `0.3` | `0.3333333333333333` | `0.3333333333333333` |

Interpretation:

- `document_proxy` remains the useful cheap stage-1 baseline on this smoke
  slice
- the candidate-recall gap still matters even when final judged quality happens
  to match exact on this tiny slice
- centroid-family work should keep exact-reference candidate recall in the
  scorecard before being promoted

### Prepared Exact Runtime

Command:

```bash
PYTHONPATH=python pixi run python python/scripts/bench_prepared_exact_runtime.py \
  --task .cache/kayak/browsecomp_plus_real_subset/python_task_gold.json \
  --lane-counts 1 \
  --worker-counts 1,2 \
  --scoring-modes default_auto \
  --warmup-iterations 1 \
  --measurement-iterations 3 \
  --request-pool-count 16 \
  --max-batch-size 16 \
  --max-batch-wait-ms 2 \
  --output .cache/kayak/prepared_exact_runtime_scorecard_small.json
```

Measured slice:

| Field | Value |
| --- | ---: |
| dataset | `Tevatron/browsecomp-plus/gold-slice` |
| documents | `90` |
| distinct queries | `4` |
| request pool | `16` |
| vector dim | `128` |

Result:

| lanes | workers | mean batch s | throughput q/s | avg batch size | avg batch execution ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `1` | `0.022356930332` | `715.662` | `16.000` | `17.814` |
| `1` | `2` | `0.018586750333` | `860.828` | `16.000` | `14.040` |

Interpretation:

- even a small prepared-runtime run supports keeping worker count explicit in
  the optimization scorecard
- RSS fields were `-1` because the sandbox blocked Python's `ps` subprocess;
  timing still completed and results matched the prepared-session reference

## Implemented Optimization: Mojo Index-Payload Cache

Change:

- added a bounded object-identity cache for Mojo index payloads used by repeated
  exact searches over the same `LateIndex`
- changed the hybrid-flat exact dispatch path to reuse that payload
- added a benchmark flag to force uncached behavior for before/after
  comparisons

Files:

- `python/kayak_bridge/mojo_payload_cache.py`
- `python/kayak_bridge/backend_dispatch.py`
- `python/kayak_bridge/batch_dispatch.py`
- `python/scripts/bench_batch_maxsim.py`

Why this was the first implementation target:

- it is Lane 1 exact-throughput work
- it preserves exact scoring semantics
- it targets the repeated-query same-index workload that Kayak already presents
  as a first-class use case
- it avoids touching candidate-generation behavior before the exact baseline is
  cheaper and better instrumented

What is verified:

- cached dispatch still matches existing batch API tests
- the new cache is bounded and identity-scoped
- packaging includes the new runtime bridge module
- larger repeated-query `hybrid_flat_dim128` shape improved by `1.139x` under
  forced quiet-wrapper measurement

What is not verified:

- a universal speedup on small indexes
- a hosted-engine speedup
- memory tradeoff beyond the bounded cache size

## Next Target

Next optimization should stay in Lane 1 unless a new benchmark contradicts this.

Recommended next target:

- add equivalent payload/cache instrumentation for query payloads or move the
  hybrid-flat batch path toward a true batch Mojo entrypoint

Evidence needed before implementation:

- profile or benchmark that separates:
  - query payload conversion
  - index payload conversion
  - Mojo scoring time
  - Python result materialization

Reason:

- after index payload reuse, the default `shared_batch` and `naive_loop` timings
  on the small shape are nearly identical, which suggests remaining overhead is
  not yet isolated well enough for another blind optimization.
