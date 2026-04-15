# Trace: LanceDB Indexed Query Controls on Legal and R2MED

## Question

Can the indexed LanceDB instability we observed on the harder legal slice be
explained and controlled by explicit indexed-query settings, without changing
the benchmark structure or conflating storage effects with search effects?

## Why this change was justified

The previous benchmark path in
`python/kayak_bridge/lancedb_benchmark.py` always used LanceDB's default indexed
query behavior. That made two things impossible to audit:

1. whether the legal-slice variance was a probe-budget problem or a rerank problem
2. which indexed settings produced a given artifact

That was a gap in epistemic hygiene. The benchmark was measuring a real effect,
but the indexed query contract was implicit.

## Code change

I added one narrow configuration module:

- `python/kayak_bridge/lancedb_index_controls.py`

and threaded it through the existing benchmark path:

- `python/kayak_bridge/lancedb_benchmark.py`
- `python/scripts/bench_lancedb_multivector.py`
- `python/scripts/bench_lancedb_rebuild_variance.py`
- `python/scripts/bench_task_comparison_suite.py`

The benchmark now exposes and records:

- index build controls:
  - `index_num_partitions`
  - `index_num_sub_vectors`
  - `index_target_partition_size`
- indexed query controls:
  - `indexed_nprobes`
  - `indexed_refine_factor`

The code rejects indexed-only knobs on scan runs so the benchmark cannot
silently ignore them.

## Verification

Targeted tests:

```bash
env PYTHONPATH=python uv run --python 3.11 python -m unittest \
  python.tests.test_lancedb_index_controls \
  python.tests.test_benchmark_variance
```

Result: passed.

## Focused legal diagnosis

### Baseline indexed LanceDB on the legal slice

Artifact:

- `.cache/kayak/legal_rag_bench_real_subset/legal_rag_bench_real_subset_lancedb_ivf_pq_variance.json`

Observed:

- primary metric: `mrr`
- frozen mean indexed quality: `0.09694444444444444`
- mean indexed search time: `0.022955144839943386`

This is the unstable behavior we wanted to explain.

### Probe budget alone: `indexed_nprobes=64`

Artifact:

- `.cache/kayak/legal_rag_bench_real_subset/legal_rag_bench_nprobes64_variance.json`

Observed:

- mean indexed quality: `0.11506944444444445`
- mean indexed search time: `0.027262177772354335`

Interpretation:

- `nprobes=64` helps a little versus the default indexed run
- it does **not** repair the legal instability
- it is also slower than the default indexed setting

So probe budget alone is not the main explanation.

### Query-time refine: `indexed_refine_factor=1`

Artifact:

- `.cache/kayak/legal_rag_bench_real_subset/legal_rag_bench_refine1_variance.json`

Observed:

- mean indexed quality: `0.1675297619047619`
- mean indexed search time: `0.02518377081141807`

Interpretation:

- even a minimal refine step largely closes the legal quality gap

### Query-time refine: `indexed_refine_factor=2`

Artifact:

- `.cache/kayak/legal_rag_bench_real_subset/legal_rag_bench_refine2_variance.json`

Observed:

- mean indexed quality: `0.17722222222222223`
- mean indexed search time: `0.02483741320320405`

Interpretation:

- `refine_factor=2` is materially better than the default indexed path
- it is also materially better than `nprobes=64` alone
- on this slice it kept the indexed lane below scan latency while recovering
  judged quality

## Scorecard reruns with explicit `indexed_refine_factor=2`

I reran the comparison suite in a separate output root so the previous artifacts
stay intact.

### Legal scorecard

Artifacts:

- `.cache/kayak/legal_rag_bench_real_subset/indexed_refine2_suite/legal_rag_bench_real_subset_indexed_refine2_comparison_scorecard.json`
- `.cache/kayak/legal_rag_bench_real_subset/indexed_refine2_suite/legal_rag_bench_real_subset_indexed_refine2_comparison_bundle.json`

Observed:

- Kayak exact: `mrr=0.16666666666666666`, `0.0016393802070524544 s`
- LanceDB scan: `mrr=0.16666666666666666`, `0.029454392206389457 s`
- LanceDB indexed frozen with `refine_factor=2`:
  - `mrr=0.17444444444444443`
  - `0.027550233728834427 s`

Interpretation:

- on the legal slice, explicit refine removes the large quality collapse
- the indexed lane remains a little faster than LanceDB scan
- Kayak exact remains far faster than either LanceDB path

### R2MED scorecard

Artifacts:

- `.cache/kayak/r2med_biology_real_subset/indexed_refine2_suite/r2med_biology_real_subset_indexed_refine2_comparison_scorecard.json`
- `.cache/kayak/r2med_biology_real_subset/indexed_refine2_suite/r2med_biology_real_subset_indexed_refine2_comparison_bundle.json`

Observed:

- Kayak exact: `ndcg=0.8646255554317486`, `0.00174860937113408 s`
- LanceDB scan: `ndcg=0.8646255554317486`, `0.025750755245098844 s`
- LanceDB indexed frozen with `refine_factor=2`:
  - `ndcg=0.8768260111698816`
  - `0.02120311560671932 s`

Interpretation:

- the same refine setting also helps the biomedical slice
- it remains faster than LanceDB scan there as well

## What this does and does not prove

Verified:

- the old benchmark path was under-specifying indexed LanceDB behavior
- query-time refine is the dominant lever among the settings tested here
- probe budget alone is not enough to fix the legal indexed instability
- explicit indexed-query settings should be part of any future LanceDB scorecard

Not claimed:

- that `refine_factor=2` is globally optimal
- that LanceDB scan is "wrong"
- that these small slices fully predict larger production behavior

The judged metrics can favor an ANN candidate set that differs from the exact
vector top-k, so an indexed run beating scan on judged quality is possible on a
small judged subset. That is a property of the benchmark objective, not proof
that exact nearest-neighbor ranking is incorrect.

## Practical conclusion

For these harder slices, the useful operational baseline is no longer
"LanceDB indexed with whatever the default query path does". The benchmark
should treat indexed query controls as first-class configuration and record them
in every artifact.

The next sensible step is to keep `indexed_refine_factor` configurable in any
future LanceDB-vs-Kayak benchmark bundles, rather than benchmarking only the
default LanceDB indexed query behavior.
