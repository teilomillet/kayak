# Native Latent-Proxy Collection Runtime

## Claim

Measure the exported trained LEMUR latent-proxy artifact through Kayak's native
collection runtime instead of the Python reference evaluator, then compare that
native path against the native exact full-scan path on real tasks.

## Implementation

Added a narrow native bridge with two explicit pieces:

- `ensure_one_segment_collection_mirror_with_latent_proxy(...)`
  in `kayak/collections/mirror_latent_proxy.mojo`
  Reason: materialize one packed index plus one latent-proxy sidecar with
  manifest validation, byte-size accounting, and optional text corpus.
- `materialize_native_latent_proxy_task_collection(...)` and
  `build_materialized_collection_search_summary(...)`
  in `kayak/benchmarks/native_latent_proxy_task.mojo`
  Reason: reuse the existing collection runtime and faithfulness summary path
  instead of adding another scorer.

The Python entrypoint is:

- `python/scripts/bench_native_latent_proxy_task.py`

It materializes one collection mirror, then benchmarks one or more
`candidate_k` values through the native Mojo path.

## Validation

### Code checks

- `pixi run mojo -I . tests/test_collection_mirror_latent_proxy.mojo`
- `pixi run mojo -I . tests/test_latent_proxy_stage.mojo`
- `PYTHONPATH=python pixi run python -m unittest python.tests.test_native_latent_proxy_task_benchmark`

All passed locally on 2026-04-19.

### Real task commands

R2MED latent-proxy sweep:

```bash
PYTHONPATH=python pixi run python python/scripts/bench_native_latent_proxy_task.py \
  --task .cache/kayak/r2med_biology_real_subset/python_task.json \
  --artifact-root .cache/kayak/lemur_upstream_r2med_biology_real/artifact \
  --output .cache/kayak/lemur_upstream_r2med_biology_real/native_benchmark_k10_k20.json \
  --candidate-ks 10,20
```

R2MED native exact baseline:

```bash
PYTHONPATH=python pixi run python python/scripts/bench_native_latent_proxy_task.py \
  --task .cache/kayak/r2med_biology_real_subset/python_task.json \
  --artifact-root .cache/kayak/lemur_upstream_r2med_biology_real/artifact \
  --output .cache/kayak/lemur_upstream_r2med_biology_real/native_exact_benchmark_k178.json \
  --candidate-generator-kind exact_full_scan \
  --candidate-ks 178
```

LEMB latent-proxy:

```bash
PYTHONPATH=python pixi run python python/scripts/bench_native_latent_proxy_task.py \
  --task .cache/kayak/lemb_narrativeqa_real_subset/python_task.json \
  --artifact-root .cache/kayak/lemur_upstream_lemb_narrativeqa_real/artifact \
  --output .cache/kayak/lemur_upstream_lemb_narrativeqa_real/native_benchmark_k10.json \
  --candidate-ks 10
```

LEMB native exact baseline:

```bash
PYTHONPATH=python pixi run python python/scripts/bench_native_latent_proxy_task.py \
  --task .cache/kayak/lemb_narrativeqa_real_subset/python_task.json \
  --artifact-root .cache/kayak/lemur_upstream_lemb_narrativeqa_real/artifact \
  --output .cache/kayak/lemur_upstream_lemb_narrativeqa_real/native_exact_benchmark_k355.json \
  --candidate-generator-kind exact_full_scan \
  --candidate-ks 355
```

## Results

### R2MED/Biology

Native materialized latent-proxy stage-1 size:

- `2,540,892` bytes for `178` documents
- exact packed stage-1 size on the same collection path:
  `14,586,253` bytes

Measured native exact baseline:

- `mean_search_seconds = 0.001949125`
- `mean_ndcg_at_k = 0.8646255554228253`

Measured native latent-proxy:

- `candidate_k = 10`
- `mean_search_seconds = 0.00536925`
- `mean_ndcg_at_k = 0.864296376016278`
- `candidate_recall_at_final_k = 0.7875000000000001`
- latency ratio vs native exact: `2.7546976207272493`
- quality ratio vs native exact: `0.9996192809656357`

- `candidate_k = 20`
- `mean_search_seconds = 0.005397`
- `mean_ndcg_at_k = 0.8646255554228253`
- `candidate_recall_at_final_k = 0.9625`
- latency ratio vs native exact: `2.768934778426217`
- quality ratio vs native exact: `1.0`

Comparison against the earlier Python artifact-reference bridge:

- native `k=10` was `9.18320448034426x` slower than
  `.cache/kayak/lemur_upstream_r2med_biology_real/artifact_benchmark_k10.json`
- native `k=20` was `6.4836926240210575x` slower than
  `.cache/kayak/lemur_upstream_r2med_biology_real/artifact_benchmark_k20.json`

### LEMB/NarrativeQA

Native materialized latent-proxy stage-1 size:

- `3,984,617` bytes for `355` documents
- exact packed stage-1 size on the same collection path:
  `152,898,326` bytes

Measured native exact baseline:

- `mean_search_seconds = 0.003977625`
- `mean_ndcg_at_k = 0.375`

Measured native latent-proxy at `candidate_k = 10`:

- `mean_search_seconds = 0.005530125`
- `mean_ndcg_at_k = 0.375`
- `candidate_recall_at_final_k = 0.3625`
- latency ratio vs native exact: `1.390308286980296`
- quality ratio vs native exact: `1.0`

Comparison against the earlier Python artifact-reference bridge:

- native `k=10` was `2.680782322816319x` slower than
  `.cache/kayak/lemur_upstream_lemb_narrativeqa_real/artifact_benchmark_k10.json`

## Conclusion

The native collection-backed benchmark path is now implemented and validated.

The measurement result is not the optimistic one:

- quality matches the earlier artifact-reference story
- stage-1 storage remains much smaller than exact packed storage
- but the current native `latent_proxy` runtime is slower than native
  `exact_full_scan` on both real slices measured here

So the earlier timing distortion story was incomplete. Replacing the Python
reference evaluator with the native runtime did not recover the expected
latency win; instead, it exposed a native stage-1 bottleneck.

## Most Likely Bottleneck

This is a hypothesis from direct code inspection, not yet a profiler result.

`kayak/planning/execution_proxy_family.mojo` builds one latent query proxy with
`build_query_latent_proxy_vector(...)` for every query, and
`kayak/index/latent_proxy.mojo` currently evaluates:

- explicit per-token projection loops
- explicit per-dimension layer norm loops
- explicit activation loops over the full latent dimension

That work is then followed by a per-document dot product loop. The exact path
uses different late-interaction kernels, so the native latent-proxy path may be
losing on projection cost before the cheaper document-level scoring can pay
back.

What would verify this:

- a focused microbenchmark for `build_query_latent_proxy_vector(...)`
- a profiler trace separating projection time from proxy-dot-product time
- one optimized projection kernel or fused projection benchmark against the
  current scalar-loop implementation
