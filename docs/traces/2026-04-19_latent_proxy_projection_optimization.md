# Latent-Proxy Projection Optimization

## Objective

Profile the native latent-proxy runtime, identify the hot path with direct
evidence, implement one or two bounded optimizations, and re-measure the real
native task benchmarks before claiming a latency story.

This trace follows the earlier native collection runtime trace:

- `docs/traces/2026-04-19_native_latent_proxy_collection_runtime.md`

That earlier trace showed that the native latent-proxy path was slower than
native exact on the measured real slices. The goal here was to explain and
test that bottleneck instead of guessing.

## New Benchmark Entry Point

Added a focused microbenchmark in:

- `kayak/benchmarks/latent_proxy_projection_profile.mojo`
- `python/kayak_bridge/latent_proxy_projection_profile.py`
- `python/scripts/bench_latent_proxy_projection_profile.py`

Reason:

- the end-to-end native benchmark only showed that latent-proxy was slow
- it did not separate query projection cost from proxy-document scan cost
- the new benchmark measures:
  - selected projection path
  - generic multi-block projection path
  - single-block projection path
  - proxy-document scan with preprojected queries
  - projection plus scan

## Baseline Evidence

Commands:

```bash
PYTHONPATH=python pixi run python python/scripts/bench_latent_proxy_projection_profile.py \
  --task .cache/kayak/r2med_biology_real_subset/python_task.json \
  --artifact-root .cache/kayak/lemur_upstream_r2med_biology_real/artifact \
  --output .cache/kayak/lemur_upstream_r2med_biology_real/projection_profile_before.json
```

```bash
PYTHONPATH=python pixi run python python/scripts/bench_latent_proxy_projection_profile.py \
  --task .cache/kayak/lemb_narrativeqa_real_subset/python_task.json \
  --artifact-root .cache/kayak/lemur_upstream_lemb_narrativeqa_real/artifact \
  --output .cache/kayak/lemur_upstream_lemb_narrativeqa_real/projection_profile_before.json
```

Measured baseline:

### R2MED

- `mean_projection_seconds = 0.004807375`
- `mean_scan_seconds = 0.00028425`
- `mean_projection_plus_scan_seconds = 0.00514`

Projection share of projection+scan:

- `0.004807375 / 0.00514 = 93.53%`

### LEMB

- `mean_projection_seconds = 0.00486`
- `mean_scan_seconds = 0.000573625`
- `mean_projection_plus_scan_seconds = 0.005356625`

Projection share of projection+scan:

- `0.00486 / 0.005356625 = 90.73%`

Verified interpretation:

- projection dominates the latent-proxy stage-1 cost on both real slices
- optimizing the document scan first would have targeted the wrong bottleneck

## Shape Verification

The real trained artifacts used here all reported:

- `projection_block_count = 1`
- `projection_block_0_order_kind = linear_norm_activation`
- `projection_block_0_activation_kind = gelu`
- `projection_block_0_layer_norm_affine = 1`

Reason this matters:

- it justified testing a single-block optimization path
- but only after verifying that the actual artifacts matched that shape

## Optimization Loop

### Attempt 1: single-block fused path

Implemented a fused single-block path in `kayak/index/latent_proxy.mojo` to
remove some per-token list churn and intermediate allocations.

Result:

- semantically correct
- not faster than the generic path after measurement

This attempt was kept as an internal benchmarkable helper, but it was not left
as the selected public path.

Reason for not selecting it:

- the benchmark showed it did not beat the generic implementation

### Attempt 2: SIMD-backed linear projection

Changed the latent projection linear step to use the existing SIMD-aware
`dot_product(...)` primitive instead of a scalar inner multiply-accumulate loop
over all `128` input dimensions.

Reason:

- the projection is dominated by repeated row-wise dot products
- the repo already has a SIMD-aware `dot_product(...)` for `Float32`
- using that primitive is smaller and safer than inventing a new bespoke SIMD
  kernel first

This changed both:

- the generic multi-block projection path
- the single-block helper path

### Attempt 3: select the faster kernel

After the SIMD change, the generic path was still slightly faster than the
single-block helper on both measured real slices.

Measured after SIMD:

#### R2MED

- `mean_projection_generic_seconds = 0.001433328125`
- `mean_projection_single_block_seconds = 0.001513333984375`

#### LEMB

- `mean_projection_generic_seconds = 0.001415048828125`
- `mean_projection_single_block_seconds = 0.001480591796875`

Decision:

- route `build_query_latent_proxy_vector(...)` back to the generic multi-block
  path

Reason:

- the generic path is the measured faster implementation on both real slices
- keeping the slower specialized path selected would waste part of the SIMD win

## Post-Optimization Microbenchmark

Commands:

```bash
PYTHONPATH=python pixi run python python/scripts/bench_latent_proxy_projection_profile.py \
  --task .cache/kayak/r2med_biology_real_subset/python_task.json \
  --artifact-root .cache/kayak/lemur_upstream_r2med_biology_real/artifact \
  --output .cache/kayak/lemur_upstream_r2med_biology_real/projection_profile_compare_v2.json
```

```bash
PYTHONPATH=python pixi run python python/scripts/bench_latent_proxy_projection_profile.py \
  --task .cache/kayak/lemb_narrativeqa_real_subset/python_task.json \
  --artifact-root .cache/kayak/lemur_upstream_lemb_narrativeqa_real/artifact \
  --output .cache/kayak/lemur_upstream_lemb_narrativeqa_real/projection_profile_compare_v2.json
```

### R2MED after optimization

- `mean_projection_seconds = 0.001433328125`
- `mean_scan_seconds = 0.000294162109375`
- `mean_projection_plus_scan_seconds = 0.001815373046875`

Speedups vs baseline:

- projection speedup:
  `0.004807375 / 0.001433328125 = 3.354588452772858`
- projection+scan speedup:
  `0.00514 / 0.001815373046875 = 2.831373975089056`

### LEMB after optimization

- `mean_projection_seconds = 0.001415048828125`
- `mean_scan_seconds = 0.000580173828125`
- `mean_projection_plus_scan_seconds = 0.00207328515625`

Speedups vs baseline:

- projection speedup:
  `0.00486 / 0.001415048828125 = 3.434518269263901`
- projection+scan speedup:
  `0.005356625 / 0.00207328515625 = 2.5836412245813087`

## End-to-End Native Benchmark Re-run

### R2MED

Command:

```bash
PYTHONPATH=python pixi run python python/scripts/bench_native_latent_proxy_task.py \
  --task .cache/kayak/r2med_biology_real_subset/python_task.json \
  --artifact-root .cache/kayak/lemur_upstream_r2med_biology_real/artifact \
  --output .cache/kayak/lemur_upstream_r2med_biology_real/native_benchmark_k10_k20_after_v2.json \
  --candidate-ks 10,20
```

Exact baseline from the earlier native trace:

- `mean_search_seconds = 0.001949125`
- `mean_ndcg_at_k = 0.8646255554228253`

Final latent-proxy measurements:

#### `candidate_k = 10`

- `mean_search_seconds = 0.00198125`
- `mean_candidate_generation_seconds = 0.001818625`
- `mean_ndcg_at_k = 0.864296376016278`
- `mean_candidate_recall_at_final_k = 0.7875000000000001`
- time ratio vs native exact:
  `0.00198125 / 0.001949125 = 1.0164817546334894`
- speedup vs pre-optimization native latent-proxy:
  `0.00536925 / 0.00198125 = 2.710031545741325`

#### `candidate_k = 20`

- `mean_search_seconds = 0.00193075`
- `mean_candidate_generation_seconds = 0.001785125`
- `mean_ndcg_at_k = 0.8646255554228253`
- `mean_candidate_recall_at_final_k = 0.9625`
- time ratio vs native exact:
  `0.00193075 / 0.001949125 = 0.990572692875008`
- speedup vs pre-optimization native latent-proxy:
  `0.005397 / 0.00193075 = 2.7952868056454747`

Interpretation:

- `candidate_k = 20` now achieves exact measured quality on the primary metric
  while being slightly faster than native exact on this real slice
- stage-1 artifact size remains much smaller than exact packed storage:
  `2,540,892` bytes vs `14,586,253` bytes

### LEMB

Command:

```bash
PYTHONPATH=python pixi run python python/scripts/bench_native_latent_proxy_task.py \
  --task .cache/kayak/lemb_narrativeqa_real_subset/python_task.json \
  --artifact-root .cache/kayak/lemur_upstream_lemb_narrativeqa_real/artifact \
  --output .cache/kayak/lemur_upstream_lemb_narrativeqa_real/native_benchmark_k10_after_v2.json \
  --candidate-ks 10
```

Exact baseline from the earlier native trace:

- `mean_search_seconds = 0.003977625`
- `mean_ndcg_at_k = 0.375`

Final latent-proxy measurement at `candidate_k = 10`:

- `mean_search_seconds = 0.002214625`
- `mean_candidate_generation_seconds = 0.001953125`
- `mean_ndcg_at_k = 0.375`
- `mean_candidate_recall_at_final_k = 0.3625`
- time ratio vs native exact:
  `0.002214625 / 0.003977625 = 0.556770686024952`
- speedup vs pre-optimization native latent-proxy:
  `0.005530125 / 0.002214625 = 2.497093187334199`

Interpretation:

- the optimized latent-proxy path is now substantially faster than native exact
  on this slice while preserving the same measured task quality
- stage-1 artifact size remains much smaller than exact packed storage:
  `3,984,617` bytes vs `152,898,326` bytes

## Verification Commands

Mojo semantic checks:

- `pixi run mojo -I . tests/test_latent_proxy_projection.mojo`
- `pixi run mojo -I . tests/test_latent_proxy_stage.mojo`
- `pixi run mojo -I . tests/test_collection_mirror_latent_proxy.mojo`

Python end-to-end bridge check:

- `PYTHONPATH=python pixi run python -m unittest python.tests.test_native_latent_proxy_task_benchmark`

All passed locally on 2026-04-19 after the final kernel selection change.

## Conclusion

The original native slowdown claim has now been debunked for the optimized
kernel on the measured real slices.

What was true before optimization:

- native latent-proxy was slower than native exact
- projection was the dominant hot path

What is true after optimization:

- projection time is roughly `3.35x` faster on R2MED and `3.43x` faster on LEMB
- end-to-end native latent-proxy is roughly `2.71x` to `2.80x` faster than its
  earlier native version on R2MED
- end-to-end native latent-proxy is roughly `2.50x` faster than its earlier
  native version on LEMB
- R2MED `candidate_k = 20` now matches exact measured quality and is slightly
  faster than native exact
- LEMB `candidate_k = 10` now matches exact measured quality and is much
  faster than native exact

The justified story now is:

- Kayak can load a trained latent-proxy artifact into its native collection
  runtime
- the projection kernel was the bottleneck
- replacing the scalar projection inner loop with the existing SIMD-aware
  `dot_product(...)` primitive changed the end-to-end result materially
- the optimized native latent-proxy path now supports a credible quality,
  latency, and storage story on real slices
