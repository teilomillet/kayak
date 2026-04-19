# Latent-Proxy Tiled4 Runtime Selection

## Objective

Take the benchmark-only fused multi-row linear result and test whether it
deserves to become the selected runtime kernel in the native latent-proxy path.

This follows two earlier traces:

- `docs/traces/2026-04-19_latent_proxy_linear_tiled4_experiment.md`
- `docs/traces/2026-04-19_latent_proxy_projection_optimization.md`

The key question was:

- does the fused `tiled4` row-reuse kernel improve the **real trained artifact**
  path, or was the synthetic hotspot too artificial to trust for runtime
  selection?

## Implementation

Changed:

- `kayak/index/latent_proxy.mojo`

Added:

- `tests/test_latent_proxy_linear_kernel.mojo`

Also updated the synthetic benchmark entry points to keep using an explicit
reference kernel even after the runtime switched:

- `benchmarks/profile_latent_proxy_linear_storage_experiment.mojo`
- `benchmarks/profile_latent_proxy_linear_tiled4_experiment.mojo`
- `tests/test_latent_proxy_linear_profile_support.mojo`

### Runtime change

The previous row-wise dot-product loop is now preserved as:

- `linear_block_output_reference(...)`

The selected runtime helper is now:

- `linear_block_output(...) -> linear_block_output_tiled4(...)`

Why this shape:

- it preserves the existing block storage and projection semantics
- it avoids a broader latent-proxy storage/schema change
- it keeps the old implementation available as the reference path for tests and
  benchmarks

What was **not** changed:

- the single-block specialized projection helper is still not selected
- the scratch-based multi-block path is still not selected

Reason:

- the earlier real-artifact measurements already showed the generic
  multi-block-orchestrated path was faster than the specialized single-block
  helper
- this step only changes the linear kernel used inside that selected generic
  path

## Verification

Mojo kernel correctness:

```bash
pixi run mojo -I . tests/test_latent_proxy_linear_kernel.mojo
pixi run mojo -I . tests/test_latent_proxy_linear_profile_support.mojo
pixi run mojo -I . tests/test_latent_proxy_projection.mojo
```

Passed:

- `2/2` tests in `test_latent_proxy_linear_kernel.mojo`
- `2/2` tests in `test_latent_proxy_linear_profile_support.mojo`
- `3/3` tests in `test_latent_proxy_projection.mojo`

Native bridge regression check:

```bash
PYTHONPATH=python pixi run python -m unittest python.tests.test_native_latent_proxy_task_benchmark
```

Passed:

- `1` test run
- `1` passed

## Real Artifact Microbenchmark Re-run

Commands:

```bash
PYTHONPATH=python pixi run python python/scripts/bench_latent_proxy_projection_profile.py \
  --task .cache/kayak/r2med_biology_real_subset/python_task.json \
  --artifact-root .cache/kayak/lemur_upstream_r2med_biology_real/artifact \
  --output .cache/kayak/lemur_upstream_r2med_biology_real/projection_profile_tiled4_row_reuse.json
```

```bash
PYTHONPATH=python pixi run python python/scripts/bench_latent_proxy_projection_profile.py \
  --task .cache/kayak/lemb_narrativeqa_real_subset/python_task.json \
  --artifact-root .cache/kayak/lemur_upstream_lemb_narrativeqa_real/artifact \
  --output .cache/kayak/lemur_upstream_lemb_narrativeqa_real/projection_profile_tiled4_row_reuse.json
```

### R2MED

Before this step from `projection_profile_compare_v2.json`:

- `mean_projection_seconds = 0.00151245703125`
- `mean_projection_plus_scan_seconds = 0.001815373046875`

After this step:

- `mean_projection_seconds = 0.00119432421875`
- `mean_projection_plus_scan_seconds = 0.0012544375`
- `mean_projection_generic_seconds = 0.001191265625`
- `mean_projection_single_block_seconds = 0.001522865234375`

Derived change:

- projection speedup vs prior selected path: `1.266x`
- projection+scan speedup vs prior selected path: `1.447x`

### LEMB

Before this step from `projection_profile_compare_v2.json`:

- `mean_projection_seconds = 0.001483326171875`
- `mean_projection_plus_scan_seconds = 0.00207328515625`

After this step:

- `mean_projection_seconds = 0.00120641015625`
- `mean_projection_plus_scan_seconds = 0.0013300703125`
- `mean_projection_generic_seconds = 0.001206642578125`
- `mean_projection_single_block_seconds = 0.001522861328125`

Derived change:

- projection speedup vs prior selected path: `1.230x`
- projection+scan speedup vs prior selected path: `1.559x`

Interpretation:

- the selected generic path remains faster than the single-block helper on both
  real trained artifacts
- the tiled row-reuse kernel improved the real selected path rather than merely
  helping the synthetic benchmark harness

## End-to-End Native Benchmark Re-run

Commands:

```bash
PYTHONPATH=python pixi run python python/scripts/bench_native_latent_proxy_task.py \
  --task .cache/kayak/r2med_biology_real_subset/python_task.json \
  --artifact-root .cache/kayak/lemur_upstream_r2med_biology_real/artifact \
  --output .cache/kayak/lemur_upstream_r2med_biology_real/native_benchmark_k10_k20_tiled4_row_reuse.json \
  --candidate-ks 10,20
```

```bash
PYTHONPATH=python pixi run python python/scripts/bench_native_latent_proxy_task.py \
  --task .cache/kayak/lemb_narrativeqa_real_subset/python_task.json \
  --artifact-root .cache/kayak/lemur_upstream_lemb_narrativeqa_real/artifact \
  --output .cache/kayak/lemur_upstream_lemb_narrativeqa_real/native_benchmark_k10_tiled4_row_reuse.json \
  --candidate-ks 10
```

### R2MED

Previous selected runtime from `native_benchmark_k10_k20_after_v2.json`:

- `candidate_k = 10`: `mean_search_seconds = 0.00198125`
- `candidate_k = 20`: `mean_search_seconds = 0.00193075`

Current selected runtime:

- `candidate_k = 10`
  - `mean_search_seconds = 0.00171325`
  - `mean_candidate_generation_seconds = 0.001612875`
  - `mean_ndcg_at_k = 0.864296376016278`
  - `mean_candidate_recall_at_final_k = 0.7875000000000001`

- `candidate_k = 20`
  - `mean_search_seconds = 0.001714375`
  - `mean_candidate_generation_seconds = 0.001538625`
  - `mean_ndcg_at_k = 0.8646255554228253`
  - `mean_candidate_recall_at_final_k = 0.9625`

Derived change vs prior selected runtime:

- `candidate_k = 10`: `1.156x` speedup
- `candidate_k = 20`: `1.126x` speedup

Comparison against the native exact baseline from the earlier trace:

- native exact: `0.001949125`
- `candidate_k = 10` ratio vs exact: `0.87898`
- `candidate_k = 20` ratio vs exact: `0.87956`

Interpretation:

- both candidate settings are now clearly faster than native exact
- `candidate_k = 20` still matches the earlier exact measured NDCG
- `candidate_k = 10` keeps the same slightly lower NDCG it had before this
  step, but is now materially faster than both exact and the previous
  latent-proxy runtime

### LEMB

Previous selected runtime from `native_benchmark_k10_after_v2.json`:

- `candidate_k = 10`: `mean_search_seconds = 0.002214625`

Current selected runtime:

- `candidate_k = 10`
  - `mean_search_seconds = 0.001982`
  - `mean_candidate_generation_seconds = 0.001758625`
  - `mean_ndcg_at_k = 0.375`
  - `mean_candidate_recall_at_final_k = 0.3625`

Derived change vs prior selected runtime:

- `candidate_k = 10`: `1.117x` speedup

Comparison against the native exact baseline from the earlier trace:

- native exact: `0.003977625`
- `candidate_k = 10` ratio vs exact: `0.49829`

Interpretation:

- the latent-proxy path keeps the same measured quality as the prior selected
  runtime
- it is now about half the latency of native exact on this slice

## Decision

Keep the fused `tiled4` linear kernel selected in the runtime.

Reason:

- direct kernel tests pass against the preserved reference implementation
- the synthetic benchmark still shows the expected multi-row reuse win
- the real trained-artifact projection microbenchmarks improved on both slices
- the end-to-end native task benchmarks improved on both slices
- no measured quality metric regressed relative to the prior selected runtime

## Conclusion

The benchmark-only `tiled4` row-reuse result translated into a real runtime win.

What is now verified:

- the selected latent-proxy runtime path is faster than the prior `v2`
  optimized runtime on both `R2MED` and `LEMB`
- `R2MED candidate_k = 20` remains exact-quality on the measured metric while
  now running clearly faster than native exact
- `LEMB candidate_k = 10` keeps its prior measured quality and is now roughly
  `2x` faster than native exact

So this kernel is no longer just an experiment. It is justified as the current
selected linear kernel for Kayak's latent-proxy stage-1 path.
