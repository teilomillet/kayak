# Latent-Proxy Linear Tiled4 Experiment

## Objective

Test whether the dominant multi-block latent-proxy hotspot can be reduced by
reusing each loaded input chunk across multiple output rows, instead of scoring
one output row at a time.

This follows:

- `docs/traces/2026-04-19_latent_proxy_multiblock_breakdown.md`

That earlier trace established two facts:

- the `q32_lat2048_b2_docs128` outlier is dominated by block `1`
- block `1` is dominated by the `1024 -> 2048` linear projection

The next sound question was:

- does fusing several output rows into one SIMD pass materially help the real
  hotspot, or is flat row storage still the whole story?

## Implementation

Added benchmark-only support in:

- `kayak/benchmarks/latent_proxy_linear_profile_support.mojo`

Added a new experiment entry point:

- `benchmarks/profile_latent_proxy_linear_tiled4_experiment.mojo`

Added a correctness test:

- `tests/test_latent_proxy_linear_profile_support.mojo`

Added `pyproject.toml` commands:

- `bench_profile_latent_proxy_linear_tiled4_experiment_raw`
- `bench_profile_latent_proxy_linear_tiled4_experiment`

Also updated the existing linear-storage benchmark to emit `== section ==` plus
`Mean:` lines so the quiet benchmark wrapper can summarize repeated runs:

- `benchmarks/profile_latent_proxy_linear_storage_experiment.mojo`

## Kernel Shape

The experimental kernel is still benchmark-only. It does **not** change the
runtime-selected latent-proxy path.

What it does:

- flattens each block's linear rows into one contiguous scalar buffer
- loads one SIMD chunk from the input vector
- multiplies that same input chunk into `4` output-row accumulators before
  advancing
- falls back to the scalar-tail path for the remaining dimensions and rows

Why `tiled4`:

- it is a bounded experiment rather than a full GEMV rewrite
- the codebase already uses `tiled4` as a practical SIMD batching shape in
  nearby exact-stage kernels
- it directly tests the claim that input reuse, not row-list indirection alone,
  is the next meaningful lever

## Verification

Correctness check:

```bash
pixi run mojo -I . tests/test_latent_proxy_linear_profile_support.mojo
```

Passed:

- `2` tests run
- `2` passed
- `0` failed

Benchmark command used for the decision-quality summary:

```bash
bash scripts/run_bench_quiet.sh --max-other-cpu 80 -- pixi run bench_profile_latent_proxy_linear_tiled4_experiment_raw
```

Artifacts:

- `.cache/kayak/profile_latent_proxy_linear_tiled4_experiment.tsv`
- `.cache/kayak/bench_quiet/20260419T101852Z/section_summary.tsv`
- `.cache/kayak/bench_quiet/20260419T101852Z/section_summary.txt`

Measurement context:

- profile: `q32_lat2048_b2_docs128`
- query vectors: `32`
- input dim: `128`
- block `0`: `128 -> 1024`
- block `1`: `1024 -> 2048`
- vector scalar: `Float32`

Host note:

- the default quiet threshold of `40` was not reachable on this desktop host
  because steady background activity stayed around the low `60s`
- a stale `profile_latent_proxy_primitives_sweep.mojo` process from this Codex
  session was also found and terminated before the final benchmark run
- the final summary therefore used the quiet wrapper with
  `--max-other-cpu 80`, which matched the verified background-load floor on
  this machine

## Results

Median summary across `3` quiet-wrapper runs:

- block `0`, row-list: `0.0003715925925925926`
- block `0`, flat rows: `0.00032335483870967743`
- block `0`, flat rows tiled4: `0.00020744628099173551`

- block `1`, row-list: `0.00909753125`
- block `1`, flat rows: `0.00890796875`
- block `1`, flat rows tiled4: `0.0035345625`

Derived comparisons:

- block `0`, flat rows vs row-list: `12.53%` lower mean time
- block `0`, tiled4 vs row-list: `43.28%` lower mean time
- block `0`, tiled4 speedup vs row-list: `1.76x`

- block `1`, flat rows vs row-list: `1.77%` lower mean time
- block `1`, tiled4 vs row-list: `61.05%` lower mean time
- block `1`, tiled4 speedup vs row-list: `2.57x`
- block `1`, tiled4 speedup vs flat rows: `2.52x`

Run-to-run range for the dominant block-1 tiled4 kernel:

- min: `0.0035256875`
- median: `0.0035345625`
- max: `0.00357728125`

## Interpretation

### 1. Flat rows still are not the main story

Under the repeated quiet-wrapper summary, block `1` flat rows help only
slightly:

- `1.77%` lower than row-list

That is directionally positive, but still far too small to explain the
multi-block hotspot on its own.

### 2. Reusing input chunks across output rows is materially effective

The same block `1` workload drops from about `0.00910s` to about `0.00353s`
when scored with the fused `tiled4` kernel.

That is large enough to move the optimization from "speculative micro-tweak"
into "production-candidate compute strategy."

### 3. The hotspot is now better explained

The earlier storage-only experiment suggested that row indirection was not the
dominant issue. This experiment sharpens that conclusion:

- the expensive part is not mainly the list-of-lists container shape
- the expensive part is repeatedly reloading the same `1024`-wide input for
  many separate output-row dot products

The `tiled4` kernel materially helps because it changes that reuse pattern.

## Decision

Promote this result from benchmark-only evidence to the next production-facing
candidate.

What is justified now:

- a production latent-proxy linear kernel experiment based on fused multi-row
  compute

What is **not** justified yet:

- claiming a full end-to-end latent-proxy runtime win without wiring the kernel
  into the actual projection path and re-measuring task-level latency

## Next Step

The next sound implementation step is:

- add a production-facing multi-row linear kernel behind a narrow helper
- compare it against the current runtime path on the same latent-proxy tasks
- keep the current path available as the reference until correctness and
  task-level timings are both verified
