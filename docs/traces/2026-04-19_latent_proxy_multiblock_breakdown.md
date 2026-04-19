# Latent-Proxy Multi-Block Breakdown

## Objective

Explain the synthetic multi-block latent-proxy slowdown with direct measurements
 before attempting another production optimization.

This followed the earlier shape sweep:

- `docs/traces/2026-04-19_latent_proxy_primitive_shape_sweep.md`

That sweep showed one clear outlier:

- `q32_lat2048_b2_docs128`

The next sound question was:

- which part of that two-block projection actually dominates?

## New Benchmark Entry Points

Added:

- `benchmarks/profile_latent_proxy_multiblock_breakdown.mojo`
- `benchmarks/profile_latent_proxy_linear_storage_experiment.mojo`

Added `pyproject.toml` commands:

- `bench_profile_latent_proxy_multiblock_breakdown_raw`
- `bench_profile_latent_proxy_multiblock_breakdown`
- `bench_profile_latent_proxy_linear_storage_experiment_raw`
- `bench_profile_latent_proxy_linear_storage_experiment`

## Breakdown Measurement

Profile measured:

- `q32_lat2048_b2_docs128`

This profile has:

- query vectors: `32`
- input dim: `128`
- block 0: `128 -> 1024`
- block 1: `1024 -> 2048`

Command:

```bash
pixi run mojo -I . benchmarks/profile_latent_proxy_multiblock_breakdown.mojo
```

Generated artifact:

- `.cache/kayak/profile_latent_proxy_multiblock_breakdown.tsv`

Measured results:

- block `0`, linear:
  `0.0003713185185185185`
- block `0`, post-linear:
  `0.00034537931034482757`
- block `0`, full block:
  `0.0007246811594202898`

- block `1`, linear:
  `0.0096809375`
- block `1`, post-linear:
  `0.0006866438356164384`
- block `1`, full block:
  `0.0104166875`

## Verified Interpretation

### 1. Block 1 dominates

Comparing full-block times:

- block 1 / block 0:
  `0.0104166875 / 0.0007246811594202898 = 14.37403934496903`

So the second block is about `14.37x` more expensive than the first block on
this hotspot profile.

### 2. Block 1 linear work is the real hotspot

Inside block 1:

- linear share:
  `0.0096809375 / 0.0104166875 = 0.9293625340903748`
- post-linear share:
  `0.0006866438356164384 / 0.0104166875 = 0.06591605799057472`

So about `92.94%` of block-1 cost is the row-wise linear projection itself.

### 3. The two-block cliff is mostly arithmetic, not norm/activation overhead

The first synthetic suspicion after the failed scratch-reuse attempt was that
allocation or normalization overhead might still dominate.

This breakdown debunks that.

What is actually expensive is:

- running `2048` row dot products
- each over an input width of `1024`
- for `32` query vectors

That is the right next target if multi-block latent-proxy matters.

## Flat Row Storage Experiment

Hypothesis:

- if row-list indirection is materially hurting the dominant block-1 linear
  projection, flattening the rows into one contiguous scalar buffer and scoring
  row segments directly might recover a worthwhile fraction of the gap

Command:

```bash
pixi run mojo -I . benchmarks/profile_latent_proxy_linear_storage_experiment.mojo
```

Generated artifact:

- `.cache/kayak/profile_latent_proxy_linear_storage_experiment.tsv`

Measured results:

- block `0`
  - row-list linear: `0.0003784887218045113`
  - flat-rows linear: `0.00033303311258278147`

- block `1`
  - row-list linear: `0.00973159375`
  - flat-rows linear: `0.0096936875`

## Interpretation of the Experiment

### Block 0

Flat rows helped meaningfully:

- `0.00033303311258278147 / 0.0003784887218045113 = 0.879904042395979`

So block 0 improved by about `12.01%`.

### Block 1

Flat rows barely moved the dominant hotspot:

- `0.0096936875 / 0.00973159375 = 0.996104766614495`

So block 1 improved by about `0.39%`.

## Decision

Do **not** treat flat row storage alone as the next production change.

Reason:

- it helps the smaller block
- it barely changes the dominant block-1 hotspot

That means a production layout refactor to flatten latent-proxy rows would not
currently have a strong enough measured payoff on the real synthetic hotspot
that motivated the work.

## Conclusion

The multi-block slowdown is now explained more precisely.

What is verified:

- the synthetic two-block cliff is dominated by the second block
- the second block is dominated by linear row-dot-product work
- layer norm and activation are not the main problem
- flat contiguous row storage alone is not enough to materially reduce the
  dominant block-1 cost

So the next sound optimization, if we decide multi-block latent-proxy is worth
promoting further, has to be heavier than scratch reuse or flat row storage.

The likely next class of experiment is:

- a matrix-vector or GEMV-style kernel
- or a learned projection/layout choice that reduces the `1024 -> 2048`
  second-block compute itself
