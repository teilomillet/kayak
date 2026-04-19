# Latent-Proxy Primitive Shape Sweep

## Objective

Measure the new latent-proxy primitive layer on deterministic synthetic shapes so
the next optimization target is chosen from evidence instead of extrapolating
from only the two real one-block artifacts.

This step intentionally focused on:

- query vector budget
- latent dimension
- projection block count

It held the other axes fixed:

- input vector dim `128`
- document count `128`
- candidate shortlist `k=10`

## Why This Sweep Was Needed

The real artifact-backed projection profile already showed:

- projection dominates latent-proxy stage-1 cost on the measured real slices
- the current real artifacts are all one-block projections

That still left one open question:

- if future latent-proxy families use more than one block, does the current
  primitive shape stay reasonable?

## New Benchmark Entry Point

Added:

- `benchmarks/profile_latent_proxy_primitives_sweep.mojo`

Added `pyproject.toml` commands:

- `bench_profile_latent_proxy_primitives_sweep_raw`
- `bench_profile_latent_proxy_primitives_sweep`

The benchmark writes:

- `.cache/kayak/profile_latent_proxy_primitives_sweep.tsv`

## Baseline Sweep Result

Measured synthetic profiles:

- `q8_lat256_b1_docs128`
- `q32_lat256_b1_docs128`
- `q32_lat1024_b1_docs128`
- `q32_lat2048_b1_docs128`
- `q32_lat2048_b2_docs128`
- `q64_lat2048_b1_docs128`

Representative baseline numbers from the initial run:

- `q32_lat256_b1_docs128`
  - projection: `0.00020483673469387756`
  - segment hits: `5.388362068965518e-06`
- `q32_lat1024_b1_docs128`
  - projection: `0.0007311304347826087`
  - segment hits: `2.3089566020313944e-05`
- `q32_lat2048_b1_docs128`
  - projection: `0.001391222222222222`
  - segment hits: `4.9953046953046954e-05`
- `q32_lat2048_b2_docs128`
  - projection: `0.011066625`
  - segment hits: `4.990219560878243e-05`
- `q64_lat2048_b1_docs128`
  - projection: `0.00285634375`
  - segment hits: `5.065856129685917e-05`

## Verified Interpretation

### 1. Projection still dominates

For every non-tiny shape in the sweep, projection remained the dominant share of
`projection + shortlist hits`.

The smallest `q8_lat256_b1_docs128` row was too small and noisy to interpret
precisely. The larger rows were stable enough to support the conclusion.

### 2. One-block cost scales as expected

Holding document count fixed:

- moving from latent `256` to `1024` at `q32` raised projection time from about
  `0.205 ms` to about `0.731 ms`
- moving from latent `1024` to `2048` at `q32` raised projection time from
  about `0.731 ms` to about `1.391 ms`
- moving from `q32` to `q64` at latent `2048` raised projection time from about
  `1.391 ms` to about `2.856 ms`

This is consistent with projection cost being driven mainly by query-token count
and row-wise dot-product work.

### 3. Two-block cost is the real synthetic outlier

At latent `2048` and `q32`:

- one-block projection: `0.001391222222222222`
- two-block projection: `0.011066625`

That is about:

- `0.011066625 / 0.001391222222222222 = 7.95493040752732x`

So the next synthetic hotspot was clear:

- multi-block projection, not shortlist-hit construction

## Optimization Attempt

### Attempt: multi-block scratch reuse

Hypothesis:

- the multi-block path might be losing too much time to per-token temporary list
  allocation and copying

Implemented:

- `MutableLatentQueryProjectionScratch`
- `linear_block_output_into(...)`
- `layer_normalized_output_into(...)`
- `projected_query_token_for_block_into(...)`
- `build_query_latent_proxy_vector_multi_block_with_scratch(...)`

Guardrail:

- `tests/test_latent_proxy_projection.mojo`
  - `test_multi_block_scratch_path_matches_reference_multi_block_path`

This verified the scratch path matched the older allocative multi-block
reference implementation numerically on a two-block fixture.

## Optimization Result

The optimization did **not** help.

Rerun on the synthetic two-block hotspot:

- baseline `q32_lat2048_b2_docs128` projection:
  `0.011066625`
- scratch-path-selected `q32_lat2048_b2_docs128` projection:
  `0.0113394375`

Projection delta:

- `0.0113394375 / 0.011066625 = 1.0246515218952576`

So the scratch version was about `2.47%` slower on the target profile.

Total `projection + shortlist hits` also worsened:

- baseline total:
  `0.01119746875`
- scratch-path-selected total:
  `0.0116828125`

Total delta:

- `0.0116828125 / 0.01119746875 = 1.0433443480942343`

So the end-to-end primitive pair was about `4.33%` slower there.

## Decision

Do **not** select the scratch-reuse path publicly.

The public generic path stays on the older reference multi-block
implementation.

Reason:

- the added complexity did not buy a measured speedup on the target shape

The scratch helper remains in-tree only as an experimental comparison path.

## Conclusion

This sweep produced two useful outcomes:

1. It validated the next hotspot:
   - multi-block projection is where synthetic latent-proxy cost blows up
2. It debunked one plausible but wrong fix:
   - allocation reuse alone is not enough to make the two-block path faster

The next sound direction, if multi-block latent-proxy becomes strategically
important, is no longer "reuse scratch and hope". It has to be a more
substantial compute-facing change such as:

- flatter projection kernels
- batched matrix-style execution
- or a different multi-block projection design
