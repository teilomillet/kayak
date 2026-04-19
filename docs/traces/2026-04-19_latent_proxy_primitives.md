# Latent-Proxy Primitives

## Objective

Make the latent-proxy stage-1 path expose explicit Mojo primitives instead of
leaving the runtime boundary implicit inside one execution branch.

The narrow goal was:

- keep the external search-plan contract unchanged
- extract the stage-1 latent-proxy work into named primitives
- route the proxy-family runtime through those primitives
- verify the primitive benchmark path still works on real artifacts

## Why This Was Justified

Before this step, the latent-proxy path already worked and was benchmarked, but
the primitive boundary was still implicit:

- projection lived in `kayak/index/latent_proxy.mojo`
- the proxy-family runtime rebuilt the stage-1 scan loop inline in
  `kayak/planning/execution_proxy_family.mojo`
- the benchmark path measured projection and raw scan cost, but it did not do
  so through an explicit stage-1 primitive surface

That was weaker than the centroid family, which already had first-class stage-1
primitives and a benchmark path tied to them.

## What Changed

Added a new primitive module:

- `kayak/planning/latent_proxy_primitives.mojo`

New primitive surface:

- `ProjectedLatentQuery`
- `project_query_with_latent_proxy(...)`
- `project_query_with_latent_proxy_generic(...)`
- `project_query_with_latent_proxy_single_block(...)`
- `score_projected_latent_query_against_document(...)`
- `sum_projected_latent_query_scores_against_index(...)`
- `segment_hits_for_projected_latent_query(...)`
- `segment_hits_for_latent_proxy(...)`

Design choice:

- `ProjectedLatentQuery` carries `vector_count` explicitly and sets it to `1`

Reason:

- latent-proxy stage 1 is intentionally a single-vector document-aligned proxy
  representation
- the repo requires vector count to stay explicit in APIs and measurements

Runtime rewiring:

- `kayak/planning/execution_proxy_family.mojo` now projects the query through
  `project_query_with_latent_proxy(...)`
- it then builds per-segment shortlist hits through
  `segment_hits_for_projected_latent_query(...)`

Reason:

- this makes the execution boundary align with the primitive boundary
- the runtime no longer reimplements the latent-proxy scan loop inline

Benchmark rewiring:

- `kayak/benchmarks/latent_proxy_projection_profile.mojo` now measures the
  primitive API directly instead of calling the raw projection helper and a
  private scan helper separately

## Verification

### Focused tests

Ran:

- `pixi run mojo -I . tests/test_latent_proxy_primitives.mojo`
- `pixi run mojo -I . tests/test_latent_proxy_projection.mojo`
- `pixi run mojo -I . tests/test_latent_proxy_stage.mojo`
- `pixi run mojo -I . tests/test_collection_mirror_latent_proxy.mojo`
- `PYTHONPATH=python pixi run python -m unittest python.tests.test_native_latent_proxy_task_benchmark`

All passed locally on `2026-04-19`.

### Real primitive benchmark invocations

Ran:

```bash
PYTHONPATH=python pixi run python python/scripts/bench_latent_proxy_projection_profile.py \
  --task .cache/kayak/r2med_biology_real_subset/python_task.json \
  --artifact-root .cache/kayak/lemur_upstream_r2med_biology_real/artifact \
  --output /tmp/kayak-latent-proxy-primitives-r2med.json
```

```bash
PYTHONPATH=python pixi run python python/scripts/bench_latent_proxy_projection_profile.py \
  --task .cache/kayak/lemb_narrativeqa_real_subset/python_task.json \
  --artifact-root .cache/kayak/lemur_upstream_lemb_narrativeqa_real/artifact \
  --output /tmp/kayak-latent-proxy-primitives-lemb.json
```

Observed outputs:

### R2MED

- `mean_projection_seconds = 0.00140958984375`
- `mean_projection_generic_seconds = 0.0014116875`
- `mean_projection_single_block_seconds = 0.001478025390625`
- `mean_scan_seconds = 5.997721822541967e-05`
- `mean_projection_plus_scan_seconds = 0.001474916015625`

### LEMB

- `mean_projection_seconds = 0.001403837890625`
- `mean_projection_generic_seconds = 0.001417478515625`
- `mean_projection_single_block_seconds = 0.001509921875`
- `mean_scan_seconds = 0.000117080078125`
- `mean_projection_plus_scan_seconds = 0.00153601171875`

## Conclusion

This step is complete.

What is now true:

- latent-proxy stage 1 has an explicit primitive layer in Mojo
- the proxy-family runtime uses that primitive layer directly
- the primitive benchmark path now measures the explicit primitive surface
- the real artifact-backed projection profile still behaves consistently with
  the earlier optimization evidence

What is still not claimed:

- that this is the final best long-term primitive boundary for every future
  proxy family
- that the single-block helper should be selected publicly
- that raw primitive timings alone settle future GPU or multi-block design
  choices

The narrower sound conclusion is:

- the latent-proxy stage-1 boundary is now explicit, testable, and benchmarked
  in the same style as the centroid primitive layer
