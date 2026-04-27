# 2026-04-27: GPU I8 Candidate-Generation Negative Results

## Claim

The next optimization after no-reference GPU top-k should not be another
candidate-generation hot-path edit unless the edit wins on both the default and
wide sweeps.

Reason: candidate generation is now part of the end-to-end GPU rerank envelope.
A change that helps one shape but hurts another makes the serving boundary less
predictable, even if the code looks locally reasonable.

## Experiments

Three CPU-side candidate-generation follow-ups were tested after the pushed
`2f9ff03` checkpoint:

- preallocating candidate-generation lists with `reserve(...)`
- replacing per-query-vector centroid-score materialization with streamed
  top-centroid heap selection
- bypassing `FlatQueryDim128` materialization in the typed-address candidate
  bridge and scoring centroid proxies directly from the query pointer

Reasons:

- `reserve(...)` tested whether remaining allocation growth was measurable
- streamed centroid selection tested whether avoiding a full centroid-score
  list and second scan would reduce non-full candidate generation
- direct pointer scoring tested whether the typed-address bridge was still
  spending meaningful time copying query tensors into Mojo lists

## Measurement

Commands:

```bash
pixi run env PYTHONPATH=python python -m unittest python/tests/test_gpu_i8_rerank_contract.py python/tests/test_fastplaid_speed_track.py
pixi run profile_gpu_i8_address_serve_sweep
pixi run profile_gpu_i8_address_serve_wide_topk
```

Final reverted-code artifacts:

- default sweep: `.cache/kayak/bench_quiet/20260427T090039Z`
- wide sweep: `.cache/kayak/bench_quiet/20260427T090116Z`
- default report: `.cache/kayak/gpu_i8_address_serve_sweep/summary.json`
- wide report: `.cache/kayak/gpu_i8_address_serve_sweep/wide_topk_summary.json`

Rejected experiment artifacts:

- streamed centroid selection default sweep:
  `.cache/kayak/bench_quiet/20260427T085311Z`
- streamed centroid selection wide sweep:
  `.cache/kayak/bench_quiet/20260427T085335Z`
- direct pointer candidate bridge default sweep:
  `.cache/kayak/bench_quiet/20260427T085650Z`
- reserve-only default sweep:
  `.cache/kayak/bench_quiet/20260427T085837Z`
- reserve-only wide sweep:
  `.cache/kayak/bench_quiet/20260427T085924Z`

## Results

Final reverted-code default sweep:

- status: `ok`
- ok cases: `6 / 6`
- best CPU-candidate-plus-no-reference-top-k ratio:
  `0.14505501998253167`
- worst CPU-candidate-plus-no-reference-top-k ratio:
  `0.6714773519347889`

Final reverted-code wide sweep:

- status: `ok`
- ok cases: `5 / 5`
- best CPU-candidate-plus-no-reference-top-k ratio:
  `0.10323403143036211`
- worst CPU-candidate-plus-no-reference-top-k ratio:
  `0.5103655426245011`

Rejected results:

| experiment | evidence | decision |
| --- | --- | --- |
| dense token reset | widened non-full wide rows; `query_vectors32`, `doc_vectors64`, and `query_batch4` all regressed in the exploratory wide run | removed |
| streamed centroid top selection | default worst envelope rose to about `0.693`; wide worst envelope rose to about `0.526` | removed |
| direct pointer candidate bridge | default worst envelope rose to about `0.686`; the default sweep falsified it before a wide run was needed | removed |
| reserve-only preallocation | wide worst envelope moved slightly lower, but the default envelope stayed within noise and did not create a decision-quality win | removed |

All rejected code paths preserved correctness in the focused tests before they
were removed.

## Interpretation

Verified:

- GPU no-reference top-k is still correct on the default and wide sweeps after
  returning to the pushed implementation
- full-window candidate generation remains effectively removed from the
  measured envelope
- non-full CPU candidate generation remains the visible limiter in the wide
  `query_vectors32`, `doc_vectors64`, and `query_batch4` rows

Debunked:

- list preallocation alone is not a decision-quality optimization
- streaming centroid heap selection is not automatically faster than the
  materialize-then-select path
- avoiding query-list materialization in the current bridge shape does not
  reduce the default sweep envelope

## Decision

Do not keep any code from these follow-up experiments.

Reason: the current evidence favors a larger profiling boundary or a different
primitive, not more local edits inside the same candidate-generation loop.
The next useful step is a dedicated candidate-generation breakdown that reports
centroid scoring, centroid selection, posting accumulation, final candidate
top-k, posting visits, query vector count, and document vector count.

## Validation

Ran:

```bash
pixi run env PYTHONPATH=python python -m unittest python/tests/test_gpu_i8_rerank_contract.py python/tests/test_fastplaid_speed_track.py
pixi run profile_gpu_i8_address_serve_sweep
pixi run profile_gpu_i8_address_serve_wide_topk
```

Observed:

- focused tests: `49 / 49` passed
- final default sweep status: `ok`
- final wide sweep status: `ok`
