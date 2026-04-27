# 2026-04-27: GPU I8 Selected-Posting Candidate Window

## Question

Can the selected-posting accumulation primitive expose a serving-shaped
`candidate_k` window that feeds exact rerank directly, without CPU reference
scores or hidden CPU fallback?

## Change

Added two benchmark-only candidate-window boundaries:

- `score_i8_selected_posting_candidate_positions_addresses`
- `score_i8_selected_posting_dense_scores_addresses`
- `score_i8_selected_posting_dense_candidate_positions_addresses`

The first boundary runs GPU selected-posting accumulation and performs the
`candidate_k` document selection inside the Mojo bridge. The second returns
dense per-document scores from the GPU bridge and performs deterministic
thresholded `argpartition` candidate selection in Python.

Reason: the previous accumulation probe proved exact dense scores and fast
non-full accumulation, but it did not return candidate positions as a reusable
pipeline primitive. This change tests the boundary explicitly before wiring it
into a larger search path.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_candidate_posting_accumulation
```

Artifacts:

- report: `.cache/kayak/gpu_i8_candidate_posting_accumulation/summary.json`
- quiet wrapper: `.cache/kayak/bench_quiet/20260427T163028Z`
- smoke report:
  `.cache/kayak/gpu_i8_candidate_posting_accumulation/dense_candidate_smoke.json`

Status:

- benchmark status: `ok`
- rows: `5 / 5`
- non-full candidate-generation rows: `3`
- GPU target: `nvidia:sm_89`
- device: `NVIDIA GeForce RTX 4070 Ti`

All candidate-window rows validated:

- `candidate_position_agreement = 1.0`
- `candidate_score_delta_max_abs = 0.0`
- `score_delta_max_abs = 0.0` for the dense-score boundary
- `validation_reference_scores_sent_to_extension = false`
- no selected-position or doc-index range violations

The smoke case verified the corrected boundary: `candidate_k=32` returned
`candidate_position_count=32`, not final `top_k=10`.

## Result

Non-full rows:

| case | queries | query vectors | docs | candidate_k | CPU candidate s | naive Mojo candidate s | dense extension s | Python selection s | dense total s | dense total / CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `2` | `32` | `512` | `256` | `0.0004071293333254289` | `0.0119638699979987` | `0.0002754339984676335` | `0.00017268200099351816` | `0.00044811599946115166` | `1.1006723485162417` |
| `doc_vectors64` | `2` | `8` | `512` | `256` | `0.00036807666522994015` | `0.006857362997834571` | `0.00031023799965623766` | `0.00016960600260063075` | `0.0004798440022568684` | `1.3036523300305016` |
| `query_batch4` | `4` | `8` | `512` | `256` | `0.0005700729998352472` | `0.014108201998169534` | `0.00047457500113523565` | `0.000293676999717718` | `0.0007682520008529536` | `1.3476379359748327` |

Summary:

- naive Mojo candidate selection is correct but rejected: `18.63x` to
  `29.39x` of CPU candidate generation on non-full rows
- dense-score plus Python candidate selection is correct and much faster, but
  still costs `1.10x` to `1.35x` of CPU candidate generation on non-full rows
- dense GPU accumulation itself remains fast: the same quiet run reports
  non-full all-measured accumulation at `0.050x` to `0.117x` of CPU candidate
  generation
- full-window rows are not optimization targets because CPU candidate
  generation is intentionally near-zero when `candidate_k == document_count`

## Interpretation

This falsifies the naive serving boundary, not the accumulation primitive.

The selected-posting kernels are still promising: they produce exact dense
scores and the measured accumulation-only envelope is much faster than CPU
candidate generation on non-full rows. The losing part is the boundary that
turns dense scores into a `candidate_k` position window:

- selecting `candidate_k` positions inside the Mojo bridge serializes too much
  work and is decisively slower than CPU
- returning dense scores and selecting in Python removes most of that waste,
  but still loses by about `10%` to `35%` on the target non-full rows
- exact rerank was already shown not to be the bottleneck in
  `docs/traces/2026-04-27_gpu_i8_hybrid_shortlist_rerank_scope.md`

## Decision

Do not wire this candidate-window boundary into the search pipeline yet.

Reason: the primitive is correct, explicit, and useful as a diagnostic, but the
serving-shaped non-full candidate window has not produced a measured win. The
next optimization should attack candidate selection/readback, not exact rerank
and not the selected-posting accumulation kernels.

The next candidate primitive should test one of:

- a parallel segmented candidate selector for dense document scores
- a resident-payload dense-score boundary that avoids per-call payload copies
- a fused selected-posting candidate selector that returns positions without
  materializing or Python-converting the full dense score row

Any of those must keep the same contract: explicit query/document vector
counts, `candidate_k`, no silent CPU fallback, no reference scores passed into
the extension, and exact agreement against the CPU i8 reference.

## Follow-Up

This negative result was superseded by a different boundary, not by changing
the scoring semantics. The accepted follow-up is recorded in
`docs/traces/2026-04-27_gpu_i8_resident_selected_posting_candidate_window.md`.

That follow-up keeps selected-posting metadata resident on GPU and writes dense
scores into caller-owned NumPy `Float32` memory. On the same non-full wide
case family, projected CPU centroid scoring/selection plus resident GPU
selected-posting candidate generation measured `0.564x` to `0.663x` of CPU i8
candidate generation while preserving `candidate_position_agreement = 1.0`.

Reason: this trace remains useful because it falsifies the Python-list and
serial/block-selector boundaries. The accepted primitive is the resident
typed-output boundary.
