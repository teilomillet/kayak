# 2026-04-27: GPU I8 Candidate Window And Positive Centroid Tests

## Claim

After unordered candidate windows, the next candidate-generation questions are:

- can the internal GPU pipeline shrink `candidate_k` without losing recall?
- can candidate generation skip selected centroid postings whose proxy score is
  non-positive?

Reason: `candidate_k` controls final candidate top-k work and GPU rerank work,
while non-positive centroid postings may add irregular posting work without
helping the retained candidate set.

## Candidate Window Reduction

Exploratory CPU FastPlaid comparisons tested `candidate_k=128` and `192` on
the three wide non-full policy rows.

Artifacts:

- `candidate_k=128` raw report:
  `.cache/kayak/gpu_i8_fastplaid_policy_compare/window_k128_cpu_summary.json`
- `candidate_k=192` raw report:
  `.cache/kayak/gpu_i8_fastplaid_policy_compare/window_k192_cpu_summary.json`

Results:

| variant | status | minimum Kayak recall delta vs FastPlaid | top-k agreement |
| --- | ---: | ---: | ---: |
| `candidate_k=128` | `ok`, `3 / 3` | `-0.4499999999999999` | `1.0` |
| `candidate_k=192` | `ok`, `3 / 3` | `-0.1999999999999999` | `1.0` |

Decision: do not add a broad smaller-window policy.

Reason: lower `candidate_k` was faster, but it lost recall on
`doc_vectors64` and `query_batch4`. A candidate-window policy cannot be
accepted only because one shape, `query_vectors32`, survives.

## Positive Centroid Postings

Added an opt-in benchmark-only path:

- `kayak/search/plaid_i8_positive_centroid_candidates_dim128.mojo`
- `KayakPlaidApproxIndex.i8_candidate_positions_batch_positive_centroids_unordered(...)`
- `--kayak-i8-positive-centroids-only`

The path is unordered-only and internal. It skips postings for selected
centroids with `centroid_score <= 0`.

Reason: this tests a concrete proxy-scoring hypothesis without changing public
search behavior or the default internal GPU pipeline.

Exploratory raw artifact:

- `.cache/kayak/gpu_i8_fastplaid_policy_compare/positive_centroids_cpu_summary.json`

Quiet-wrapper artifact:

- `.cache/kayak/bench_quiet/20260427T120041Z`

The quiet-wrapper run is not usable as performance evidence because the GPU
probe returned `partial_gpu_unavailable` after an NVML initialization warning.
The raw run below is exploratory, not decision-quality quiet evidence.

Raw CPU FastPlaid policy results:

- status: `ok`, `3 / 3` rows
- minimum top-k position agreement: `1.0`
- minimum Kayak recall delta versus FastPlaid:
  `1.1102230246251565e-16`
- mean scoped envelope / FastPlaid batch: `0.03814584404513458`
- mean CPU candidate-generation share of scoped envelope:
  `0.6982745254523847`

Baseline unordered candidate windows versus positive-centroid candidate
windows on the same CPU rows:

| case | baseline envelope s/window | positive envelope s/window | baseline candidate s/window | positive candidate s/window | recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `0.000561194499823614` | `0.0005679000000782253` | `0.00037759950009785825` | `0.0003968205001001479` | `0.7` |
| `doc_vectors64` | `0.0006751869998424809` | `0.0006779325001389225` | `0.0003881667498717434` | `0.00039148800010480045` | `0.65` |
| `query_batch4` | `0.0006344890002765169` | `0.000634353250006825` | `0.0005176909999136114` | `0.0005192812500354194` | `0.7` |

Decision: keep positive-centroid postings as an explicit benchmark switch, not
as the internal GPU-pipeline default.

Reason: it preserved recall on these rows but did not improve candidate
generation or the scoped envelope. The useful result is the falsifiable knob;
the default path remains unchanged.
