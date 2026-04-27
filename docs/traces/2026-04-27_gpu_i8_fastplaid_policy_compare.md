# 2026-04-27: GPU I8 Shape-Policy FastPlaid Compare

## Claim

The shape-only centroid-budget policy should be compared against FastPlaid by
an automated matrix, not by hand-running selected rows.

Reason: the prior selected-budget comparison was correct but manual. A
repeatable policy comparison needs to derive `centroids_per_query_vector` from
explicit shape fields, write per-row reports, and summarize the CPU/CUDA
FastPlaid scope ratios in one artifact.

## Change

Added an automated policy-driven FastPlaid comparison:

- `python/kayak_bridge/gpu_i8_fastplaid_policy_compare.py`
- `python/scripts/compare_gpu_i8_fastplaid_policy.py`
- `python/tests/test_gpu_i8_fastplaid_policy_compare.py`
- `compare_gpu_i8_fastplaid_policy_raw`
- `compare_gpu_i8_fastplaid_policy`

The wrapper selects the policy budget from the case shape, then calls the
existing `compare_gpu_i8_fastplaid.py` path for each case and FastPlaid device.
It writes full per-row reports and a compact summary report.

Reason: this preserves the established FastPlaid comparison contract while
removing the manual selected-row step.

## Measurement

Commands:

```bash
pixi run python -m py_compile python/kayak_bridge/gpu_i8_fastplaid_policy_compare.py python/scripts/compare_gpu_i8_fastplaid_policy.py python/tests/test_gpu_i8_fastplaid_policy_compare.py
pixi run env PYTHONPATH=python python -m unittest python/tests/test_gpu_i8_fastplaid_policy_compare.py
pixi run env PYTHONPATH=python python -m unittest python/tests/test_gpu_i8_fastplaid_policy_compare.py python/tests/test_gpu_i8_centroid_budget_policy.py python/tests/test_gpu_i8_rerank_contract.py python/tests/test_fastplaid_speed_track.py
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 90 --force -- pixi run compare_gpu_i8_fastplaid_policy_raw
```

Artifacts:

- quiet log: `.cache/kayak/bench_quiet/20260427T102324Z`
- summary report: `.cache/kayak/gpu_i8_fastplaid_policy_compare/summary.json`
- per-row reports:
  `.cache/kayak/gpu_i8_fastplaid_policy_compare/reports/`

## Results

The automated wide non-full policy comparison reported status `ok` on all
`6 / 6` rows.

| case | FastPlaid device | seed | cpqv | Kayak i8 recall | FastPlaid recall | CPU candidates + GPU no-ref top-k/window s | FastPlaid batch s | envelope / FastPlaid |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `cpu` | `7` | `4` | `0.7` | `0.4` | `0.0005641172500645553` | `0.013735887000166258` | `0.04106886217524411` |
| `query_vectors32` | `cuda` | `7` | `4` | `0.7` | `0.4` | `0.0005653894999113618` | `0.002906660999997257` | `0.19451511542346883` |
| `doc_vectors64` | `cpu` | `8` | `16` | `0.65` | `0.65` | `0.0006445645000212608` | `0.02580610900031388` | `0.02497720597915094` |
| `doc_vectors64` | `cuda` | `8` | `16` | `0.65` | `0.6` | `0.0006411607499785532` | `0.005982657999993535` | `0.107169881677884` |
| `query_batch4` | `cpu` | `9` | `8` | `0.7` | `0.625` | `0.0006956447499533169` | `0.01236659400001372` | `0.056251927568136` |
| `query_batch4` | `cuda` | `9` | `8` | `0.7` | `0.625` | `0.000692231499783702` | `0.00447700899985648` | `0.15461918879454856` |

Summary:

- mean envelope / FastPlaid batch: `0.09643369693640541`
- max envelope / FastPlaid batch: `0.19451511542346883`
- mean CPU candidate-generation share of envelope:
  `0.6942162036037187`
- mean GPU no-reference top-k share of envelope:
  `0.3057837963962813`
- minimum Kayak i8 recall delta versus FastPlaid: `0.0`
- minimum no-reference top-k position agreement: `1.0`

## Interpretation

Verified:

- `shape_rule_v0` now has an automated CPU/CUDA FastPlaid comparison surface
- all rows used explicit document count, document vector count, query count,
  query vector count, `candidate_k`, seed, and selected `cpqv`
- Kayak i8 recall matched or exceeded FastPlaid recall on every automated row
- CPU candidate generation plus GPU no-reference top-k stayed faster than
  FastPlaid full search in the scoped comparison
- CPU candidate generation is now the larger measured share of the remaining
  envelope, so candidate generation/workspace behavior should be optimized
  before the GPU no-reference top-k kernel

Not claimed:

- this is still not an apples-to-apples backend comparison
- candidate generation is still CPU-side
- this does not make `shape_rule_v0` a public default
- this does not prove the same policy on real encoded corpora

## Decision

Use the automated policy comparison as the FastPlaid speed track for the GPU
i8 candidate-window lane.

Reason: it is reproducible, records the same explicit vector-count contract as
the CPU/GPU reports, and prevents future comparisons from depending on
manually selected commands.

## Validation

Observed:

- Python syntax check passed
- focused policy FastPlaid tests: `3 / 3` passed
- focused GPU/FastPlaid suite including policy tests: `61 / 61` passed
- quiet automated policy FastPlaid comparison status: `ok`

## Follow-Up: Unordered Internal Candidate Windows

The policy comparison now runs the internal GPU pipeline with
`kayak_i8_candidate_order=unordered` by default. The ordered public candidate
API remains unchanged.

Latest quiet artifact:

- quiet log: `.cache/kayak/bench_quiet/20260427T105039Z`
- summary report: `.cache/kayak/gpu_i8_fastplaid_policy_compare/summary.json`

The unordered run reported status `ok` on all `6 / 6` rows, top-k position
agreement minimum `1.0`, and minimum Kayak recall delta versus FastPlaid
`1.1102230246251565e-16`.

Summary:

- mean envelope / FastPlaid batch: `0.09869997626737713`
- max envelope / FastPlaid batch: `0.21780632157235916`
- mean CPU candidate-generation share of envelope:
  `0.6723469052696253`
- mean GPU no-reference top-k share of envelope:
  `0.3276530947303748`

Reason: candidate-window rerank only needs the retained document set. The
unordered path is kept behind an explicit internal flag so we can use it in GPU
pipeline profiling without changing public candidate ordering semantics.

## Follow-Up: Resident Selected-Posting Exact Rerank

The policy comparison now includes a resident selected-posting exact-rerank
row. This row starts from CPU-selected centroids, uses the resident GPU
selected-posting candidate-window primitive, then exact-reranks that candidate
window with the GPU i8 address scorer.

Latest quiet artifact:

- quiet log: `.cache/kayak/bench_quiet/20260427T172856Z`
- summary report: `.cache/kayak/gpu_i8_fastplaid_policy_compare/summary.json`

The run reported status `ok` on all `6 / 6` rows:

- resident selected candidate position agreement minimum: `1.0`
- resident selected exact-rerank final top-k agreement minimum: `1.0`
- mean resident selected exact-rerank / FastPlaid batch:
  `0.10213825609061737`
- max resident selected exact-rerank / FastPlaid batch:
  `0.218733060397411`
- max cold resident selected exact-rerank / FastPlaid batch:
  `0.23550834938348914`
- minimum resident selected recall delta versus FastPlaid:
  `-0.04999999999999993`
- mean CPU selected-centroid share:
  `0.3432215448787619`
- mean resident candidate scoring/selection share:
  `0.3320694691766924`
- mean exact-rerank share:
  `0.3247089859445457`

Interpretation:

This is the first policy row in this matrix that is both much faster than
FastPlaid full search and built from a GPU candidate-generation primitive plus
exact rerank. The headline timing includes CPU selected-centroid work, resident
GPU selected-posting scoring/selection, and GPU exact rerank. It still is not a
public backend claim because the selected centroid step is CPU-side and the
matrix is synthetic.

The remaining blocker is quality, not speed. The `doc_vectors64` CPU row
measured Kayak resident selected recall `0.65` versus FastPlaid recall `0.70`.
Until that candidate coverage gap is closed or explained by variance, the
correct next target is the shape-policy candidate budget/coverage, not another
low-level kernel speed pass.

## Follow-Up: Candidate-Window Width Policy

The comparison now has a benchmark-only candidate-window policy,
`doc_vectors64_125pct_v0`. It leaves each explicit `candidate_k` unchanged
except for non-full `document_vector_count >= 64` windows, where it widens the
window to `ceil(1.25 * input_candidate_k)`.

Latest quiet artifacts:

- repeated quiet log: `.cache/kayak/bench_quiet/20260427T175127Z`
- summary report:
  `.cache/kayak/gpu_i8_fastplaid_policy_compare/doc64_125pct_repeat_summary.json`

The run reported status `ok` on all `6 / 6` rows:

- minimum resident selected recall delta versus FastPlaid:
  `0.050000000000000044`
- mean resident selected exact-rerank / FastPlaid batch:
  `0.11238619389244382`
- max resident selected exact-rerank / FastPlaid batch:
  `0.24276467216832875`
- max cold resident selected exact-rerank / FastPlaid batch:
  `0.26184393079042456`
- candidate and final top-k agreement minima: `1.0`

Reason: the fixed-seed diagnostic showed the `doc_vectors64` miss was a
candidate-coverage issue. `candidate_k=256` returned recall `0.65`;
`candidate_k=260` returned `0.70`; `candidate_k=320` returned `0.75`; and the
full `512` window returned `1.0`. The repeated policy run kept positive recall
delta on every row, but the minimum observed margin was `0.05`, so this remains
a benchmark-track policy rather than a public default.
