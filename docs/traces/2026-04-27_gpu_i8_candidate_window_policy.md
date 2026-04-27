# 2026-04-27: GPU I8 Candidate-Window Policy

## Claim

The remaining FastPlaid quality gap on the resident selected-posting path is a
candidate-window width issue, not an i8 quantization ceiling or another kernel
speed issue.

Reason: the previous resident selected-posting row was already much faster than
FastPlaid, but `doc_vectors64` at `candidate_k=256` could trail FastPlaid recall
by `0.05`. Candidate count is therefore the next design axis to test.

## Change

Added a benchmark-only candidate-window policy:

- `python/kayak_bridge/gpu_i8_candidate_window_policy.py`
- `--candidate-window-policy doc_vectors64_125pct_v0`

The policy keeps each case's explicit `candidate_k` except for non-full
`document_vector_count >= 64` windows, where it widens the window to
`ceil(1.25 * input_candidate_k)` capped by `document_count`.

Reason: the focused seed-8 diagnostic showed `candidate_k=260` was enough to
repair `doc_vectors64`, but that is a knife-edge `+4` rule. The `1.25x` rule
maps the measured `256` window to `320`, provides recall headroom, keeps vector
counts explicit, and remains a benchmark policy rather than a public default.

## Measurement

Focused diagnostics:

```bash
pixi run compare_gpu_i8_fastplaid_policy --case doc_vectors64_k256_seed8:documents=512,document_vectors=64,queries=2,query_vectors=8,candidate_k=256 --policy-name static16 --seed 8 --fastplaid-devices cpu --overwrite-index-root --report-root .cache/kayak/gpu_i8_fastplaid_policy_compare/candidate_k_fixed_seed_reports --output .cache/kayak/gpu_i8_fastplaid_policy_compare/doc_vectors64_k256_static16_seed8_summary.json
pixi run compare_gpu_i8_fastplaid_policy --case doc_vectors64_k260_seed8:documents=512,document_vectors=64,queries=2,query_vectors=8,candidate_k=260 --policy-name static16 --seed 8 --fastplaid-devices cpu --overwrite-index-root --report-root .cache/kayak/gpu_i8_fastplaid_policy_compare/candidate_k_fixed_seed_reports --output .cache/kayak/gpu_i8_fastplaid_policy_compare/doc_vectors64_k260_static16_seed8_summary.json
pixi run compare_gpu_i8_fastplaid_policy --case doc_vectors64_full_seed8:documents=512,document_vectors=64,queries=2,query_vectors=8,candidate_k=512 --include-full-window --policy-name static16 --seed 8 --fastplaid-devices cpu,cuda --overwrite-index-root --report-root .cache/kayak/gpu_i8_fastplaid_policy_compare/full_window_seed8_reports --output .cache/kayak/gpu_i8_fastplaid_policy_compare/doc_vectors64_full_static16_seed8_summary.json
```

Policy run:

```bash
pixi run compare_gpu_i8_fastplaid_policy --candidate-window-policy doc_vectors64_125pct_v0 --overwrite-index-root --report-root .cache/kayak/gpu_i8_fastplaid_policy_compare/doc64_125pct_reports --output .cache/kayak/gpu_i8_fastplaid_policy_compare/doc64_125pct_summary.json
bash scripts/run_bench_quiet.sh --repeats 2 --timeout-seconds 60 --force -- pixi run compare_gpu_i8_fastplaid_policy_raw --candidate-window-policy doc_vectors64_125pct_v0 --overwrite-index-root --report-root .cache/kayak/gpu_i8_fastplaid_policy_compare/doc64_125pct_repeat_reports --output .cache/kayak/gpu_i8_fastplaid_policy_compare/doc64_125pct_repeat_summary.json
```

Artifacts:

- single-run quiet log: `.cache/kayak/bench_quiet/20260427T174709Z`
- repeated quiet log: `.cache/kayak/bench_quiet/20260427T175127Z`
- summary report:
  `.cache/kayak/gpu_i8_fastplaid_policy_compare/doc64_125pct_repeat_summary.json`
- per-row reports:
  `.cache/kayak/gpu_i8_fastplaid_policy_compare/doc64_125pct_repeat_reports/`

## Results

Focused seed-8 diagnostics:

| candidate_k | resident selected recall | FastPlaid CPU recall | resident / FastPlaid CPU batch |
| ---: | ---: | ---: | ---: |
| `256` | `0.65` | `0.70` | `0.023612626717001154` |
| `260` | `0.70` | `0.55` to `0.70` across repeated runs | `0.027966391259386617` |
| `320` | `0.75` | `0.65` to `0.70` across repeated runs | `0.02830738081252415` to `0.03238226352096101` |
| `512` | `1.0` | `0.70` CPU, `0.60` CUDA | `0.038824174921719405` CPU, `0.1575053693075278` CUDA |

The full-window row reaching `1.0` recall on seed 8 falsifies the quantization
ceiling hypothesis for this case. The miss at `candidate_k=256` is candidate
coverage.

`doc_vectors64_125pct_v0` policy matrix:

| case | FastPlaid device | input k | effective k | resident recall | FastPlaid recall | recall delta | resident / FastPlaid batch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `cpu` | `256` | `256` | `0.7` | `0.45` | `0.25` | `0.04245836170947288` |
| `query_vectors32` | `cuda` | `256` | `256` | `0.7` | `0.4` | `0.3` | `0.24276467216832875` |
| `doc_vectors64` | `cpu` | `256` | `320` | `0.75` | `0.55` | `0.2` | `0.03184824175010281` |
| `doc_vectors64` | `cuda` | `256` | `320` | `0.75` | `0.7` | `0.05` | `0.14904418442819303` |
| `query_batch4` | `cpu` | `256` | `256` | `0.7` | `0.575` | `0.125` | `0.0500939445281642` |
| `query_batch4` | `cuda` | `256` | `256` | `0.7` | `0.65` | `0.05` | `0.1581077587704013` |

Summary:

- status: `ok`
- rows: `6 / 6`
- minimum resident selected recall delta versus FastPlaid:
  `0.050000000000000044`
- mean resident selected exact-rerank / FastPlaid batch:
  `0.11238619389244382`
- max resident selected exact-rerank / FastPlaid batch:
  `0.24276467216832875`
- max cold resident selected exact-rerank / FastPlaid batch:
  `0.26184393079042456`
- candidate and final top-k agreement minima: `1.0`

## Decision

Keep `doc_vectors64_125pct_v0` as a benchmark-only candidate-window policy and
use it for the next FastPlaid speed/quality track.

Reason: it is the first repeated policy matrix where the resident selected GPU
path beats FastPlaid recall on every CPU/CUDA row while staying well under
FastPlaid full-search time. It is still not a public backend claim because the
matrix is synthetic, the minimum observed margin is only `0.05`, and the
selected-centroid step remains CPU-side.

## Next

The next optimization target is not a wider window by default. The next target
is variance and generality:

- test real encoded corpora or larger synthetic shape families
- repeat more than two runs before treating the `0.05` recall margin as stable
- then move CPU selected-centroid work to a resident GPU selector if the policy
  continues to hold

## Follow-Up

The broader `candidate_window_generalization` matrix falsified
`doc_vectors64_125pct_v0` as a general benchmark policy. It still missed
`doc_vectors96` and `documents1024_k256` rows.

Follow-up trace:

- `docs/traces/2026-04-27_gpu_i8_candidate_window_policy_matrix.md`

Decision update: keep `doc_vectors64_125pct_v0` as a useful narrow diagnostic,
but use `coverage_safety_v0` for the next benchmark-policy validation step.
