# 2026-04-27: GPU I8 Candidate-Window Policy Matrix

## Claim

The resident selected-posting GPU path is fast enough that the next correctness
problem is candidate-window coverage, not exact rerank speed.

Reason: previous FastPlaid comparisons showed the resident selected-posting
path well below FastPlaid batch time, but with negative recall deltas on some
non-full windows. The policy needed to be tested across related shapes instead
of tuned on one `doc_vectors64` case.

## Change

Added a repeatable matrix layer:

- `python/kayak_bridge/gpu_i8_candidate_window_policy_matrix.py`
- `python/kayak_bridge/gpu_i8_candidate_window_policy_matrix_runner.py`
- `python/scripts/compare_gpu_i8_fastplaid_candidate_window_policies.py`
- `compare_gpu_i8_fastplaid_candidate_window_policies_raw`
- `compare_gpu_i8_fastplaid_candidate_window_policies`

The matrix runs one or more candidate-window policies on the same explicit
shape family and summarizes:

- per-policy minimum resident selected recall delta versus FastPlaid
- per-policy resident selected time divided by FastPlaid batch time
- effective/input `candidate_k`
- baseline comparisons against the input `candidate_k`
- negative recall-delta row counts in new summaries

Reason: this keeps execution separate from aggregation, preserves explicit
vector counts, and makes failed policies visible instead of buried in one-off
JSON reports.

## Fixed-Seed Control

Added `--fixed-case-seed` to
`python/scripts/compare_gpu_i8_fastplaid_policy.py`.

Reason: candidate-k sweeps must keep generated data constant while only
changing `candidate_k`. The existing policy compare intentionally used
`seed + case_index` to diversify shape matrices, which is correct for broad
policy runs but invalid for isolated `candidate_k` diagnostics.

## New Shape Family

Added `candidate_window_generalization`:

| case | documents | document vectors | queries | query vectors | input k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `doc_vectors48` | `512` | `48` | `2` | `8` | `256` |
| `doc_vectors64` | `512` | `64` | `2` | `8` | `256` |
| `doc_vectors96` | `512` | `96` | `2` | `8` | `256` |
| `documents1024_k256` | `1024` | `16` | `2` | `8` | `256` |

Reason: this family tests below-threshold, threshold, above-threshold, and
larger-document-count candidate coverage while keeping vector dimension and
query shape explicit.

## Policy

Added benchmark-only `coverage_safety_v0`:

| condition | effective k rule | reason |
| --- | ---: | --- |
| full input window | input k | already covers every document |
| `document_count >= 1024` | `document_count` | `k=768` still had a CUDA miss; full window recovered recall |
| `document_vector_count >= 96` | `ceil(1.75 * input k)` | fixed-seed `doc_vectors96` required `k=448` from input `256` |
| `document_vector_count >= 64 or document_count >= 512` | `ceil(1.25 * input k)` | `doc_vectors64` and `doc_vectors48` needed `k=320` headroom |
| otherwise | input k | no measured coverage-risk rule matched |

This remains an internal benchmark policy, not a public search default.

## Measurements

Policy falsification run:

```bash
pixi run compare_gpu_i8_fastplaid_candidate_window_policies
```

Artifact:

- `.cache/kayak/gpu_i8_fastplaid_candidate_window_policies/summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260427T181122Z`

Result: `doc_vectors64_125pct_v0` still had negative recall deltas on
`doc_vectors96` and `documents1024_k256`.

Fixed-seed diagnostics:

```bash
pixi run env UV_CACHE_DIR=.cache/uv uv run --python 3.11 --with fast-plaid==1.4.6.2110 python python/scripts/compare_gpu_i8_fastplaid_policy.py --case doc_vectors96_k320_seed9:documents=512,document_vectors=96,queries=2,query_vectors=8,candidate_k=320 --case doc_vectors96_k384_seed9:documents=512,document_vectors=96,queries=2,query_vectors=8,candidate_k=384 --case doc_vectors96_k448_seed9:documents=512,document_vectors=96,queries=2,query_vectors=8,candidate_k=448 --case doc_vectors96_k512_seed9:documents=512,document_vectors=96,queries=2,query_vectors=8,candidate_k=512 --include-full-window --fixed-case-seed --seed 9 --policy-name shape_rule_v0 --fastplaid-devices cpu --overwrite-index-root --require-fastplaid --allow-missing-gpu --output .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/doc_vectors96_fixed_seed9_k_sweep_cpu_summary.json --report-root .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/doc_vectors96_fixed_seed9_k_sweep_cpu_reports
pixi run env UV_CACHE_DIR=.cache/uv uv run --python 3.11 --with fast-plaid==1.4.6.2110 python python/scripts/compare_gpu_i8_fastplaid_policy.py --case documents1024_k256_seed10:documents=1024,document_vectors=16,queries=2,query_vectors=8,candidate_k=256 --case documents1024_k512_seed10:documents=1024,document_vectors=16,queries=2,query_vectors=8,candidate_k=512 --case documents1024_k768_seed10:documents=1024,document_vectors=16,queries=2,query_vectors=8,candidate_k=768 --case documents1024_k1024_seed10:documents=1024,document_vectors=16,queries=2,query_vectors=8,candidate_k=1024 --include-full-window --fixed-case-seed --seed 10 --policy-name shape_rule_v0 --fastplaid-devices cpu --overwrite-index-root --require-fastplaid --allow-missing-gpu --output .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/documents1024_fixed_seed10_k_sweep_cpu_summary.json --report-root .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/documents1024_fixed_seed10_k_sweep_cpu_reports
```

Key fixed-seed results:

| diagnostic | k | resident recall | FastPlaid CPU recall | delta |
| --- | ---: | ---: | ---: | ---: |
| `doc_vectors96`, seed 9 | `320` | `0.35` | `0.50` | `-0.15` |
| `doc_vectors96`, seed 9 | `384` | `0.55` | `0.60` | `-0.05` |
| `doc_vectors96`, seed 9 | `448` | `0.85` | `0.60` | `+0.25` |
| `documents1024`, seed 10 | `256` | `0.25` | `0.40` | `-0.15` |
| `documents1024`, seed 10 | `512` | `0.40` | `0.45` | `-0.05` |
| `documents1024`, seed 10 | `768` | `0.50` | `0.45` | `+0.05` |
| `documents1024`, seed 10 | `1024` | `1.0` | `0.45` | `+0.55` |

Final validation:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 360 --force -- pixi run env UV_CACHE_DIR=.cache/uv uv run --python 3.11 --with fast-plaid==1.4.6.2110 python python/scripts/compare_gpu_i8_fastplaid_candidate_window_policies.py --candidate-window-policy input --candidate-window-policy coverage_safety_v0 --allow-missing-gpu --require-fastplaid --overwrite-index-root --emit-quiet-mean --output .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/input_vs_coverage_safety_summary.json --report-root .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/input_vs_coverage_safety_reports
```

Artifact:

- `.cache/kayak/gpu_i8_fastplaid_candidate_window_policies/input_vs_coverage_safety_summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260427T183550Z`

Final result:

| policy | rows ok | min recall delta vs FastPlaid | mean resident / FastPlaid | max resident / FastPlaid | max effective/input k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `input` | `8 / 8` | `-0.25` | `0.10208807022610542` | `0.24680414979710544` | `1.0` |
| `coverage_safety_v0` | `8 / 8` | `+0.050000000000000044` | `0.13798288231784603` | `0.396522913897683` | `4.0` |

Agreement minima for `coverage_safety_v0`:

- candidate positions: `1.0`
- final top-k positions: `1.0`

Original `wide_topk` non-full validation:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 300 --force -- pixi run env UV_CACHE_DIR=.cache/uv uv run --python 3.11 --with fast-plaid==1.4.6.2110 python python/scripts/compare_gpu_i8_fastplaid_candidate_window_policies.py --case-set wide_topk --candidate-window-policy input --candidate-window-policy coverage_safety_v0 --allow-missing-gpu --require-fastplaid --overwrite-index-root --emit-quiet-mean --output .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/wide_topk_input_vs_coverage_safety_summary.json --report-root .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/wide_topk_input_vs_coverage_safety_reports
```

Artifact:

- `.cache/kayak/gpu_i8_fastplaid_candidate_window_policies/wide_topk_input_vs_coverage_safety_summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260427T184333Z`

| policy | rows ok | min recall delta vs FastPlaid | mean resident / FastPlaid | max resident / FastPlaid | negative rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `input` | `6 / 6` | `~0.0` | `0.09688456479341993` | `0.20682108635153335` | `0` |
| `coverage_safety_v0` | `6 / 6` | `+0.050000000000000044` | `0.11243990380894976` | `0.25440229610377485` | `0` |

## Reusable Selected-Posting Session

Follow-up claim: selected-posting candidate generation should prepare the GPU
posting session once per benchmark probe, not once per measured query window.

Reason: the resident selected-posting path is modeling a resident GPU primitive.
Per-window session construction measures repeated device setup that a serving
primitive would not repeat for every query batch. The one-shot API is still
kept as a compatibility wrapper, but the FastPlaid comparison probe now uses a
reusable selected-posting session handle.

Validation run:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 300 --force -- pixi run env UV_CACHE_DIR=.cache/uv uv run --python 3.11 --with fast-plaid==1.4.6.2110 python python/scripts/compare_gpu_i8_fastplaid_candidate_window_policies.py --candidate-window-policy coverage_safety_v0 --allow-missing-gpu --require-fastplaid --overwrite-index-root --emit-quiet-mean --output .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/coverage_safety_reusable_selected_session_summary.json --report-root .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/coverage_safety_reusable_selected_session_reports
```

Artifact:

- `.cache/kayak/gpu_i8_fastplaid_candidate_window_policies/coverage_safety_reusable_selected_session_summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260427T185342Z`

Result versus the previous quiet `candidate_window_generalization` validation:

| version | rows ok | min recall delta vs FastPlaid | mean resident / FastPlaid | max resident / FastPlaid | candidate agreement min | final agreement min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| per-window selected-posting prepare | `8 / 8` | `+0.050000000000000044` | `0.13798288231784603` | `0.396522913897683` | `1.0` | `1.0` |
| reusable selected-posting session | `8 / 8` | `+0.050000000000000044` | `0.12560525946839413` | `0.35583497335955816` | `1.0` | `1.0` |

This is a measured improvement of roughly `9%` on the mean resident/FastPlaid
ratio and roughly `10%` on the max row ratio for this matrix, with no observed
loss in recall or order agreement.

Reusable-session row-stage shares:

| case | device | documents | document vectors | effective k | resident/FastPlaid | CPU select share | candidate share | exact share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `doc_vectors48` | `cpu` | `512` | `48` | `320` | `0.041803869127046034` | `0.3569816635887928` | `0.27831837196441955` | `0.36469996444678765` |
| `doc_vectors48` | `cuda` | `512` | `48` | `320` | `0.17155928961607578` | `0.35525280182627933` | `0.27681270542277336` | `0.3679344927509473` |
| `doc_vectors64` | `cpu` | `512` | `64` | `320` | `0.031419028865149405` | `0.2554410936948984` | `0.2644255523843633` | `0.48013335392073825` |
| `doc_vectors64` | `cuda` | `512` | `64` | `320` | `0.15207282874333414` | `0.2555739595402934` | `0.2693590646369914` | `0.47506697582271523` |
| `doc_vectors96` | `cpu` | `512` | `96` | `448` | `0.01609191456201011` | `0.1729729208057524` | `0.2171748262267604` | `0.6098522529674872` |
| `doc_vectors96` | `cuda` | `512` | `96` | `448` | `0.1437829910794396` | `0.17081428378417277` | `0.21536409404587906` | `0.6138216221699482` |
| `documents1024_k256` | `cpu` | `1024` | `16` | `1024` | `0.09227718039453975` | `0.3025955636672748` | `0.5018771218859398` | `0.1955273144467854` |
| `documents1024_k256` | `cuda` | `1024` | `16` | `1024` | `0.35583497335955816` | `0.3058644701299257` | `0.4957593839313375` | `0.1983761459387368` |

Interpretation: there is no single next bottleneck. The vector-heavy rows are
exact-rerank dominated, while the `1024`-document full-window rows are
candidate-generation dominated. The next optimization should therefore be
chosen from the row being targeted, not from the matrix mean alone.

## Decision

Keep `coverage_safety_v0` as the next benchmark policy candidate.

Reason: on the generalized synthetic matrix it repaired the input-window recall
failures while staying below `0.40x` FastPlaid batch time on every CPU/CUDA row.

Do not promote it to a public default yet.

Reason: the policy is deliberately conservative, can use full windows for
`document_count >= 1024`, and has only been validated on synthetic dim128
families. It is a good benchmark policy for the next primitive, not a serving
policy.

## Next

The next optimization target is now the cost of safe candidate coverage:

- profile the resident selected-posting row under `coverage_safety_v0`
- move selected-centroid scoring/selection onto the GPU only if the broader
  policy continues to hold
- test larger or real encoded corpora before claiming a backend-level win
