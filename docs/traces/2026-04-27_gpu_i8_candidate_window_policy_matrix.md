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

## Full-Window Ranking Fast Path

Follow-up claim: when `candidate_k == document_count`, the dense selected-posting
candidate scorer still needs to preserve score order, but it should not use the
partial-top-k path.

Reason: the old helper used `argpartition` plus threshold handling even though a
full window needs a complete score ordering. A stable descending `argsort`
preserves the existing tie-break by document id and avoids the extra partition
work.

Local focused check:

```bash
pixi run env PYTHONPATH=python python -c 'import numpy as np, timeit; from kayak_bridge.mojo_gpu_i8_rerank import _rank_document_scores_numpy; rng=np.random.default_rng(7); scores=rng.normal(size=2048).astype(np.float32); print(timeit.timeit(lambda: _rank_document_scores_numpy(scores, query_count=2, document_count=1024, top_k=1024), number=1000))'
```

Result:

| version | 1000 calls on 2x1024 full-window scores |
| --- | ---: |
| before | `0.320596072000626` |
| after | `0.05686295300256461` |

Targeted validation:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 300 --force -- pixi run env UV_CACHE_DIR=.cache/uv uv run --python 3.11 --with fast-plaid==1.4.6.2110 python python/scripts/compare_gpu_i8_fastplaid_policy.py --case documents1024_k256:documents=1024,document_vectors=16,queries=2,query_vectors=8,candidate_k=256 --candidate-window-policy coverage_safety_v0 --fastplaid-devices cuda --seed 10 --fixed-case-seed --allow-missing-gpu --require-fastplaid --overwrite-index-root --emit-quiet-mean --output .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/documents1024_full_window_rank_fastpath_summary.json --report-root .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/documents1024_full_window_rank_fastpath_reports
```

Artifact:

- `.cache/kayak/gpu_i8_fastplaid_candidate_window_policies/documents1024_full_window_rank_fastpath_summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260427T190542Z`

Targeted `documents1024_k256` CUDA result:

| metric | before reusable-session matrix | after targeted run |
| --- | ---: | ---: |
| candidate selection seconds | `0.0003908500002580695` | `0.00010980500519508496` |
| resident / FastPlaid | `0.35583497335955816` | `0.25730388724538644` |
| recall delta vs FastPlaid | `+0.5` | `+0.75` |
| candidate agreement min | `1.0` | `1.0` |
| final agreement min | `1.0` | `1.0` |

Full matrix validation:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 360 --force -- pixi run env UV_CACHE_DIR=.cache/uv uv run --python 3.11 --with fast-plaid==1.4.6.2110 python python/scripts/compare_gpu_i8_fastplaid_candidate_window_policies.py --candidate-window-policy coverage_safety_v0 --allow-missing-gpu --require-fastplaid --overwrite-index-root --emit-quiet-mean --output .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/coverage_safety_full_window_rank_fastpath_summary.json --report-root .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/coverage_safety_full_window_rank_fastpath_reports
```

Artifact:

- `.cache/kayak/gpu_i8_fastplaid_candidate_window_policies/coverage_safety_full_window_rank_fastpath_summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260427T190618Z`

Aggregate result:

| version | rows ok | min recall delta vs FastPlaid | mean resident / FastPlaid | max resident / FastPlaid | mean candidate share |
| --- | ---: | ---: | ---: | ---: | ---: |
| reusable selected-posting session | `8 / 8` | `+0.050000000000000044` | `0.12560525946839413` | `0.35583497335955816` | `0.314886390062308` |
| full-window ranking fast path | `8 / 8` | `+0.050000000000000044` | `0.12108878426361634` | `0.28820741786941106` | `0.2565679072911855` |

Decision: keep the fast path.

Reason: it preserves candidate and final order agreement, improves the worst
matrix row, and is limited to the shape where the old partial-top-k algorithm
was provably doing unnecessary work.

Updated interpretation: after this change, the `documents1024` full-window rows
are CPU selected-centroid dominated, while the `doc_vectors96` rows remain
exact-rerank dominated. The next optimization should target CPU selected
centroids only if the goal is reducing the previous worst row; otherwise the
vector-heavy rows require exact-rerank work.

## Identity Full-Window Candidate Stage

Follow-up claim: when a coverage policy expands `candidate_k` to
`document_count`, the candidate stage should become an identity stage.

Reason: a full candidate window contains every document, so selected-centroid
generation and selected-posting approximate scoring cannot change exact-rerank
coverage. Keeping that work only measures a redundant stage. The exact rerank
still runs on the same full document set and remains checked against CPU i8
order.

Targeted validation:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 300 --force -- pixi run env UV_CACHE_DIR=.cache/uv uv run --python 3.11 --with fast-plaid==1.4.6.2110 python python/scripts/compare_gpu_i8_fastplaid_policy.py --case documents1024_k256:documents=1024,document_vectors=16,queries=2,query_vectors=8,candidate_k=256 --candidate-window-policy coverage_safety_v0 --fastplaid-devices cuda --seed 10 --fixed-case-seed --allow-missing-gpu --require-fastplaid --overwrite-index-root --emit-quiet-mean --output .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/documents1024_identity_full_window_summary.json --report-root .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/documents1024_identity_full_window_reports
```

Artifact:

- `.cache/kayak/gpu_i8_fastplaid_candidate_window_policies/documents1024_identity_full_window_summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260427T191612Z`

Targeted `documents1024_k256` CUDA result:

| metric | full-window ranking fast path | identity full-window stage |
| --- | ---: | ---: |
| candidate generation kind | selected posting | identity full window |
| resident / FastPlaid | `0.25730388724538644` | `0.07397499279760904` |
| CPU selected-centroid share | `0.44839039693805993` | `0.0` |
| candidate share | `0.26290440330278986` | `0.0` |
| exact share | `0.28870519975915016` | `1.0` |
| final agreement min | `1.0` | `1.0` |

Full matrix validation:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 360 --force -- pixi run env UV_CACHE_DIR=.cache/uv uv run --python 3.11 --with fast-plaid==1.4.6.2110 python python/scripts/compare_gpu_i8_fastplaid_candidate_window_policies.py --candidate-window-policy coverage_safety_v0 --allow-missing-gpu --require-fastplaid --overwrite-index-root --emit-quiet-mean --output .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/coverage_safety_identity_full_window_summary.json --report-root .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/coverage_safety_identity_full_window_reports
```

Artifact:

- `.cache/kayak/gpu_i8_fastplaid_candidate_window_policies/coverage_safety_identity_full_window_summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260427T191639Z`

Aggregate result:

| version | rows ok | min recall delta vs FastPlaid | mean resident / FastPlaid | max resident / FastPlaid | mean CPU select share | mean candidate share | mean exact share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full-window ranking fast path | `8 / 8` | `+0.050000000000000044` | `0.12108878426361634` | `0.28820741786941106` | `0.30834912065328457` | `0.2565679072911855` | `0.4350829720555299` |
| identity full-window stage | `8 / 8` | `+0.09999999999999998` | `0.08421350632826388` | `0.1983504776306026` | `0.19631806264881366` | `0.18989524363318638` | `0.613786693718` |

Decision: keep the identity full-window stage.

Reason: it removes provably redundant work, keeps candidate and final agreement
at `1.0`, improves the full matrix mean and max ratios, and makes the report
explicitly label full-window rows as `identity_full_window`.

Updated interpretation: the previous worst row is no longer the bottleneck.
The new max resident/FastPlaid row is `doc_vectors48` on FastPlaid CUDA at
`0.1983504776306026`. That row is not dominated by one stage: CPU selection,
candidate generation, and exact rerank are all material, so the next change
should start with a row-local breakdown rather than a single-stage assumption.

## Guarded Partial Ranking Fast Path

Follow-up claim: non-full candidate ranking should use the cheaper partitioned
path when the top-k boundary is unambiguous, and should only fall back to
explicit score/doc ordering when extra documents tie at the threshold.

Reason: `np.argpartition` cheaply identifies the top-k score set, but the
candidate contract still requires deterministic score-descending order with
document-id tie breaks. The guarded path keeps the fast case fast and handles
the threshold-tie case explicitly instead of relying on partition order.

Local focused check:

```bash
pixi run env PYTHONPATH=python python -c 'import numpy as np, timeit; from kayak_bridge.mojo_gpu_i8_rerank import _rank_document_scores_numpy; rng=np.random.default_rng(7); scores=rng.normal(size=1024).astype(np.float32); print(timeit.timeit(lambda: _rank_document_scores_numpy(scores, query_count=2, document_count=512, top_k=320), number=1000))'
```

Result:

| version | 1000 calls on 2x512 scores, top-k 320 |
| --- | ---: |
| before | `0.11358778800058644` |
| after | `0.0364334499972756` |

Targeted validation:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 300 --force -- pixi run env UV_CACHE_DIR=.cache/uv uv run --python 3.11 --with fast-plaid==1.4.6.2110 python python/scripts/compare_gpu_i8_fastplaid_policy.py --case doc_vectors48:documents=512,document_vectors=48,queries=2,query_vectors=8,candidate_k=256 --candidate-window-policy coverage_safety_v0 --fastplaid-devices cuda --seed 7 --fixed-case-seed --allow-missing-gpu --require-fastplaid --overwrite-index-root --emit-quiet-mean --output .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/doc_vectors48_partial_rank_fastpath_summary.json --report-root .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/doc_vectors48_partial_rank_fastpath_reports
```

Artifact:

- `.cache/kayak/gpu_i8_fastplaid_candidate_window_policies/doc_vectors48_partial_rank_fastpath_summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260427T192241Z`

Targeted `doc_vectors48` CUDA result:

| metric | identity full-window matrix | after targeted run |
| --- | ---: | ---: |
| candidate selection seconds | `0.00016324400348821655` | `0.00008264500502264127` |
| resident / FastPlaid | `0.1983504776306026` | `0.17605215330541546` |
| recall delta vs FastPlaid | `+0.15` | `+0.15` |
| candidate agreement min | `1.0` | `1.0` |
| final agreement min | `1.0` | `1.0` |

Full matrix validation:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 360 --force -- pixi run env UV_CACHE_DIR=.cache/uv uv run --python 3.11 --with fast-plaid==1.4.6.2110 python python/scripts/compare_gpu_i8_fastplaid_candidate_window_policies.py --candidate-window-policy coverage_safety_v0 --allow-missing-gpu --require-fastplaid --overwrite-index-root --emit-quiet-mean --output .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/coverage_safety_partial_rank_fastpath_summary.json --report-root .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/coverage_safety_partial_rank_fastpath_reports
```

Artifact:

- `.cache/kayak/gpu_i8_fastplaid_candidate_window_policies/coverage_safety_partial_rank_fastpath_summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260427T192317Z`

Aggregate result:

| version | rows ok | min recall delta vs FastPlaid | mean resident / FastPlaid | max resident / FastPlaid | mean CPU select share | mean candidate share | mean exact share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity full-window stage | `8 / 8` | `+0.09999999999999998` | `0.08421350632826388` | `0.1983504776306026` | `0.19631806264881366` | `0.18989524363318638` | `0.613786693718` |
| guarded partial ranking fast path | `8 / 8` | `+0.050000000000000044` | `0.08161427814748465` | `0.1855417397226204` | `0.21925823643775907` | `0.1321313879906215` | `0.6486103755716195` |

Decision: keep the guarded partial ranking fast path.

Reason: it preserves candidate and final order agreement at `1.0`, improves
both the targeted row and the full matrix mean/max ratios, and its correctness
condition is covered by an explicit boundary-tie unit test.

Updated interpretation: candidate ranking is no longer the largest row-local
component. In the current worst CUDA row, CPU centroid selection is roughly
`39%`, candidate generation is roughly `20%`, and exact rerank is roughly
`41%` of resident selected time. The next optimization should inspect exact
rerank and CPU centroid selection before assuming which one has more headroom.

## Selected-Centroid Array Export

Follow-up claim: CPU selected-centroid export should fill typed arrays directly
instead of returning nested Python lists.

Reason: the resident selected-posting path immediately converts selected
centroid positions and scores back into `Int64` and `Float32` arrays before
sending them to the GPU. Returning lists makes Python object materialization
part of the measured CPU selected-centroid boundary, even though the next stage
needs contiguous arrays.

Local focused check:

```bash
pixi run env PYTHONPATH=python:python/scripts python -c 'import timeit; from bench_fastplaid_speed_track import SpeedTrackShape, build_synthetic_inputs; from kayak_bridge.plaid_approx import KayakPlaidApproxConfig, KayakPlaidApproxIndex; shape=SpeedTrackShape(document_count=512, document_vector_count=48, query_count=2, query_vector_count=8, vector_dim=128, top_k=10, update_document_count=0); inputs=build_synthetic_inputs(shape, seed=7, normalize_vectors=False); index=KayakPlaidApproxIndex.build(doc_ids=inputs.doc_ids, documents=inputs.documents, config=KayakPlaidApproxConfig(centroid_count=128, centroids_per_query_vector=24, candidate_k=320, payload="i8"), final_k=10); print(timeit.timeit(lambda: index.i8_selected_centroids_batch(inputs.queries, centroids_per_query_vector=24), number=1000)); selected=index.i8_selected_centroids_batch(inputs.queries, centroids_per_query_vector=24); print(timeit.timeit(lambda: (selected.positions_array(), selected.scores_array()), number=10000)); legacy=index.i8_selected_centroids_batch_legacy_lists(inputs.queries, centroids_per_query_vector=24); print((selected.positions_array()==legacy.positions_array()).all(), abs(selected.scores_array()-legacy.scores_array()).max())'
```

Result:

| measurement | before | after |
| --- | ---: | ---: |
| selected-centroid export, 1000 calls | `0.23306827500346117` | `0.06276784299552673` |
| `positions_array()`/`scores_array()`, 10000 calls | `0.1013079350013868` | `0.0013429950049612671` |
| positions equal to legacy list export | n/a | `True` |
| max score delta versus legacy list export | n/a | `0.0` |

Targeted validation:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 300 --force -- pixi run env UV_CACHE_DIR=.cache/uv uv run --python 3.11 --with fast-plaid==1.4.6.2110 python python/scripts/compare_gpu_i8_fastplaid_policy.py --case doc_vectors48:documents=512,document_vectors=48,queries=2,query_vectors=8,candidate_k=256 --candidate-window-policy coverage_safety_v0 --fastplaid-devices cuda --seed 7 --fixed-case-seed --allow-missing-gpu --require-fastplaid --overwrite-index-root --emit-quiet-mean --output .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/doc_vectors48_selected_centroid_array_export_summary.json --report-root .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/doc_vectors48_selected_centroid_array_export_reports
```

Artifact:

- `.cache/kayak/gpu_i8_fastplaid_candidate_window_policies/doc_vectors48_selected_centroid_array_export_summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260427T193717Z`

Targeted `doc_vectors48` CUDA result:

| metric | guarded partial ranking | selected-centroid array export |
| --- | ---: | ---: |
| resident / FastPlaid | `0.17605215330541546` | `0.1302957418231845` |
| CPU selected-centroid share | `0.39058932579864264` | `0.18063840676669618` |
| candidate share | `0.1998876751902321` | `0.2723268252614697` |
| exact share | `0.4095229990111252` | `0.5470347679718341` |
| candidate agreement min | `1.0` | `1.0` |
| final agreement min | `1.0` | `1.0` |

Full matrix validation:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 360 --force -- pixi run env UV_CACHE_DIR=.cache/uv uv run --python 3.11 --with fast-plaid==1.4.6.2110 python python/scripts/compare_gpu_i8_fastplaid_candidate_window_policies.py --candidate-window-policy coverage_safety_v0 --allow-missing-gpu --require-fastplaid --overwrite-index-root --emit-quiet-mean --output .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/coverage_safety_selected_centroid_array_export_summary.json --report-root .cache/kayak/gpu_i8_fastplaid_candidate_window_policies/coverage_safety_selected_centroid_array_export_reports
```

Artifact:

- `.cache/kayak/gpu_i8_fastplaid_candidate_window_policies/coverage_safety_selected_centroid_array_export_summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260427T193808Z`

Aggregate result:

| version | rows ok | min recall delta vs FastPlaid | mean resident / FastPlaid | max resident / FastPlaid | mean CPU select share | mean candidate share | mean exact share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| guarded partial ranking fast path | `8 / 8` | `+0.050000000000000044` | `0.08161427814748465` | `0.1855417397226204` | `0.21925823643775907` | `0.1321313879906215` | `0.6486103755716195` |
| selected-centroid array export | `8 / 8` | `+0.050000000000000044` | `0.06363883871257384` | `0.136749921661277` | `0.09932719207595908` | `0.16042751747461778` | `0.7402452904494231` |

Environment note: a non-escalated full-matrix run failed with `No CUDA GPUs
are available`. The comparable matrix above was rerun with GPU access enabled.

Decision: keep selected-centroid array export.

Reason: it removes Python list materialization from a measured bridge boundary,
keeps exact equality with the legacy list export in the focused test, preserves
candidate/final agreement at `1.0`, and improves both the matrix mean and max
ratios.

Updated interpretation: exact rerank is now the dominant resident-selected
component across the non-full rows. The next optimization should profile the
exact address scorer before changing candidate generation again.

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

The next optimization target is now exact rerank:

- inspect the resident exact-rerank call path for avoidable host overhead
- profile the address scorer kernels versus readback and host top-k
- test larger or real encoded corpora before claiming a backend-level win
