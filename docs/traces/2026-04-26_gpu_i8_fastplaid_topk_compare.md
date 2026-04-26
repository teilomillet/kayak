# 2026-04-26: GPU I8 FastPlaid Top-K Boundary Compare

## Claim

The FastPlaid comparison should include the prepared-handle top-k boundary, not
only the synthetic candidate-score kernel probe.

Reason: the top-k handle row is the closest current Kayak GPU rerank boundary:
it uses real Kayak i8 payload snapshots, CPU-provided candidate windows, an
explicit resident GPU handle, and returns only top-k positions/scores. FastPlaid
is still a full-search system baseline, so the report must keep scope warnings.

## Added

- `gpu_prepared_handle_topk_primitive` in
  `python/scripts/compare_gpu_i8_fastplaid.py`
- `gpu_prepared_handle_topk_vs_fastplaid_scope_comparison`
- quiet-wrapper sections:
  - `kayak_gpu_i8_prepared_handle_topk_per_window`
  - `kayak_cpu_candidates_gpu_i8_topk_per_window`
- contract coverage in `python/tests/test_gpu_i8_rerank_contract.py`

The row reports:

- query count, query vector count, document count, document vector count, and
  total document vector count
- candidate score count per window
- top-k return count per window
- top-k position agreement versus CPU i8
- score delta max abs versus CPU i8
- CPU candidate generation plus GPU top-k per-window timing
- ratios versus FastPlaid full-search batch timing, with scope warnings

## Measurement

Commands:

```bash
pixi run compare_gpu_i8_fastplaid
pixi run compare_gpu_i8_fastplaid_cuda
```

Artifacts:

- CPU FastPlaid report:
  `.cache/kayak/gpu_i8_fastplaid_compare/summary.json`
- CUDA FastPlaid report:
  `.cache/kayak/gpu_i8_fastplaid_compare/cuda_summary.json`
- CPU quiet log:
  `.cache/kayak/bench_quiet/20260426T191305Z`
- CUDA quiet log:
  `.cache/kayak/bench_quiet/20260426T191822Z`

Common shape:

- documents: `256`
- document vectors per document: `16`
- total document vectors: `4096`
- queries per window: `2`
- query vectors per query: `8`
- total query vectors per window: `16`
- candidate window: `128`
- candidate scores per window: `256`
- top-k: `10`
- top-k returns per window: `20`
- GPU target: `nvidia:sm_89`
- FastPlaid version: `1.4.6.2110`

## Results

| FastPlaid device | FastPlaid batch s | GPU top-k/window s | CPU candidates + GPU top-k/window s | top-k/FastPlaid batch | candidates+top-k/FastPlaid batch | top-k/CPU score | candidates+top-k/CPU candidates+score | top-k agreement | max delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | `0.003948755998862907` | `0.00010890049907175126` | `0.0006470917487604311` | `0.027578432069013756` | `0.16387230534040825` | `0.3681913195685502` | `0.7759240428771922` | `1.0` | `0.0000457763671875` |
| CUDA | `0.0016726499998185318` | `0.00011942050059587928` | `0.0006539250007335795` | `0.0713959887656326` | `0.39095148465281127` | `0.4107435380712352` | `0.7923993645632785` | `1.0` | `0.0000457763671875` |

Recall context from the same quiet reports:

| Row | CPU FastPlaid report recall@10 | CUDA FastPlaid report recall@10 |
| --- | ---: | ---: |
| Kayak i8 CPU candidate-window row | `0.5` | `0.5` |
| FastPlaid | `0.45` | `0.6` |

## Interpretation

Verified:

- the FastPlaid comparison report now contains the real-payload prepared-handle
  top-k boundary
- top-k order agreement against CPU i8 was `1.0` in both CPU and CUDA
  FastPlaid comparison reports
- CPU candidate generation plus GPU top-k was faster than CPU candidate
  generation plus CPU same-candidate scoring in both reports
- the report keeps explicit scope warnings because FastPlaid is full search and
  the Kayak GPU row is an internal rerank boundary

Not claimed:

- this is not a production GPU backend speedup claim
- this does not move candidate generation to GPU
- this does not prove broader superiority against FastPlaid on larger or real
  encoded workloads

## Decision

Keep the FastPlaid comparison row and use it as the external-baseline context
for the next GPU optimization step.

Reason: it makes the current best GPU boundary visible beside FastPlaid while
preserving the distinction between full-search and rerank-only scopes.

## Validation

Ran:

```bash
pixi run python -m py_compile \
  python/scripts/compare_gpu_i8_fastplaid.py \
  python/kayak_bridge/gpu_i8_fastplaid_topk_compare.py \
  python/kayak_bridge/gpu_i8_fastplaid_topk_metrics.py \
  python/tests/test_gpu_i8_rerank_contract.py
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_gpu_i8_rerank_contract -v
pixi run compare_gpu_i8_fastplaid_raw
pixi run compare_gpu_i8_fastplaid_cuda_raw
pixi run compare_gpu_i8_fastplaid
pixi run compare_gpu_i8_fastplaid_cuda
```

Observed:

- compile checks passed
- GPU i8 contract tests: `32/32` passed
- raw CPU FastPlaid comparison status: `ok`
- raw CUDA FastPlaid comparison status: `ok`
- quiet CPU FastPlaid comparison status: `ok`, `7` sections
- quiet CUDA FastPlaid comparison status: `ok`, `7` sections

One implementation bug was found and fixed during validation: the first raw
CPU comparison run used a stale local name for the exact reference positions and
failed before writing a valid report.
