# 2026-04-25: GPU I8 FastPlaid Comparison

## Claim

FastPlaid should be in the GPU optimization loop, but the comparison must keep
scope explicit.

Reason:

- FastPlaid is a full-search system baseline.
- The current Kayak GPU path is only a benchmark-only candidate-score
  primitive.
- A single report is useful, but the GPU-to-FastPlaid ratio is profiling
  context until the GPU primitive is connected to real Kayak i8 payloads.

## Added

- `python/scripts/compare_gpu_i8_fastplaid.py`
- `pyproject.toml` tasks:
  - `compare_gpu_i8_fastplaid_raw`
  - `compare_gpu_i8_fastplaid`
  - `compare_gpu_i8_fastplaid_cuda_raw`
  - `compare_gpu_i8_fastplaid_cuda`
- `python/tests/test_gpu_i8_rerank_contract.py` coverage for the explicit
  GPU-versus-FastPlaid scope ratios

Reason:

- reuse the existing FastPlaid speed-track harness instead of creating a second
  FastPlaid implementation path
- emit one report containing Kayak exact CPU, Kayak i8 CPU, FastPlaid, and the
  Kayak GPU candidate-score primitive
- keep FastPlaid CPU and FastPlaid CUDA selectable and measured separately

## Shape

Both comparison runs used:

- documents: `256`
- document vectors per document: `16`
- total document vectors: `4096`
- queries: `2`
- query vectors per query: `8`
- total query vectors: `16`
- vector dim: `128`
- top-k: `10`
- Kayak/GPU candidate window: `128`
- GPU candidate scores: `256`
- FastPlaid version: `1.4.6.2110`
- FastPlaid nbits: `4`
- FastPlaid K-means iterations: `4`

Reason: this matches the current GPU profile smoke shape closely enough to
compare against FastPlaid without expanding the benchmark surface before the
primitive is integrated.

## FastPlaid CPU Run

Latest verification command:

```bash
pixi run compare_gpu_i8_fastplaid_raw
```

Artifacts:

- report: `.cache/kayak/gpu_i8_fastplaid_compare/summary.json`
- prior quiet log: `.cache/kayak/bench_quiet/20260425T192130Z`

Results:

| Row | Query batch mean s | QPS | Recall@10 vs exact | Index bytes |
| --- | ---: | ---: | ---: | ---: |
| Kayak exact CPU | `0.08281107799848542` | `24.15135810738485` | `1.0` | `2099208` |
| Kayak i8 CPU | `0.0006068820002838038` | `3295.5335618204444` | `0.5` | `575312` |
| FastPlaid CPU | `0.0034389139982522465` | `581.5789522554087` | `0.55` | `2997798` |

GPU candidate-score primitive:

- status: `ok`
- device: `nvidia:sm_89`
- score delta max abs: `0.0`
- two-pass kernel mean: `2.7561501607717042e-05`
- host-to-device + kernel + device-to-host mean:
  `6.505407685174666e-05`
- two-pass kernel candidate scores/sec: `9288318.308764486`
- kernel seconds / FastPlaid CPU full-search batch second:
  `0.00801459461380092`
- h2d+kernel+d2h seconds / FastPlaid CPU full-search batch second:
  `0.01891704092769083`

Interpretation:

- Kayak i8 CPU full search was about `5.67x` FastPlaid CPU QPS on this
  small synthetic shape at the same measured recall.
- The GPU candidate-score primitive is much smaller than a FastPlaid full
  search operation, so its ratio versus FastPlaid is not a backend speedup
  claim.

## FastPlaid CUDA Run

Latest verification command:

```bash
pixi run compare_gpu_i8_fastplaid_cuda_raw
```

Artifacts:

- report: `.cache/kayak/gpu_i8_fastplaid_compare/cuda_summary.json`
- prior quiet log: `.cache/kayak/bench_quiet/20260425T192335Z`

Results:

| Row | Query batch mean s | QPS | Recall@10 vs exact | Index bytes |
| --- | ---: | ---: | ---: | ---: |
| Kayak exact CPU | `0.08209100499880151` | `24.363205201705096` | `1.0` | `2099208` |
| Kayak i8 CPU | `0.0006045479967724532` | `3308.25676485168` | `0.5` | `575312` |
| FastPlaid CUDA | `0.0016751939983805642` | `1193.8915743092625` | `0.35` | `2997548` |

GPU candidate-score primitive:

- status: `ok`
- device: `nvidia:sm_89`
- score delta max abs: `0.0`
- two-pass kernel mean: `2.7635473110637288e-05`
- host-to-device + kernel + device-to-host mean:
  `6.508911235960225e-05`
- two-pass kernel candidate scores/sec: `9263456.390817566`
- kernel seconds / FastPlaid CUDA full-search batch second:
  `0.016496879249420022`
- h2d+kernel+d2h seconds / FastPlaid CUDA full-search batch second:
  `0.03885467141269901`

Interpretation:

- FastPlaid CUDA ran successfully on this host.
- Kayak i8 CPU full search was about `2.77x` FastPlaid CUDA QPS on this
  small synthetic shape, but this is a one-run smoke comparison, not a
  broad device conclusion.
- FastPlaid CUDA recall was `0.35` versus Kayak exact in this run, while Kayak
  i8 CPU recall was `0.5`.

## Decision

The next GPU step should still be the internal Kayak i8 payload integration,
not public API work.

Reason:

- the GPU primitive is correct on deterministic flat tensors
- FastPlaid CPU and CUDA comparison is now available in the same report shape
- the missing evidence is end-to-end Kayak GPU rerank timing on real i8
  candidate positions and token payloads

## Validation

Ran:

```bash
pixi run python -m py_compile \
  python/scripts/compare_gpu_i8_fastplaid.py \
  python/tests/test_gpu_i8_rerank_contract.py
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_gpu_i8_rerank_contract.GpuI8RerankContractTests.\
test_gpu_fastplaid_compare_keeps_scope_ratios_explicit -v
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_gpu_i8_rerank_contract -v
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_fastplaid_speed_track -v
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_sdist_manifest -v
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_repo_semantic_inventory -v
pixi run compare_gpu_i8_fastplaid_raw
pixi run compare_gpu_i8_fastplaid_cuda_raw
git diff --check
```

Observed:

- comparison script compile check passed
- focused scope-ratio unit test passed
- GPU i8 rerank contract tests: `23/23` passed
- FastPlaid speed-track tests: `8/8` passed
- sdist manifest tests: `3/3` passed
- repo semantic inventory tests: `3/3` passed
- `git diff --check` passed
- FastPlaid CPU comparison status: `ok` after rerunning without
  `PYTHONPATH=python`
- FastPlaid CUDA comparison status: `ok` after rerunning without
  `PYTHONPATH=python`
- both GPU candidate-score primitive rows reported `score_delta_max_abs=0.0`

Open validation still needed:

- repeat CPU and CUDA FastPlaid comparison runs across larger shapes
- compare against real encoded task tensors, not only synthetic deterministic
  tensors
- connect the GPU primitive to Kayak i8 payloads and measure true end-to-end
  rerank timing
