# 2026-04-27: GPU I8 Selected-Budget FastPlaid Compare

## Claim

The centroid-budget sweep should feed a same-shape FastPlaid comparison before
any default policy changes.

Reason: the sweep showed lower `centroids_per_query_vector` budgets can reduce
posting fanout without losing baseline recall, but it did not by itself show
whether the resulting Kayak GPU rerank envelope is still competitive against
FastPlaid.

## Measurement

Selected budget rows came from the non-full wide centroid-budget sweep:

- `query_vectors32`: seed `7`, `centroids_per_query_vector=4`
- `doc_vectors64`: seed `8`, `centroids_per_query_vector=16`
- `query_batch4`: seed `9`, `centroids_per_query_vector=8`

Commands:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 20 --force -- pixi run compare_gpu_i8_fastplaid_raw --document-count 512 --document-vector-count 16 --query-count 2 --query-vector-count 32 --candidate-k 256 --kayak-plaid-centroids-per-query-vector 4 --output .cache/kayak/gpu_i8_fastplaid_compare/query_vectors32_cpqv4_summary.json --index-root .cache/kayak/gpu_i8_fastplaid_compare/query_vectors32_cpqv4_indexes
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 20 --force -- pixi run compare_gpu_i8_fastplaid_cuda_raw --document-count 512 --document-vector-count 16 --query-count 2 --query-vector-count 32 --candidate-k 256 --kayak-plaid-centroids-per-query-vector 4 --output .cache/kayak/gpu_i8_fastplaid_compare/query_vectors32_cpqv4_cuda_summary.json --index-root .cache/kayak/gpu_i8_fastplaid_compare/query_vectors32_cpqv4_cuda_indexes
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 20 --force -- pixi run compare_gpu_i8_fastplaid_raw --seed 8 --document-count 512 --document-vector-count 64 --query-count 2 --query-vector-count 8 --candidate-k 256 --kayak-plaid-centroids-per-query-vector 16 --output .cache/kayak/gpu_i8_fastplaid_compare/doc_vectors64_cpqv16_seed8_summary.json --index-root .cache/kayak/gpu_i8_fastplaid_compare/doc_vectors64_cpqv16_seed8_indexes
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 20 --force -- pixi run compare_gpu_i8_fastplaid_cuda_raw --seed 8 --document-count 512 --document-vector-count 64 --query-count 2 --query-vector-count 8 --candidate-k 256 --kayak-plaid-centroids-per-query-vector 16 --output .cache/kayak/gpu_i8_fastplaid_compare/doc_vectors64_cpqv16_seed8_cuda_summary.json --index-root .cache/kayak/gpu_i8_fastplaid_compare/doc_vectors64_cpqv16_seed8_cuda_indexes
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 20 --force -- pixi run compare_gpu_i8_fastplaid_raw --seed 9 --document-count 512 --document-vector-count 16 --query-count 4 --query-vector-count 8 --candidate-k 256 --kayak-plaid-centroids-per-query-vector 8 --output .cache/kayak/gpu_i8_fastplaid_compare/query_batch4_cpqv8_seed9_summary.json --index-root .cache/kayak/gpu_i8_fastplaid_compare/query_batch4_cpqv8_seed9_indexes
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 20 --force -- pixi run compare_gpu_i8_fastplaid_cuda_raw --seed 9 --document-count 512 --document-vector-count 16 --query-count 4 --query-vector-count 8 --candidate-k 256 --kayak-plaid-centroids-per-query-vector 8 --output .cache/kayak/gpu_i8_fastplaid_compare/query_batch4_cpqv8_seed9_cuda_summary.json --index-root .cache/kayak/gpu_i8_fastplaid_compare/query_batch4_cpqv8_seed9_cuda_indexes
```

Artifacts:

- `.cache/kayak/bench_quiet/20260427T093213Z`
- `.cache/kayak/bench_quiet/20260427T093323Z`
- `.cache/kayak/bench_quiet/20260427T093441Z`
- `.cache/kayak/bench_quiet/20260427T093512Z`
- `.cache/kayak/bench_quiet/20260427T093540Z`
- `.cache/kayak/bench_quiet/20260427T093603Z`

The first CUDA attempt inside the default sandbox failed with PyTorch reporting
no CUDA GPUs. The escalated run passed and is the recorded artifact.

## Results

All six selected-budget FastPlaid comparisons reported status `ok`.

| case | FastPlaid device | seed | cpqv | Kayak i8 recall | FastPlaid recall | CPU candidates + GPU no-ref top-k/window s | FastPlaid batch s | envelope / FastPlaid |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `cpu` | `7` | `4` | `0.70` | `0.40` | `0.0005746920005549327` | `0.0139461239996308` | `0.04120800880374696` |
| `query_vectors32` | `cuda` | `7` | `4` | `0.70` | `0.35` | `0.0005659562502842164` | `0.0024931709995144047` | `0.22700258040641727` |
| `doc_vectors64` | `cpu` | `8` | `16` | `0.65` | `0.65` | `0.0006430769994949515` | `0.025221355999747175` | `0.025497320584246058` |
| `doc_vectors64` | `cuda` | `8` | `16` | `0.65` | `0.60` | `0.0006441089999498217` | `0.0049050600009650225` | `0.13131521323349757` |
| `query_batch4` | `cpu` | `9` | `8` | `0.70` | `0.60` | `0.0006865057503091521` | `0.013113901999531663` | `0.052349464738540005` |
| `query_batch4` | `cuda` | `9` | `8` | `0.70` | `0.65` | `0.0006831992495790473` | `0.003898512000887422` | `0.17524615787344763` |

All rows reported no-reference top-k position agreement `1.0` against the CPU
i8 same-candidate reference.

## Interpretation

Verified:

- selected-budget non-full windows remain faster than FastPlaid CPU full search
  in the scoped CPU-candidate-generation-plus-GPU-top-k envelope
- selected-budget non-full windows also remain faster than FastPlaid CUDA full
  search on the same synthetic shapes
- Kayak i8 recall matched or exceeded the measured FastPlaid recall on all six
  rows
- the no-reference top-k serving boundary preserved CPU i8 top-k agreement

Not claimed:

- this is not a public GPU backend result
- this is not an apples-to-apples backend comparison because Kayak still starts
  from CPU-generated candidate windows and FastPlaid is measured as full search
- this does not prove a deployable static centroid-budget default
- this does not prove the same frontier on real encoded corpora

## Decision

Use selected-budget non-full shapes as the next FastPlaid comparison surface,
but keep the boundary internal.

Reason: the evidence now supports a stronger optimization direction than direct
GPU candidate-generation porting: shape-aware candidate-budget policy plus the
existing GPU no-reference top-k primitive. The missing step is turning that
policy into a reproducible calibration contract rather than hard-coding the
best budget observed on a small synthetic sweep.
