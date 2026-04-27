# 2026-04-26: GPU I8 FastPlaid Wide Candidate1024 Compare

## Claim

The wider GPU top-k sweep needs a matching FastPlaid context row on at least one
wider shape.

Reason: the `wide_topk` sweep shows the internal prepared-handle top-k boundary
is still useful at larger candidate windows, but that evidence is incomplete
without the external full-search baseline on the same explicit vector counts.

## Added

- `compare_gpu_i8_fastplaid_wide_candidate1024_raw`
- `compare_gpu_i8_fastplaid_wide_candidate1024`
- `compare_gpu_i8_fastplaid_wide_candidate1024_cuda_raw`
- `compare_gpu_i8_fastplaid_wide_candidate1024_cuda`

Reason: these tasks keep the wide FastPlaid comparison reproducible instead of
requiring a long custom command line.

## Measurement

Commands:

```bash
pixi run compare_gpu_i8_fastplaid_wide_candidate1024
pixi run compare_gpu_i8_fastplaid_wide_candidate1024_cuda
```

Artifacts:

- CPU FastPlaid report:
  `.cache/kayak/gpu_i8_fastplaid_compare/wide_candidate1024_summary.json`
- CUDA FastPlaid report:
  `.cache/kayak/gpu_i8_fastplaid_compare/wide_candidate1024_cuda_summary.json`
- CPU quiet log:
  `.cache/kayak/bench_quiet/20260426T195308Z`
- CUDA quiet log:
  `.cache/kayak/bench_quiet/20260426T195326Z`

Common shape:

- documents: `1024`
- document vectors per document: `16`
- total document vectors: `16384`
- queries per window: `2`
- query vectors per query: `8`
- total query vectors per window: `16`
- candidate window: `1024`
- candidate scores per window: `2048`
- top-k: `10`
- top-k returns per window: `20`
- GPU target: `nvidia:sm_89`
- FastPlaid version: `1.4.6.2110`

## Results

| FastPlaid device | FastPlaid batch s | FastPlaid recall@10 | Kayak i8 CPU batch s | Kayak i8 CPU recall@10 | GPU top-k/window s | CPU candidates + GPU top-k/window s | top-k/FastPlaid batch | candidates+top-k/FastPlaid batch | top-k agreement | max delta | tolerance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | `0.007979334000992822` | `0.35` | `0.0011375520007277373` | `1.0` | `0.00021881324937567115` | `0.004825643999538443` | `0.02742249532961592` | `0.6047677661992863` | `1.0` | `0.0000762939453125` | `0.0001` |
| CUDA | `0.0025957119978556875` | `0.45` | `0.0011585309985093772` | `1.0` | `0.0002280552507727407` | `0.004840218501158233` | `0.08785845693248584` | `1.8646978190017722` | `1.0` | `0.0000762939453125` | `0.0001` |

Primitive-only GPU candidate-score context:

| FastPlaid device | GPU twopass kernel s | H2D s | D2H s |
| --- | ---: | ---: | ---: |
| CPU | `0.00012414520661157026` | `0.00011277091651031895` | `0.000004580645560435364` |
| CUDA | `0.00012398695661157023` | `0.0000948376547901821` | `0.0000046447351951417625` |

## Interpretation

Verified:

- prepared-handle top-k preserved `topk_position_agreement=1.0` on the wider
  `candidate1024` shape
- the isolated GPU top-k boundary was much smaller than both CPU and CUDA
  FastPlaid full-search batch time
- CPU candidate generation plus GPU top-k was faster than CPU FastPlaid full
  search on this shape
- CPU candidate generation plus GPU top-k was slower than CUDA FastPlaid full
  search on this shape

Not claimed:

- this is still not an apples-to-apples backend comparison
- the Kayak GPU row still starts after CPU candidate generation
- the current CUDA FastPlaid result has lower recall than Kayak i8 CPU on this
  synthetic shape, so speed and recall must be considered together

## Decision

Keep the wide FastPlaid tasks as external-baseline probes, but keep the next
GPU implementation focused on reducing or eliminating CPU candidate-generation
and score-readback cost.

Reason: the isolated GPU top-k boundary is not the end-to-end limiter on the
CUDA FastPlaid comparison; CPU candidate generation dominates the current
candidate-window envelope.

Follow-up: [2026-04-27 GPU I8 Candidate-Generation Cleanup](2026-04-27_gpu_i8_candidate_generation_cleanup.md)
added a full-window candidate shortcut and bounded heap selection. On the same
wide `candidate1024` shape, the updated CPU-candidates-plus-GPU-top-k boundary
measured about `0.000793s/window` against FastPlaid CUDA full search at about
`0.002271s/batch`. This remains a scope-limited comparison, not a public GPU
backend claim.

## Validation

Ran:

```bash
pixi run compare_gpu_i8_fastplaid_wide_candidate1024
pixi run compare_gpu_i8_fastplaid_wide_candidate1024_cuda
```

Observed:

- CPU FastPlaid wide comparison status: `ok`
- CUDA FastPlaid wide comparison status: `ok`
- each quiet report emitted `7` sections
