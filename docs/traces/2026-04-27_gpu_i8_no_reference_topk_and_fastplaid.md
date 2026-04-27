# 2026-04-27: GPU I8 No-Reference Top-K And FastPlaid Context

## Claim

The next GPU rerank boundary should be measured without validation-only CPU
reference scores in the serving call.

Reason: a production-shaped primitive must not require CPU scores as input.
Keeping CPU scores only for post-call validation separates correctness evidence
from the runtime surface we want to optimize.

## Changed

- Added an explicit prepared-handle no-reference top-k call.
- Added no-reference rows to the address serving sweep and FastPlaid comparison
  reports.
- Added warm-up calls before prepared-handle score/top-k measurement.
- Added a Python bridge shortcut for full-window i8 candidate generation when
  `candidate_k >= document_count`.
- Added a typed-address i8 candidate-position bridge for non-full candidate
  windows.

Reasons:

- no-reference top-k tests the serving boundary we actually want
- warm-up removes first-use timing from the measured score windows
- full candidate windows are exactly every document id, so calling Mojo to
  rediscover that set cannot improve correctness
- non-full candidate windows already receive contiguous `float32` query
  tensors, so typed-address ingestion avoids Python list materialization
  without changing candidate semantics

## Measurement

Commands:

```bash
pixi run profile_gpu_i8_address_serve_sweep
pixi run profile_gpu_i8_address_serve_wide_topk
pixi run compare_gpu_i8_fastplaid
pixi run compare_gpu_i8_fastplaid_cuda
pixi run compare_gpu_i8_fastplaid_wide_candidate1024
pixi run compare_gpu_i8_fastplaid_wide_candidate1024_cuda
```

Artifacts:

- default sweep:
  `.cache/kayak/gpu_i8_address_serve_sweep/summary.json`
- wide sweep:
  `.cache/kayak/gpu_i8_address_serve_sweep/wide_topk_summary.json`
- FastPlaid CPU default:
  `.cache/kayak/gpu_i8_fastplaid_compare/summary.json`
- FastPlaid CUDA default:
  `.cache/kayak/gpu_i8_fastplaid_compare/cuda_summary.json`
- FastPlaid CPU wide candidate1024:
  `.cache/kayak/gpu_i8_fastplaid_compare/wide_candidate1024_summary.json`
- FastPlaid CUDA wide candidate1024:
  `.cache/kayak/gpu_i8_fastplaid_compare/wide_candidate1024_cuda_summary.json`

Host GPU:

- device: `NVIDIA GeForce RTX 4070 Ti`
- target: `nvidia:sm_89`
- driver: `595.58.03`
- compute capability: `8.9`

## Sweep Results

Default sweep:

- status: `ok`
- ok cases: `6 / 6`
- no-reference top-k ratio versus CPU same-candidate score:
  `0.1449976424705887` best, `0.2644113072547268` worst
- CPU candidate generation plus no-reference GPU top-k ratio versus CPU
  candidate generation plus CPU score:
  `0.14568033681686754` best, `0.6357242169242013` worst
- no-reference top-k versus validating top-k:
  `0.9430245129324615` best, `1.027865882570198` worst

Wide sweep:

- status: `ok`
- ok cases: `5 / 5`
- no-reference top-k ratio versus CPU same-candidate score:
  `0.10384453516524908` best, `0.22531197916999116` worst
- CPU candidate generation plus no-reference GPU top-k ratio versus CPU
  candidate generation plus CPU score:
  `0.10462076765630066` best, `0.5107093380327875` worst
- no-reference top-k versus validating top-k:
  `0.990975993053825` best, `1.0060355196105975` worst

Wide row detail:

| case | docs | queries | query vecs | doc vecs | candidate k | CPU candidates/window s | no-ref GPU top-k/window s | CPU candidates + no-ref top-k/window s | envelope ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate512 | 512 | 2 | 8 | 16 | 512 | `0.0000010419166756037157` | `0.00010609549985929334` | `0.00010713741653489706` | `0.12038766406834586` |
| candidate1024 | 1024 | 2 | 8 | 16 | 1024 | `0.0000014676666827047786` | `0.00017580300004738092` | `0.0001772706667300857` | `0.10462076765630066` |
| query_vectors32 | 512 | 2 | 32 | 16 | 256 | `0.0009234700833834116` | `0.00018130549960915232` | `0.001104775582992564` | `0.4486646704302113` |
| doc_vectors64 | 512 | 2 | 8 | 64 | 256 | `0.0004266163333189373` | `0.00028493150034591963` | `0.0007115478336648569` | `0.4207292338959676` |
| query_batch4 | 512 | 4 | 8 | 16 | 256 | `0.0007736633333479404` | `0.00012395624980854336` | `0.0008976195831564837` | `0.5107093380327875` |

The full-window rows now spend about one to two microseconds per window in CPU
candidate generation. The typed-address candidate bridge reduced the
non-full-window rows, but they remain the candidate-generation optimization
target.

## FastPlaid Context

FastPlaid rows are full search. Kayak GPU rows start from CPU candidate windows
and measure an internal prepared-handle rerank/top-k boundary. The ratios below
are profiling context, not public backend speedup claims.

| case | FastPlaid device | FastPlaid batch s | FastPlaid recall@10 | Kayak public i8 batch s | Kayak public i8 recall@10 | no-ref GPU top-k/window s | CPU candidates + no-ref top-k/window s | envelope / FastPlaid batch | top-k agreement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | CPU | `0.003600052999900072` | `0.5` | `0.00040007600000535604` | `0.5` | `0.000054531750038222526` | `0.0002911779999976716` | `0.08088158702267825` | `1.0` |
| default | CUDA | `0.001682187000369595` | `0.5` | `0.0003929329996026354` | `0.5` | `0.00005373775024963834` | `0.00028989325028305757` | `0.17233116783054728` | `1.0` |
| wide candidate1024 | CPU | `0.007613905999278359` | `0.4` | `0.0011029470006178599` | `1.0` | `0.0001637027498873067` | `0.00016594949988757435` | `0.02179558033725435` | `1.0` |
| wide candidate1024 | CUDA | `0.0022121639995020814` | `0.44999999999999996` | `0.001101884000490827` | `1.0` | `0.00016423650004071533` | `0.00016627800005153404` | `0.07516531328100459` | `1.0` |

## Interpretation

Verified:

- no-reference top-k returns the same top-k positions as CPU i8 on every swept
  default and wide case
- full-window candidate generation is no longer a visible cost in the internal
  GPU pipeline
- typed-address query ingestion reduced non-full-window CPU candidate
  generation in the wide sweep
- on the measured default and wide FastPlaid comparison shapes, CPU candidate
  generation plus GPU no-reference top-k is faster than FastPlaid full search
  for both CPU and CUDA FastPlaid rows

Debunked:

- CPU reference-score input was not the main end-to-end top-k cost. Removing it
  reduced host marshalling, but total no-reference top-k time was within noise
  of validating top-k.

Still open:

- this does not prove a public GPU backend speedup
- candidate generation is still CPU-side for non-full candidate windows
- top-k selection is host-side inside the Mojo extension after score readback,
  not a GPU-resident top-k kernel
- FastPlaid recall differs on these deterministic synthetic shapes, so speed
  and recall must stay reported together

## Decision

Keep optimizing the internal candidate-window pipeline before adding public GPU
API surface.

Reason: the current primitive is strong enough to justify integration and more
profiling, but the next measurable bottleneck is non-full-window candidate
generation and the readback/top-k boundary, not validation-score marshalling.

## Validation

Ran:

```bash
pixi run env PYTHONPATH=python python -m unittest python/tests/test_gpu_i8_rerank_contract.py python/tests/test_fastplaid_speed_track.py
```

Observed:

- GPU i8 rerank contract and FastPlaid speed-track tests: `49/49` passed
- default sweep status: `ok`
- wide sweep status: `ok`
- FastPlaid CPU and CUDA comparisons status: `ok`
- FastPlaid wide CPU and CUDA comparisons status: `ok`
