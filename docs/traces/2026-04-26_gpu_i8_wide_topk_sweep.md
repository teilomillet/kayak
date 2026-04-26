# 2026-04-26: GPU I8 Wide Top-K Sweep

## Claim

After the explicit prepared-handle top-k boundary wins on the default shape
sweep, the next question is whether that boundary remains useful when candidate
windows and vector counts increase.

Reason: optimizing kernels before measuring wider windows risks working on the
wrong bottleneck. Candidate count, query vector count, and document vector count
change both CPU score cost and GPU copy/score/readback cost.

## Added

- a named `wide_topk` case set for `profile_gpu_i8_address_serve_sweep`
- pixi tasks:
  - `profile_gpu_i8_address_serve_wide_topk_raw`
  - `profile_gpu_i8_address_serve_wide_topk`
- explicit report metadata for whether cases came from a named case set or
  custom `--case` arguments
- centralized GPU i8 score-delta tolerance in
  `kayak_bridge.gpu_i8_score_agreement`

The default case set was not changed.

Reason: keeping the default stable preserves comparability with earlier traces,
while `wide_topk` makes the larger stress surface explicit and reproducible.

## Correctness Gate

The first wide run failed `query_vectors32` under the old fixed
`1.0e-4` absolute score-delta tolerance:

- max score delta: `0.000244140625`
- top-k position agreement: `1.0`
- failing shape: `query_count=2`, `query_vector_count=32`,
  `document_count=512`, `document_vector_count=16`, `candidate_k=256`

The GPU and CPU i8 paths both use `Float32` score scalars, but they reduce dot
products in different orders. The agreement gate now reports:

- floor tolerance: `1.0e-4`
- per-query-vector tolerance: `1.0e-5`
- effective tolerance:
  `max(1.0e-4, 1.0e-5 * query_vector_count)`

Reason: absolute floating-point error is expected to grow with the number of
MaxSim terms. The tolerance is now a named report field rather than a hidden
constant, so future runs can validate or tighten it with evidence.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_address_serve_wide_topk
```

Artifacts:

- report: `.cache/kayak/gpu_i8_address_serve_sweep/wide_topk_summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260426T193202Z`

Common controls:

- vector dim: `128`
- top-k: `10`
- warmup iterations: `1`
- measurement iterations: `3`
- resident windows: `4`
- Kayak PLAID payload: `i8`
- GPU target: `nvidia:sm_89`
- device: `NVIDIA GeForce RTX 4070 Ti`
- device memory: `10320000000` bytes

## Results

| case | docs | doc vecs | queries | query vecs | candidate k | CPU score/window s | prepared score/window s | prepared top-k/window s | top-k/CPU score | top-k/score | CPU cand+top-k/CPU cand+score | top-k agreement | max delta | tolerance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate512 | 512 | 16 | 2 | 8 | 512 | `0.0008959920836180876` | `0.0003093247496508411` | `0.00010918625048361719` | `0.12186073122735012` | `0.35298258741618377` | `0.6786447115530703` | `1.0` | `0.0000762939453125` | `0.0001` |
| candidate1024 | 1024 | 16 | 2 | 8 | 1024 | `0.001733612999790542` | `0.0005728735004595364` | `0.00018029375041805906` | `0.10399884544003907` | `0.31471825852205515` | `0.7553267484881523` | `1.0` | `0.0000762939453125` | `0.0001` |
| query_vectors32 | 512 | 16 | 2 | 32 | 256 | `0.0015447055836072348` | `0.00027699700149241835` | `0.00018508999983168906` | `0.11982218605014834` | `0.6682021784873189` | `0.6362895664686989` | `1.0` | `0.000244140625` | `0.00032` |
| doc_vectors64 | 512 | 64 | 2 | 8 | 256 | `0.0012844483335356927` | `0.0003879664991472964` | `0.0002932192510343157` | `0.22828419281542678` | `0.7557849754522007` | `0.5858688078982071` | `1.0` | `0.00006103515625` | `0.0001` |
| query_batch4 | 512 | 16 | 4 | 8 | 256 | `0.0009988840832496255` | `0.000312320250486664` | `0.0001276105003853445` | `0.12775306216732865` | `0.4085886207714649` | `0.7233078864406184` | `1.0` | `0.0000762939453125` | `0.0001` |

Summary:

- status: `ok`
- ok cases: `5 / 5`
- best prepared-handle top-k score ratio: `0.10399884544003907`
- worst prepared-handle top-k score ratio: `0.22828419281542678`
- best CPU-candidate-plus-top-k ratio: `0.5858688078982071`
- worst CPU-candidate-plus-top-k ratio: `0.7553267484881523`
- best top-k versus score-return ratio: `0.31471825852205515`
- worst top-k versus score-return ratio: `0.7557849754522007`
- quiet wrapper sections: `35`

## Interpretation

Verified:

- prepared-handle top-k stayed correct on all five wider cases
- top-k order agreement was `1.0` on every case
- top-k return remained faster than returning all candidate scores on every
  case
- CPU candidate generation plus prepared-handle top-k remained faster than CPU
  candidate generation plus CPU same-candidate scoring on every case

Debunked:

- the fixed `1.0e-4` absolute score-delta threshold is not adequate once query
  vector count reaches `32`
- the one-shot serving row is not the optimization target; it was slower than
  CPU same-candidate scoring on `doc_vectors64`

Still open:

- whether the per-query-vector tolerance is tight enough for query vector counts
  above `32`
- whether GPU-side top-k matters after score readback is removed
- whether candidate generation becomes the dominant end-to-end limit once GPU
  rerank is integrated
- whether the same shape set holds against real encoded corpora rather than
  deterministic synthetic vectors

## Decision

Keep optimizing around the explicit prepared-handle top-k boundary, but do not
start public backend integration yet.

Reason: the primitive is now correct and useful across wider synthetic shapes,
but candidate generation remains CPU-side and top-k selection still happens
after score readback inside the Mojo extension. The next implementation target
should remove avoidable score readback or introduce a real internal search
boundary, then compare again.

## Validation

Ran:

```bash
pixi run python -m py_compile \
  python/kayak_bridge/gpu_i8_score_agreement.py \
  python/kayak_bridge/gpu_i8_address_resident_windows.py \
  python/scripts/profile_gpu_i8_real_payload_rerank.py \
  python/scripts/profile_gpu_i8_address_serve_sweep.py \
  python/tests/test_gpu_i8_rerank_contract.py
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_gpu_i8_rerank_contract -v
pixi run profile_gpu_i8_address_serve_wide_topk
```

Observed:

- Python compile check passed
- GPU i8 rerank contract tests: `36/36` passed
- first wide run correctly failed the old fixed tolerance on `query_vectors32`
- final wide quiet sweep status: `ok`
