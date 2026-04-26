# 2026-04-26: GPU I8 Explicit Handle Top-K Return

## Claim

After cross-call GPU index residency is working, the next boundary to measure is
score return shape: returning only top-k positions and scores may be cheaper
than returning every candidate score.

Reason: the previous explicit-handle row still returned one Python float per
candidate score. For a serving path, the caller usually needs the top-k
positions, not the complete candidate score matrix.

## Added

- `score_i8_address_session_handle_topk(...)` in
  `python/kayak_bridge/_mojo_gpu_i8_rerank_bindings.mojo`
- `MojoGpuI8AddressTopKResult`
- `MojoGpuI8AddressSessionHandle.score_topk(...)`
- a `gpu_address_prepared_handle_topk_session` row in the address serving
  sweep

The top-k row:

- reuses the explicit prepared GPU handle
- scores one query/candidate window per Python call
- reads candidate scores back to a reusable host buffer
- selects top-k positions inside the Mojo extension
- returns only `[query_count, top_k]` positions and scores
- validates score delta and top-k order agreement against CPU i8

It does not:

- perform GPU-side top-k
- move candidate generation to GPU
- expose a public GPU backend

Reason: this isolates the return-shape and top-k boundary before changing the
kernel or search planner.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_address_serve_sweep
```

Artifacts:

- report: `.cache/kayak/gpu_i8_address_serve_sweep/summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260426T182201Z`

Common controls:

- vector dim: `128`
- query count per window: `2`
- top-k: `10`
- warmup iterations: `1`
- measurement iterations: `3`
- resident windows: `4`
- Kayak PLAID payload: `i8`
- GPU target: `nvidia:sm_89`

## Results

| case | prepared score/window s | prepared top-k/window s | top-k/CPU score | top-k/score | CPU cand+top-k/CPU cand+score | top-k agreement | max delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `0.00010877524937313865` | `0.0000629147498329985` | `0.21472555894320483` | `0.5783921452312928` | `0.7222092766252833` | `1.0` | `0.00006103515625` |
| candidate32 | `0.00004826000031243893` | `0.000044522750158648705` | `0.3001565758850426` | `0.9225600884874641` | `0.8043717309441957` | `1.0` | `0.0000457763671875` |
| candidate256 | `0.00017242199965039617` | `0.000078208500326582` | `0.15912253456197387` | `0.4535877120388233` | `0.6494656692255495` | `1.0` | `0.0000762939453125` |
| query_vectors16 | `0.0001295017500524409` | `0.00008283700026368024` | `0.15960840238720292` | `0.6396593114003164` | `0.6854017262252963` | `1.0` | `0.000091552734375` |
| doc_vectors32 | `0.00015258200073731132` | `0.00010739300023487885` | `0.25720784816679376` | `0.7038379344610188` | `0.6768545248098907` | `1.0` | `0.00006103515625` |
| documents512 | `0.0001054442500389996` | `0.00006362899966916302` | `0.2150673430232824` | `0.6034373580885559` | `0.7767211783732696` | `1.0` | `0.00006103515625` |

Summary:

- ok cases: `6 / 6`
- best prepared-handle top-k score ratio: `0.15912253456197387`
- worst prepared-handle top-k score ratio: `0.3001565758850426`
- best CPU-candidate-plus-top-k ratio: `0.6494656692255495`
- worst CPU-candidate-plus-top-k ratio: `0.8043717309441957`
- best top-k versus score-return ratio: `0.4535877120388233`
- worst top-k versus score-return ratio: `0.9225600884874641`
- quiet wrapper sections: `42`

## Interpretation

Verified:

- top-k return preserves CPU i8 score agreement and top-k order agreement on
  all swept shapes
- returning top-k positions and scores is faster than returning every candidate
  score on all swept shapes
- CPU candidate generation plus prepared-handle top-k remains faster than CPU
  candidate generation plus CPU same-candidate scoring on every swept shape

Debunked:

- returning every candidate score is not the best measured handle interface for
  the swept shapes
- GPU-side top-k is not required before seeing a return-shape win

Still open:

- top-k timing on larger candidate windows
- the cost of CPU candidate generation at wider windows
- whether a GPU-side top-k selector matters after wider-window measurement
- integration into a full internal rerank call that returns top-k positions
- repeated FastPlaid comparisons on larger and real encoded shapes

## Decision

Use top-k return as the next internal serving-shaped handle boundary.

Reason: it is correct, narrower, and faster than score-return on the current
shape sweep. The next measurement should widen candidate windows before kernel
rewrites, because the return-shape result changes the bottleneck profile.

## Validation

Ran:

```bash
pixi run mojo format python/kayak_bridge/_mojo_gpu_i8_rerank_bindings.mojo
pixi run python -m py_compile \
  python/kayak_bridge/mojo_gpu_i8_rerank.py \
  python/kayak_bridge/gpu_i8_address_resident_windows.py \
  python/kayak_bridge/gpu_i8_address_serve_sweep.py \
  python/kayak_bridge/gpu_i8_address_serve_sweep_runner.py \
  python/scripts/profile_gpu_i8_address_serve_sweep.py \
  python/tests/test_gpu_i8_rerank_contract.py
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_gpu_i8_rerank_contract -v
pixi run profile_gpu_i8_address_serve_sweep_raw \
  --case smoke:documents=64,document_vectors=8,queries=1,query_vectors=4,candidate_k=16 \
  --measurement-iterations 1 \
  --resident-session-iterations 2 \
  --output .cache/kayak/gpu_i8_address_serve_sweep/topk_smoke.json
pixi run profile_gpu_i8_address_serve_sweep
```

Observed:

- Mojo format completed
- Python compile check passed
- GPU i8 rerank contract tests: `31/31` passed
- explicit-handle top-k smoke status: `ok`
- quiet full sweep status: `ok`
- quiet wrapper emitted `42` sections

## FastPlaid Follow-Up

The top-k handle boundary is now included in the FastPlaid comparison report.
See `docs/traces/2026-04-26_gpu_i8_fastplaid_topk_compare.md`.

Quiet summary:

- CPU FastPlaid report top-k/window:
  `0.00010890049907175126 s`
- CPU FastPlaid report CPU candidates plus GPU top-k/window:
  `0.0006470917487604311 s`
- CUDA FastPlaid report top-k/window:
  `0.00011942050059587928 s`
- CUDA FastPlaid report CPU candidates plus GPU top-k/window:
  `0.0006539250007335795 s`

Both comparison reports kept `topk_position_agreement=1.0`.
