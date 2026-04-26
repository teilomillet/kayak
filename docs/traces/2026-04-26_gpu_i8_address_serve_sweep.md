# 2026-04-26: GPU I8 Address Serving Shape Sweep

## Claim

The no-internal-benchmark typed-address GPU serving call and an in-call
resident-session variant should be swept across explicit vector-count shapes
before changing kernels.

Reason: the first serving call was faster than CPU same-candidate scoring on
one smoke shape, but that does not show whether the win survives smaller
candidate windows, larger prepared indexes, or more query vectors. The
resident-session variant tests whether prepared-index residency is likely to
matter before introducing a reusable GPU object.

## Added

- `python/scripts/profile_gpu_i8_address_serve_sweep.py`
- `profile_gpu_i8_address_serve_sweep_raw`
- `profile_gpu_i8_address_serve_sweep`
- contract tests for case parsing and reported ratio semantics
- an in-call resident-session probe that copies the prepared index once and
  scores the same candidate window repeatedly
- `--resident-session-iterations`

Reason: this keeps the sweep separate from the broader real-payload profiler.
The sweep measures one narrow question: CPU i8 same-candidate scoring versus
the internal GPU typed-address serving call over the same CPU-generated
candidate windows. The resident-session row is deliberately still internal and
single-call so it can test the ownership hypothesis without a hidden global
cache.

## Boundary

Each case reports:

- document count
- document vectors per document
- total document vectors
- query count
- query vectors per query
- candidate window
- candidate score count
- CPU i8 candidate generation timing
- CPU i8 same-candidate score timing
- GPU address serving extension-call timing
- GPU address resident-session extension-call timing per repeated iteration
- CPU candidate generation plus GPU serving-call envelope
- CPU candidate generation plus GPU resident-iteration envelope
- score agreement against CPU i8

It does not:

- reuse a GPU-resident prepared index across Python calls
- include CPU or GPU top-k timing
- compare against FastPlaid full search in this row
- expose a public GPU backend

The resident-session row does:

- copy token codes, token scales, and document offsets to device once inside
  one extension call
- reuse those device buffers for `resident_session_iterations`
- repeat the same query and candidate window each iteration

Reason: this is still an internal primitive measurement. It should not be
presented as full search speedup or as proof of safe cross-call GPU object
ownership.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_address_serve_sweep
```

Artifacts:

- report: `.cache/kayak/gpu_i8_address_serve_sweep/summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260426T163133Z`

Common controls:

- vector dim: `128`
- query count: `2`
- top-k: `10`
- warmup iterations: `1`
- measurement iterations: `3`
- resident-session iterations: `4`
- Kayak PLAID payload: `i8`
- GPU target: `nvidia:sm_89`

## Results

| case | docs | doc vecs | query vecs | candidate_k | candidate scores | CPU score s | GPU call s | GPU/CPU score | resident iter s | resident/CPU score | CPU cand+GPU ratio | CPU cand+resident ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 256 | 16 | 8 | 128 | 256 | `0.00030123866721017595` | `0.00026536500020786963` | `0.8809128079919533` | `0.00012104825009373599` | `0.4018350340438863` | `0.9561369398243865` | `0.77967953182653` |
| candidate32 | 256 | 16 | 8 | 32 | 64 | `0.00014605566684622318` | `0.00019923866724032754` | `1.364128291236374` | `0.00009359975001643761` | `0.6408498351179038` | `1.1040762536281072` | `0.897346609559154` |
| candidate256 | 256 | 16 | 8 | 256 | 512 | `0.0004976920002566961` | `0.0003059270002268022` | `0.6146914157129576` | `0.0001437957498637843` | `0.28892517820181624` | `0.8364465058365654` | `0.6981671925842801` |
| query_vectors16 | 256 | 16 | 16 | 128 | 256 | `0.000516620667137128` | `0.00027080866614899907` | `0.5241924750120723` | `0.00014524100015478325` | `0.28113664317697834` | `0.8197402854755539` | `0.7276585663787248` |
| doc_vectors32 | 256 | 32 | 8 | 128 | 256 | `0.0004179466662511307` | `0.00041605666653291945` | `0.9954778925857597` | `0.00021737824999945587` | `0.5201100225281862` | `0.9979952107241836` | `0.7872500159156838` |
| documents512 | 512 | 16 | 8 | 128 | 256 | `0.00030242100062120397` | `0.00037998566585883964` | `1.256479097279322` | `0.00017487625018475228` | `0.5782543203862774` | `1.07590960799445` | `0.875176770534385` |

Summary:

- ok cases: `6 / 6`
- best isolated GPU ratio: `0.5241924750120723`
- worst isolated GPU ratio: `1.364128291236374`
- best resident-iteration GPU ratio: `0.28113664317697834`
- worst resident-iteration GPU ratio: `0.6408498351179038`
- best CPU-candidate-plus-GPU ratio: `0.8197402854755539`
- worst CPU-candidate-plus-GPU ratio: `1.1040762536281072`
- best CPU-candidate-plus-resident-iteration ratio: `0.6981671925842801`
- worst CPU-candidate-plus-resident-iteration ratio: `0.897346609559154`
- maximum observed score delta: `9.1552734375e-05`

## Interpretation

Verified:

- the one-shot address serving call and the resident-session call preserved CPU
  i8 score agreement across all swept shapes
- isolated GPU scoring wins when the candidate-score work is large enough, such
  as `candidate256` and `query_vectors16`
- small candidate windows are too small for the current serving-call overhead
- larger prepared indexes move the current one-shot call toward break-even or
  loss because it still allocates and copies token codes, scales, and offsets
  inside every serving call
- the resident-session iteration is faster than CPU same-candidate scoring in
  every swept case
- the CPU-candidate-generation-plus-resident-iteration envelope is faster than
  CPU candidate generation plus CPU same-candidate scoring in every swept case

Debunked:

- the serving-shaped call is not uniformly faster than CPU i8 scoring
- kernel math is not the only or obvious next optimization target
- one-shot typed-address ingestion alone is not enough for the smaller or
  larger-index cases

Still open:

- the result after removing repeated prepared-index copies across real Python
  calls
- the result after reusing device allocations across calls with a safe ownership
  model
- the cost of returning all scores versus doing GPU or Mojo-side top-k
- the full candidate-generation plus score plus top-k envelope
- whether repeated different query/candidate windows behave like the repeated
  same-window resident-session measurement

## Decision

Design and test a real internal prepared-index ownership boundary before
editing kernel math.

Reason: the shape sweep points to fixed serving overhead and repeated
prepared-index movement. Optimizing the two-pass dim128 kernel first would not
address the losing one-shot `candidate32`, `doc_vectors32`, or `documents512`
cases, while the resident-session row shows that amortizing prepared-index copy
and allocation can make all swept cases favorable.

## Validation

Ran:

```bash
pixi run mojo format python/kayak_bridge/_mojo_gpu_i8_rerank_bindings.mojo
pixi run python -m py_compile \
  python/kayak_bridge/mojo_gpu_i8_rerank.py \
  python/kayak_bridge/gpu_i8_address_serve_sweep.py \
  python/kayak_bridge/gpu_i8_address_serve_sweep_runner.py \
  python/scripts/profile_gpu_i8_address_serve_sweep.py \
  python/tests/test_gpu_i8_rerank_contract.py
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_gpu_i8_rerank_contract -v
pixi run profile_gpu_i8_address_serve_sweep_raw
pixi run profile_gpu_i8_address_serve_sweep
```

Observed:

- Mojo format completed with file unchanged
- Python compile check passed
- GPU i8 rerank contract tests: `28/28` passed
- raw address serving sweep status: `ok`
- quiet address serving sweep status: `ok`
- quiet wrapper emitted `24` sections, including resident-session iteration
  timing for all six cases
