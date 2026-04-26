# 2026-04-26: GPU I8 Address Serving Shape Sweep

## Claim

The no-internal-benchmark typed-address GPU serving call should be swept across
explicit vector-count shapes before changing kernels.

Reason: the first serving call was faster than CPU same-candidate scoring on
one smoke shape, but that does not show whether the win survives smaller
candidate windows, larger prepared indexes, or more query vectors.

## Added

- `python/scripts/profile_gpu_i8_address_serve_sweep.py`
- `profile_gpu_i8_address_serve_sweep_raw`
- `profile_gpu_i8_address_serve_sweep`
- contract tests for case parsing and reported ratio semantics

Reason: this keeps the sweep separate from the broader real-payload profiler.
The sweep measures one narrow question: CPU i8 same-candidate scoring versus
the internal GPU typed-address serving call over the same CPU-generated
candidate windows.

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
- CPU candidate generation plus GPU serving-call envelope
- score agreement against CPU i8

It does not:

- reuse a GPU-resident prepared index across calls
- include CPU or GPU top-k timing
- compare against FastPlaid full search in this row
- expose a public GPU backend

Reason: this is still an internal primitive measurement. It should not be
presented as full search speedup.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_address_serve_sweep
```

Artifacts:

- report: `.cache/kayak/gpu_i8_address_serve_sweep/summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260426T160724Z`

Common controls:

- vector dim: `128`
- query count: `2`
- top-k: `10`
- warmup iterations: `1`
- measurement iterations: `3`
- Kayak PLAID payload: `i8`
- GPU target: `nvidia:sm_89`

## Results

| case | docs | doc vecs | query vecs | candidate_k | candidate scores | CPU score s | GPU call s | GPU/CPU score | CPU candidate+GPU / CPU candidate+score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 256 | 16 | 8 | 128 | 256 | `0.0002986399998311147` | `0.0002622959997703826` | `0.8783016338022864` | `0.9554511938519803` |
| candidate32 | 256 | 16 | 8 | 32 | 64 | `0.00014617266666997844` | `0.00020376700013002846` | `1.3940157539171376` | `1.1121702404619311` |
| candidate256 | 256 | 16 | 8 | 256 | 512 | `0.0004947393335896777` | `0.0003008413335313283` | `0.608080484218862` | `0.8330091278859598` |
| query_vectors16 | 256 | 16 | 16 | 128 | 256 | `0.0005133380000188481` | `0.00025810133350508596` | `0.5027902346906119` | `0.8127284765890352` |
| doc_vectors32 | 256 | 32 | 8 | 128 | 256 | `0.0004182303334043051` | `0.00040638166653176694` | `0.9716695181430468` | `0.9873777714351207` |
| documents512 | 512 | 16 | 8 | 128 | 256 | `0.0002949136663422299` | `0.0003729996663726827` | `1.2647757935362638` | `1.077044030889416` |

Summary:

- ok cases: `6 / 6`
- best isolated GPU ratio: `0.5027902346906119`
- worst isolated GPU ratio: `1.3940157539171376`
- best CPU-candidate-plus-GPU ratio: `0.8127284765890352`
- worst CPU-candidate-plus-GPU ratio: `1.1121702404619311`
- maximum observed score delta: `9.1552734375e-05`

## Interpretation

Verified:

- the address serving call preserved CPU i8 score agreement across all swept
  shapes
- isolated GPU scoring wins when the candidate-score work is large enough, such
  as `candidate256` and `query_vectors16`
- small candidate windows are too small for the current serving-call overhead
- larger prepared indexes move the current call toward break-even or loss
  because it still allocates and copies token codes, scales, and offsets inside
  every serving call

Debunked:

- the serving-shaped call is not uniformly faster than CPU i8 scoring
- kernel math is not the only or obvious next optimization target

Still open:

- the result after removing repeated prepared-index copies
- the result after reusing device allocations across calls
- the cost of returning all scores versus doing GPU or Mojo-side top-k
- the full candidate-generation plus score plus top-k envelope

## Decision

Design and test a real internal prepared-index ownership boundary before
editing kernel math.

Reason: the shape sweep points to fixed serving overhead and repeated
prepared-index movement. Optimizing the two-pass dim128 kernel first would not
address the losing `candidate32`, `doc_vectors32`, or `documents512` cases.

## Validation

Ran:

```bash
pixi run python -m py_compile \
  python/scripts/profile_gpu_i8_address_serve_sweep.py \
  python/tests/test_gpu_i8_rerank_contract.py
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_gpu_i8_rerank_contract -v
pixi run profile_gpu_i8_address_serve_sweep_raw
pixi run profile_gpu_i8_address_serve_sweep
```

Observed:

- Python compile check passed
- GPU i8 rerank contract tests: `27/27` passed
- raw address serving sweep status: `ok`
- quiet address serving sweep status: `ok`
