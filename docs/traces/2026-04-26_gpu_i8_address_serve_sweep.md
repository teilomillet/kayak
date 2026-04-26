# 2026-04-26: GPU I8 Address Serving Shape Sweep

## Claim

The no-internal-benchmark typed-address GPU serving call, same-window
resident-session variant, and multi-window resident-session variant should be
swept across explicit vector-count shapes before changing kernels.

Reason: the first serving call was faster than CPU same-candidate scoring on
one smoke shape, but that does not show whether the win survives smaller
candidate windows, larger prepared indexes, or more query vectors. The
resident rows test whether prepared-index residency is likely to matter before
introducing a reusable GPU object, and whether the result survives different
query/candidate windows rather than only repeating one window.

## Added

- `python/scripts/profile_gpu_i8_address_serve_sweep.py`
- `python/kayak_bridge/gpu_i8_address_resident_windows.py`
- `profile_gpu_i8_address_serve_sweep_raw`
- `profile_gpu_i8_address_serve_sweep`
- contract tests for case parsing and reported ratio semantics
- an in-call resident-session probe that copies the prepared index once and
  scores the same candidate window repeatedly
- an in-call multi-window resident-session probe that copies the prepared index
  once and scores different query/candidate windows
- `--resident-session-iterations`

Reason: this keeps the sweep separate from the broader real-payload profiler.
The sweep measures one narrow question: CPU i8 same-candidate scoring versus
internal GPU typed-address serving rows over CPU-generated candidate windows.
The resident rows are deliberately still internal and single-call so they can
test the ownership hypothesis without a hidden global cache.

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
- CPU i8 multi-window candidate generation and score timing per window
- GPU address serving extension-call timing
- GPU address resident-session extension-call timing per repeated iteration
- GPU address resident multi-window timing per window
- CPU candidate generation plus GPU serving-call envelope
- CPU candidate generation plus GPU resident-session envelopes
- score agreement against CPU i8

It does not:

- reuse a GPU-resident prepared index across Python calls
- include CPU or GPU top-k timing
- compare against FastPlaid full search in this row
- expose a public GPU backend

The resident rows do:

- copy token codes, token scales, and document offsets to device once inside
  one extension call
- reuse those device buffers for `resident_session_iterations`
- measure both repeated same-window scoring and different-window scoring

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
- quiet log: `.cache/kayak/bench_quiet/20260426T165046Z`

Common controls:

- vector dim: `128`
- query count per window: `2`
- top-k: `10`
- warmup iterations: `1`
- measurement iterations: `3`
- resident-session iterations/windows: `4`
- Kayak PLAID payload: `i8`
- GPU target: `nvidia:sm_89`

## Results

| case | docs | doc vecs | query vecs | candidate_k | candidate scores/window | CPU score s | GPU call s | GPU/CPU score | same-window s | same-window/CPU | multi-window s | multi-window/CPU multi score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 256 | 16 | 8 | 128 | 256 | `0.0002985406663356116` | `0.00026544533344955806` | `0.8891429657062243` | `0.00012076800021532108` | `0.4045278041938744` | `0.00014348274999065325` | `0.4874978022212287` |
| candidate32 | 256 | 16 | 8 | 32 | 64 | `0.00014683400028540441` | `0.0001991246669300987` | `1.3561209702320702` | `0.00009355450038128765` | `0.6371446681248468` | `0.00008196550015782123` | `0.5578793941560658` |
| candidate256 | 256 | 16 | 8 | 256 | 512 | `0.0004906753335186901` | `0.0003059276665832537` | `0.6234828728589447` | `0.00014487799990092753` | `0.29526244749657676` | `0.0002035272500506835` | `0.40975533401850467` |
| query_vectors16 | 256 | 16 | 16 | 128 | 256 | `0.000525613999949807` | `0.00027538366642450757` | `0.5239275712800745` | `0.0001481137496739393` | `0.2817918656810574` | `0.00016245325014097034` | `0.31761661665325586` |
| doc_vectors32 | 256 | 32 | 8 | 128 | 256 | `0.000421066000853898` | `0.0004228856675278318` | `1.004321571131945` | `0.00022217449986783322` | `0.5276476833020854` | `0.00021599299998342758` | `0.5138284194354665` |
| documents512 | 512 | 16 | 8 | 128 | 256 | `0.000297582000712282` | `0.00037199399957899004` | `1.250055442495171` | `0.00017294025019509718` | `0.5811515810134799` | `0.00016419124995081802` | `0.5467537677282048` |

Envelope ratios:

| case | CPU cand+GPU / CPU cand+score | CPU cand+same-window / CPU cand+score | CPU multi-cand+multi-window / CPU multi-cand+score |
| --- | ---: | ---: | ---: |
| baseline | `0.9594621966307288` | `0.7822498595669877` | `0.8179435036761529` |
| candidate32 | `1.102515642996822` | `0.8955457533495826` | `0.8774789103604915` |
| candidate256 | `0.8404487238038513` | `0.7013634499469998` | `0.7532521099858513` |
| query_vectors16 | `0.8184638963306596` | `0.7261326250745969` | `0.7462957374725315` |
| doc_vectors32 | `1.0019125416062113` | `0.7909571702297928` | `0.7880391046652836` |
| documents512 | `1.0730426440337564` | `0.8776519492363184` | `0.8707099671955416` |

Summary:

- ok cases: `6 / 6`
- best isolated GPU ratio: `0.5239275712800745`
- worst isolated GPU ratio: `1.3561209702320702`
- best same-window resident GPU ratio: `0.2817918656810574`
- worst same-window resident GPU ratio: `0.6371446681248468`
- best multi-window resident GPU ratio: `0.31761661665325586`
- worst multi-window resident GPU ratio: `0.5578793941560658`
- best CPU-candidate-plus-GPU ratio: `0.8184638963306596`
- worst CPU-candidate-plus-GPU ratio: `1.102515642996822`
- best CPU-candidate-plus-same-window ratio: `0.7013634499469998`
- worst CPU-candidate-plus-same-window ratio: `0.8955457533495826`
- best CPU-candidate-plus-multi-window ratio: `0.7462957374725315`
- worst CPU-candidate-plus-multi-window ratio: `0.8774789103604915`
- maximum observed score delta: `9.1552734375e-05`

## Interpretation

Verified:

- the one-shot address serving call, same-window resident call, and
  multi-window resident call preserved CPU i8 score agreement across all swept
  shapes
- isolated one-shot GPU scoring wins when the candidate-score work is large
  enough, such as `candidate256` and `query_vectors16`
- small candidate windows are too small for the current one-shot serving-call
  overhead
- larger prepared indexes move the current one-shot call toward break-even or
  loss because it still allocates and copies token codes, scales, and offsets
  inside every serving call
- same-window resident iteration is faster than CPU same-candidate scoring in
  every swept case
- multi-window resident scoring is also faster than CPU same-candidate scoring
  per window in every swept case
- CPU candidate generation plus multi-window resident scoring is faster than
  CPU candidate generation plus CPU same-candidate scoring in every swept case

Debunked:

- the serving-shaped call is not uniformly faster than CPU i8 scoring
- kernel math is not the only or obvious next optimization target
- one-shot typed-address ingestion alone is not enough for the smaller or
  larger-index cases
- the resident-session result is not merely a repeated same-window artifact;
  the different-window row remains favorable, although it is more conservative
  than the same-window row on some shapes

Still open:

- the result after removing repeated prepared-index copies across real Python
  calls
- the result after reusing device allocations across calls with a safe ownership
  model
- the cost of returning all scores versus doing GPU or Mojo-side top-k
- the full candidate-generation plus score plus top-k envelope

## Decision

Design and test a real internal prepared-index ownership boundary before
editing kernel math.

Reason: the shape sweep points to fixed serving overhead and repeated
prepared-index movement. Optimizing the two-pass dim128 kernel first would not
address the losing one-shot `candidate32`, `doc_vectors32`, or `documents512`
cases, while both resident rows show that amortizing prepared-index copy and
allocation can make all swept cases favorable.

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
pixi run profile_gpu_i8_address_serve_sweep_raw
pixi run profile_gpu_i8_address_serve_sweep
```

Observed:

- Mojo format completed
- Python compile check passed
- GPU i8 rerank contract tests: `29/29` passed
- raw address serving sweep status: `ok`
- quiet address serving sweep status: `ok`
- quiet wrapper emitted `30` sections, including resident multi-window timing
  for all six cases
