# 2026-04-26: GPU I8 Explicit Address Handle

## Claim

An internal `prepare -> score -> release` GPU handle can test cross-call
prepared-index residency without exposing a public GPU backend and without a
hidden global cache.

Reason: the resident in-call probes showed that reusing device buffers is high
leverage, but they did not prove that the same ownership model can survive
separate Python calls.

## Ownership Evidence

Verified locally:

- a Python-visible Mojo object that owns `DeviceContext` is still blocked by
  the Python type registration path; the local `module.add_type[...]` probe
  failed because `DeviceContext` cannot derive `Writable`
- returning `PythonObject(alloc=GpuContextOwner(...))` without registering the
  type compiled, but failed at runtime because no Python type object was
  registered
- a heap-owned Mojo struct can own `DeviceContext`, `DeviceBuffer`, and
  `HostBuffer` fields, return its `UnsafePointer.__int__()` address to Python,
  be used in a later Python call, and be released with explicit
  `destroy_pointee()` plus `free()`

Official Modular docs are consistent with that local evidence:

- `PythonObject(alloc=...)` requires a registered Python type object, and type
  registration enables `PythonObject(alloc=...)` plus
  `downcast_value_ptr[...]()`
- `UnsafePointer` memory is manual lifetime memory; `alloc[T](count)` must be
  paired with `free()`
- `UnsafePointer.__int__()` returns the pointer address as an integer

Sources:

- <https://docs.modular.com/mojo/std/python/python_object/PythonObject/>
- <https://docs.modular.com/mojo/manual/python/mojo-from-python/>
- <https://docs.modular.com/mojo/std/memory/unsafe_pointer/UnsafePointer/>
- <https://docs.modular.com/mojo/std/memory/unsafe_pointer/alloc>

Reason: the chosen handle is explicit unsafe ownership, not a Python-visible
Mojo value and not a module-level cache. That keeps the lifetime problem visible
in the API shape.

## Added

- `PreparedGpuI8AddressSession` inside
  `python/kayak_bridge/_mojo_gpu_i8_rerank_bindings.mojo`
- `prepare_i8_address_session_handle(...)`
- `score_i8_address_session_handle(...)`
- `release_i8_address_session_handle(...)`
- `MojoGpuI8AddressSessionHandle` in
  `python/kayak_bridge/mojo_gpu_i8_rerank.py`
- a prepared-handle row in `profile_gpu_i8_address_serve_sweep`

The handle owns:

- `DeviceContext`
- resident token codes: `[total_doc_vectors, 128]`
- resident token scales: `[total_doc_vectors]`
- resident document offsets: `[document_count + 1]`
- reusable query, candidate, partial-score, and score buffers sized for one
  fixed query/candidate window shape

It does not:

- expose public `search(..., gpu=True)`
- generate candidates on GPU
- perform top-k on GPU
- hide a global cache
- protect against arbitrary forged integer handles at the Mojo boundary

Reason: this is still an internal primitive. The Python wrapper makes `close()`
idempotent, but the Mojo pointer address is intentionally not public API.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_address_serve_sweep
```

Artifacts:

- report: `.cache/kayak/gpu_i8_address_serve_sweep/summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260426T170855Z`

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

| case | docs | doc vecs | query vecs | candidate_k | prepared handle score/window s | prepared/CPU score | CPU cand+prepared/CPU cand+score | prepare s | max delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 256 | 16 | 8 | 128 | `0.00010364099944126792` | `0.35092087146580947` | `0.7693998184485685` | `0.00023694199990131892` | `0.00006103515625` |
| candidate32 | 256 | 16 | 8 | 32 | `0.000048795999646245036` | `0.33459425718026564` | `0.8147401115617653` | `0.00023716199939372018` | `0.0000457763671875` |
| candidate256 | 256 | 16 | 8 | 256 | `0.00016373325024687801` | `0.3293083221115956` | `0.7197112281147191` | `0.0002375229996687267` | `0.0000762939453125` |
| query_vectors16 | 256 | 16 | 16 | 128 | `0.0001246552496922959` | `0.24400748478778` | `0.7192502774635106` | `0.0002370729998801835` | `0.000091552734375` |
| doc_vectors32 | 256 | 32 | 8 | 128 | `0.00014970674965297803` | `0.3610252937041942` | `0.7228157684618225` | `0.00045214400051918346` | `0.00006103515625` |
| documents512 | 512 | 16 | 8 | 128 | `0.00009989149975808687` | `0.33530295294556844` | `0.8101282125737786` | `0.00045230400064610876` | `0.00006103515625` |

Summary:

- ok cases: `6 / 6`
- best prepared-handle isolated score ratio: `0.24400748478778`
- worst prepared-handle isolated score ratio: `0.3610252937041942`
- best CPU-candidate-plus-prepared-handle ratio: `0.7192502774635106`
- worst CPU-candidate-plus-prepared-handle ratio: `0.8147401115617653`
- quiet wrapper sections: `36`

## Interpretation

Verified:

- explicit cross-call GPU index residency preserves CPU i8 score agreement on
  every swept vector-count shape
- the prepared-handle score row is faster than CPU same-candidate scoring in
  every swept case
- CPU candidate generation plus prepared-handle GPU scoring is faster than CPU
  candidate generation plus CPU same-candidate scoring in every swept case
- the handle row is generally faster than the in-call multi-window row,
  despite crossing the Python extension boundary for each window

Still open:

- full candidate generation plus score plus top-k integration
- whether returning every candidate score is the next bottleneck
- whether a GPU-side or Mojo-side top-k selector improves the envelope
- production lifetime protection beyond the internal explicit handle

## Decision

Continue with the explicit internal handle as the GPU rerank ownership boundary.

Reason: this is the first measured path that reuses GPU-resident prepared index
buffers across Python calls, keeps vector counts explicit, avoids silent CPU
fallback, and wins on every swept case. Kernel rewrites should wait until the
score-return and top-k boundary is measured.

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
  --output .cache/kayak/gpu_i8_address_serve_sweep/handle_smoke.json
pixi run profile_gpu_i8_address_serve_sweep
```

Observed:

- Mojo format completed
- Python compile check passed
- GPU i8 rerank contract tests: `30/30` passed
- explicit-handle smoke status: `ok`
- quiet full sweep status: `ok`
- quiet wrapper emitted `36` sections
