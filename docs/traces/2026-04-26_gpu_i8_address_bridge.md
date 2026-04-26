# 2026-04-26: GPU I8 Typed-Address Bridge Check

## Claim

The remaining prepared bridge overhead may come from Mojo-side
`PythonObject[index]` scalar access. Passing raw typed NumPy data addresses into
Mojo should avoid that per-element Python object indexing.

Reason: the direct ndarray row reduced Python-side marshalling, but the
extension call still stayed around `0.72s`.

## Added

- `profile_i8_prepared_payload_session_addresses(...)` in
  `python/kayak_bridge/_mojo_gpu_i8_rerank_bindings.mojo`
- `profile_i8_prepared_payload_session_addresses(...)` in
  `python/kayak_bridge/mojo_gpu_i8_rerank.py`
- `gpu_real_payload_prepared_address_bridge` in
  `python/scripts/profile_gpu_i8_real_payload_rerank.py`
- report status handling for the address bridge row

Reason: this tests typed host memory ingestion without changing the GPU scoring
kernel or exposing a public GPU backend.

## Boundary

The address bridge keeps the same real-payload tensors and passes only integer
addresses for contiguous NumPy arrays:

- `Float32` query values
- `Int8` token codes
- `Float32` token scales
- `Int64` document offsets
- `Int64` candidate positions
- `Float32` CPU i8 reference scores

Mojo reconstructs typed pointers with
`UnsafePointer[T, MutAnyOrigin](unsafe_from_address=address)`, copies those
typed pointers into Mojo host buffers, and then runs the same prepared-session
GPU copy, kernel, and readback boundaries.

It still does not:

- own those NumPy buffers beyond the Python call
- use Python's buffer protocol directly
- expose a reusable prepared GPU index
- remove the internal `benchmark.run` harness from the profiled function

Reason: integer addresses are unsafe as a public API, but they are useful as an
internal probe for the specific per-element indexing hypothesis.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_real_payload_rerank
```

Artifacts:

- report: `.cache/kayak/gpu_i8_real_payload_rerank/summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260426T151428Z`

Shape:

- documents: `256`
- document vectors per document: `16`
- total document vectors: `4096`
- queries: `2`
- query vectors per query: `8`
- total query vectors: `16`
- vector dim: `128`
- candidate window: `128`
- candidate scores: `256`

Reference rows:

- CPU i8 same-candidate score mean: `0.0002942856666171186`
- full-copy GPU bridge H2D + kernel + D2H mean: `6.484338735374983e-05`
- prepared list bridge score H2D + kernel + D2H mean:
  `3.859009812275832e-05`
- prepared ndarray bridge score H2D + kernel + D2H mean:
  `3.700017993287216e-05`

Prepared address bridge:

- status: `ok`
- score delta max abs: `4.57763671875e-05`
- host marshalling mean: `1.19420001283288e-05`
- extension call mean: `0.6744782670002678`
- prepare H2D mean: `2.6472184547461368e-05`
- score H2D mean: `5.431507214206438e-06`
- two-pass kernel mean: `2.817494505752524e-05`
- D2H mean: `3.4580692e-06`
- score H2D + kernel + D2H mean: `3.706452147173168e-05`

## Interpretation

Verified:

- typed address ingestion preserves CPU i8 score agreement on this shape
- Python-side marshalling is down to about `12us`
- score H2D + kernel + D2H stays in the same range as the prepared ndarray row
- avoiding Mojo-side `PythonObject[index]` is a small improvement in the
  benchmarked extension call, from about `0.719s` to about `0.674s`

Debunked:

- per-element Python object indexing is not the only reason the prepared
  profiled extension call is hundreds of milliseconds.

Still open:

- how much of the remaining extension-call time is the internal `benchmark.run`
  harness
- how fast a serving-style single call is when it does not run internal
  microbenchmarks
- what the safe ownership wrapper should be for typed host buffers and future
  resident device buffers

## Decision

Add a serving-style no-internal-benchmark address call before designing a
persistent GPU object.

Reason: the current prepared rows are profiler functions. They intentionally run
multiple internal `benchmark.run` sections, so their extension-call timings are
not serving-call timings. The next check should measure one address-ingested
prepare + score + readback call with no internal benchmark harness.

Follow-up: `docs/traces/2026-04-26_gpu_i8_address_serve.md` adds that check.
It measured `0.00026204533302613225s` extension-call time with the same
`score_delta_max_abs=4.57763671875e-05`, confirming that the `~0.67s` address
profile row was dominated by internal profiling scaffolding on this shape.

## Validation

Ran:

```bash
pixi run mojo format python/kayak_bridge/_mojo_gpu_i8_rerank_bindings.mojo
pixi run python -m py_compile \
  python/kayak_bridge/mojo_gpu_i8_rerank.py \
  python/scripts/profile_gpu_i8_real_payload_rerank.py \
  python/tests/test_gpu_i8_rerank_contract.py
pixi run python -c "import sys; sys.path.insert(0, 'python'); from kayak_bridge.mojo_gpu_i8_rerank import gpu_extension_device_probe; print(gpu_extension_device_probe(target_accelerator='nvidia:sm_89'))"
pixi run profile_gpu_i8_real_payload_rerank_raw
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_gpu_i8_rerank_contract -v
pixi run profile_gpu_i8_real_payload_rerank
```

Observed:

- Mojo format completed
- Python compile check passed
- GPU extension device probe returned `ok`
- raw real-payload profile status: `ok`
- GPU i8 rerank contract tests: `24/24` passed
- quiet real-payload profile status: `ok`
