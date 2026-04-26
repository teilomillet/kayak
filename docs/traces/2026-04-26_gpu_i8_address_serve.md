# 2026-04-26: GPU I8 Typed-Address Serving Call Check

## Claim

The prepared address bridge's remaining `~0.67s` extension-call time may be an
artifact of the internal Mojo `benchmark.run` sections rather than the cost of
one serving-shaped Python-to-Mojo call.

Reason: the previous address bridge already reduced Python host marshalling to
microseconds and preserved GPU score agreement, but its profiled function still
ran internal timing harnesses.

## Added

- `score_i8_prepared_payload_session_addresses(...)` in
  `python/kayak_bridge/_mojo_gpu_i8_rerank_bindings.mojo`
- `MojoGpuI8AddressServeResult` and
  `score_i8_prepared_payload_session_addresses(...)` in
  `python/kayak_bridge/mojo_gpu_i8_rerank.py`
- `gpu_real_payload_prepared_address_serve` in
  `python/scripts/profile_gpu_i8_real_payload_rerank.py`
- report status and quiet-wrapper output for the address serving row

Reason: this isolates one no-internal-benchmark extension call while keeping
the same real Kayak i8 payload, candidate window, and CPU i8 correctness
reference.

## Boundary

Inputs:

- `Float32` query values shaped `[query_count, query_vector_count, 128]`
- `Int8` token codes shaped `[total_doc_vectors, 128]`
- `Float32` token scales shaped `[total_doc_vectors]`
- `Int64` document offsets shaped `[document_count + 1]`
- `Int64` candidate positions shaped `[query_count, candidate_k]`
- `Float32` CPU i8 reference scores shaped `[query_count, candidate_k]`

The call passes raw contiguous NumPy data addresses into Mojo, reconstructs
typed pointers, fills Mojo host buffers, allocates device buffers, copies the
prepared tensors to device, runs the two-pass candidate-score kernels, reads
scores back, checks CPU i8 score agreement, and returns the scores.

It does not:

- reuse a GPU-resident prepared index across calls
- expose a public GPU backend
- move candidate generation or top-k to GPU
- measure H2D, kernel, or D2H internally

Reason: this row measures the serving-shaped extension-call envelope, not an
optimized resident-index backend.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_real_payload_rerank
```

Artifacts:

- report: `.cache/kayak/gpu_i8_real_payload_rerank/summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260426T154601Z`

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

- CPU i8 same-candidate score mean: `0.000295604666765333`
- prepared address bridge extension call mean with internal benchmarks:
  `0.6739463300000352`
- prepared address bridge score H2D + kernel + D2H mean:
  `3.884576746251756e-05`

Prepared address serving row:

- status: `ok`
- score delta max abs: `4.57763671875e-05`
- host marshalling mean: `1.0198999613445872e-05`
- extension call mean: `0.00026204533302613225`
- extension call / CPU i8 same-candidate score mean:
  `0.8864722465094168`

## Interpretation

Verified:

- the no-internal-benchmark typed-address call preserves CPU i8 score agreement
  on this shape
- the serving-shaped extension call is about `0.262ms`
- the previous `~0.67s` prepared address profile was dominated by internal
  profiling scaffolding, not by the Python extension boundary alone
- on this smoke shape, the serving-shaped GPU call is slightly faster than the
  CPU i8 same-candidate scoring row

Still open:

- whether the result holds across larger candidate windows, query vector counts,
  and document vector counts
- whether the call remains useful when CPU candidate generation and CPU top-k
  are included in an end-to-end rerank path
- how to safely own and reuse `DeviceContext` plus device buffers across Python
  calls, given the earlier `Writable` constraint for Mojo Python types
- whether returning all scores to Python is the right boundary once top-k moves
  closer to the GPU path

## Decision

Continue with a shape sweep and a real prepared-index ownership design before
editing kernel math.

Reason: the measured kernel and copy timings are already small on this shape.
The next likely bottlenecks are ownership, allocation, repeated index copies,
and end-to-end rerank composition, not the dim128 dot-product math.

Follow-up: `docs/traces/2026-04-26_gpu_i8_address_serve_sweep.md` adds the
shape sweep. It kept score agreement across all six cases, but showed the
current one-shot serving call loses on small candidate windows and enlarged
document counts. The added resident-session rows copy the prepared index once
inside a single extension call and make every swept same-window and
different-window scoring row faster than CPU i8. That strengthens the decision
to solve prepared-index residency before kernel math, while leaving cross-call
ownership unproven.

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
- GPU i8 rerank contract tests: `25/25` passed
- quiet real-payload profile status: `ok`
