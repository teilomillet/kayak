# 2026-04-26: GPU I8 Prepared-Session Bridge

## Claim

The current full-copy GPU bridge spends per-score transfer time copying the
whole i8 payload. Keeping token codes, token scales, and document offsets
resident on device should reduce the score path to query values, candidate
positions, kernel work, and score readback.

Reason: the previous real-payload bridge measured matching standalone GPU
copy+kernel+readback timing, but still copied all index payload buffers on
every bridge call.

## Design Decision

Add a single-call prepared profiling session instead of a Python-visible
prepared GPU index object.

Reason:

- the repo has no existing Python-owned Mojo GPU object ownership pattern
- a first attempt to expose a `PreparedGpuI8RerankDim128` type hit Mojo Python
  binding constraints: `module.add_type[...]` needs `Writable`, while
  `DeviceContext` cannot derive `Writable`
- forcing a pointer wrapper before understanding lifetime semantics would hide
  the real ownership problem
- a single extension call can still test the key hypothesis: whether resident
  index buffers reduce the measured score path

## Added

- `profile_i8_prepared_payload_session(...)` in
  `python/kayak_bridge/_mojo_gpu_i8_rerank_bindings.mojo`
- `MojoGpuI8PreparedSessionResult` and
  `profile_i8_prepared_payload_session(...)` in
  `python/kayak_bridge/mojo_gpu_i8_rerank.py`
- a `gpu_real_payload_prepared_bridge` row in
  `python/scripts/profile_gpu_i8_real_payload_rerank.py`
- contract tests for prepared bridge status handling and result serialization

Reason: this keeps the backend boundary internal and measurable while avoiding
public API or lifetime commitments.

## Boundary

The prepared-session bridge receives the same real Kayak i8 payload snapshot as
the full-copy bridge:

- query values: `[query_count, query_vector_count, 128]`
- token codes: `[document_count * document_vector_count, 128]`
- token scales: `[document_count * document_vector_count]`
- document offsets: `[document_count + 1]`
- candidate positions: `[query_count, candidate_k]`
- CPU i8 reference scores: `[query_count, candidate_k]`

Inside one extension call it:

1. decodes Python objects into Mojo host buffers
2. copies token codes, token scales, and document offsets to device once
3. warms up the score path
4. measures query/candidate H2D, two-pass kernel, and score D2H with the index
   buffers already resident
5. compares GPU scores with CPU i8 scores

It does not:

- keep buffers resident across Python calls
- expose a public prepared GPU object
- generate candidates on GPU
- run top-k on GPU
- remove Python-object decode overhead

Reason: this isolates the resident-index hypothesis from Python lifetime and
search-planner questions.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_real_payload_rerank
```

Artifacts:

- report: `.cache/kayak/gpu_i8_real_payload_rerank/summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260426T143444Z`

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

CPU rows:

- i8 same-candidate score mean: `0.00029761499998433766`
- i8 candidate generation mean: `0.0005175790000369792`
- exact CPU query batch mean: `0.08239035433340784`
- recall@10 versus exact for the i8 candidate window: `0.5`

Full-copy in-process bridge:

- status: `ok`
- score delta max abs: `4.57763671875e-05`
- host marshalling mean: `0.003656593333289493`
- extension call mean: `0.4700268750000305`
- full payload H2D mean: `3.2991594e-05`
- two-pass kernel mean: `2.830375452121546e-05`
- D2H mean: `3.4393150555555553e-06`
- H2D + kernel + D2H mean: `6.473466357677101e-05`

Prepared-session bridge:

- status: `ok`
- score delta max abs: `4.57763671875e-05`
- host marshalling mean: `0.003131667000161542`
- extension call seconds: `0.7042507779999596`
- resident index prepare H2D mean: `2.6472296933598057e-05`
- query/candidate score H2D mean: `7.038934658722964e-06`
- two-pass kernel mean: `2.8310303659976387e-05`
- D2H mean: `3.4516794666666664e-06`
- score H2D + kernel + D2H mean: `3.880091778536602e-05`

## Interpretation

Verified:

- resident index buffers preserve CPU i8 score agreement on this shape
- removing index payload copies from the score path reduces measured
  H2D + kernel + D2H from `6.47e-05s` to `3.88e-05s`
- the kernel time is unchanged, as expected, because this change only moves
  transfer ownership

Not verified:

- production speedup
- reusable prepared-object lifetime
- lower Python bridge overhead
- behavior on larger vector-count and candidate-window shapes

The prepared session's extension call is slower than the full-copy bridge
because it performs prepare timing, warmup, score timing, and final correctness
work inside one call. That number is useful as a profiling boundary, not as a
production serving estimate.

## Decision

Continue toward a real internal prepared-index backend boundary, but only after
separating two problems:

- device residency: now validated for one real-payload shape
- Python/Mojo object lifetime and low-overhead host buffer ingestion: still open

Reason: the measurement supports resident device buffers, while the `Writable`
constraint shows that the Python-visible ownership model is not settled.

## Next Check

Profile a lower-overhead host input boundary before optimizing the kernel.

Reason: the kernel is about `2.83e-05s`, while the current extension path still
decodes Python objects and allocates host/device buffers inside the call. Kernel
tuning would be premature until bridge overhead and buffer lifetime are less
dominant.

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

## Follow-Up

The next host-ingestion check was completed in
`docs/traces/2026-04-26_gpu_i8_ndarray_bridge.md`.

Result: passing contiguous NumPy arrays directly into the same prepared-session
Mojo function reduced Python host marshalling from about `3.22ms` to about
`24.8us`, while preserving `score_delta_max_abs=4.57763671875e-05`. It did not
reduce extension-call time, which stayed around `0.72s`.

Reason: list materialization is measurable but not the dominant bridge
bottleneck. The next bridge experiment should avoid per-element Python object
indexing inside Mojo.
