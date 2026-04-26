# 2026-04-26: GPU I8 Direct-Array Bridge Check

## Claim

The prepared-session bridge may be slow because Python converts NumPy payloads
to Python lists before calling Mojo.

Reason: the previous prepared-session profile measured about `3ms` of Python
host marshalling before the extension call, and list materialization is an
avoidable copy.

## Added

- `profile_i8_prepared_payload_session_ndarray(...)` in
  `python/kayak_bridge/mojo_gpu_i8_rerank.py`
- `gpu_real_payload_prepared_ndarray_bridge` in
  `python/scripts/profile_gpu_i8_real_payload_rerank.py`
- report status handling for the ndarray bridge row

Reason: this keeps the kernel and Mojo function unchanged while changing only
the Python host-ingestion surface. The comparison can therefore isolate list
materialization from Mojo-side Python object indexing.

## Boundary

The ndarray bridge passes contiguous 1-D NumPy arrays directly into the same
Mojo function used by the prepared-session list bridge:

- `Float32` query values
- `Int8` token codes
- `Float32` token scales
- `Int64` document offsets
- `Int64` candidate positions
- `Float32` CPU i8 reference scores

It still does not:

- use a native buffer pointer API
- keep buffers resident across Python calls
- expose a public GPU backend
- avoid Mojo-side Python object indexing and scalar conversion

Reason: the goal is to validate or debunk list materialization as the dominant
remaining overhead before adding a lower-level binding.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_real_payload_rerank
```

Artifacts:

- report: `.cache/kayak/gpu_i8_real_payload_rerank/summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260426T145049Z`

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

- CPU i8 same-candidate score mean: `0.0002992046665895032`
- full-copy GPU bridge H2D + kernel + D2H mean: `6.47636384231148e-05`
- prepared list bridge score H2D + kernel + D2H mean:
  `3.8825600315145584e-05`

Prepared list bridge:

- status: `ok`
- score delta max abs: `4.57763671875e-05`
- host marshalling mean: `0.003216069999780302`
- extension call mean: `0.7035108469999614`
- prepare H2D mean: `2.644595013239188e-05`
- score H2D mean: `7.039843494204702e-06`
- two-pass kernel mean: `2.8334019654274214e-05`
- D2H mean: `3.451737166666667e-06`

Prepared ndarray bridge:

- status: `ok`
- score delta max abs: `4.57763671875e-05`
- host marshalling mean: `2.4836000193317886e-05`
- extension call mean: `0.7178832189997593`
- prepare H2D mean: `2.6482221217468018e-05`
- score H2D mean: `6.769262573964497e-06`
- two-pass kernel mean: `2.831052016985138e-05`
- D2H mean: `3.4602787666666666e-06`
- score H2D + kernel + D2H mean: `3.8540061510482546e-05`

## Interpretation

Verified:

- contiguous NumPy arrays are accepted by the existing Mojo PythonObject
  indexing path for this shape
- score agreement is unchanged versus CPU i8
- Python-side host marshalling drops from about `3.22ms` to about `24.8us`

Debunked:

- Python `.tolist()` materialization is not the dominant extension-call
  bottleneck. The ndarray extension call remains about `0.72s`.

Still open:

- whether Mojo can consume a native Python buffer pointer without per-element
  Python object indexing
- whether a Python-owned prepared GPU object can safely own `DeviceContext`
  and device buffers
- whether larger candidate windows change the relative importance of kernel
  time versus bridge overhead

## Decision

Do not optimize the kernel next.

Reason: both prepared rows still spend about `2.83e-05s` in the two-pass
kernel, while the extension call stays hundreds of milliseconds. The next
useful experiment is a typed buffer protocol or capsule-style host ingestion
path that avoids `py_values[index]` for every scalar.

## Validation

Ran:

```bash
pixi run python -m py_compile \
  python/kayak_bridge/mojo_gpu_i8_rerank.py \
  python/scripts/profile_gpu_i8_real_payload_rerank.py \
  python/tests/test_gpu_i8_rerank_contract.py
pixi run profile_gpu_i8_real_payload_rerank_raw
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_gpu_i8_rerank_contract -v
pixi run profile_gpu_i8_real_payload_rerank
```

Observed:

- Python compile check passed
- raw real-payload profile status: `ok`
- GPU i8 rerank contract tests: `24/24` passed
- quiet real-payload profile status: `ok`
