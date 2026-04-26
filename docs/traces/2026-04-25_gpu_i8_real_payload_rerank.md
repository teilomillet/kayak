# 2026-04-25: GPU I8 Real-Payload Rerank Profile

## Claim

The next GPU evidence step is to score real Kayak i8 payload buffers, not only
deterministic synthetic buffers.

Reason:

- the deterministic GPU probe proved kernel structure and indexing
- production integration needs the prepared i8 token codes, token scales,
  document offsets, query values, and candidate ids used by Kayak
- candidate generation and top-k can stay on CPU while the score primitive is
  validated

## Added

- `benchmarks/gpu_i8_real_payload_score_probe.mojo`
- `python/kayak_bridge/gpu_i8_real_payload_score.py`
- `python/kayak_bridge/_mojo_gpu_i8_rerank_bindings.mojo`
- `python/kayak_bridge/mojo_gpu_i8_rerank.py`
- `python/scripts/profile_gpu_i8_real_payload_rerank.py`
- `pyproject.toml` tasks:
  - `profile_gpu_i8_real_payload_rerank_raw`
  - `profile_gpu_i8_real_payload_rerank`
- `KayakPlaidApproxIndex.i8_payload_snapshot()`
- Mojo bridge exports for prepared i8 payload fields:
  - `plaid_i8_prepared_doc_offsets`
  - `plaid_i8_prepared_token_codes`
  - `plaid_i8_prepared_token_scales`

Reason:

- benchmark-only GPU probes need flat buffers without exposing a public GPU API
- the Python/Mojo GPU bridge needs a separate accelerator-targeted extension
  instead of being folded into the exact CPU extension
- the prepared i8 index currently owns token codes and scales inside Mojo
- a narrow snapshot method makes the boundary explicit and testable

## Boundary

The real-payload profile still is not a production GPU backend. It now has two
GPU rows:

- a standalone Mojo executable row that consumes file-staged buffers
- an in-process GPU-targeted Python/Mojo extension row that receives the same
  real payload through Python objects

It does:

- build a real Kayak i8 prepared index
- generate candidate document positions on CPU
- score those exact candidate positions on CPU as the reference
- export query values, token codes, token scales, doc offsets, candidates, and
  reference scores into `.cache`
- run the GPU score probe over those exported buffers
- run the GPU extension bridge over the same logical buffers without file
  staging
- compare GPU scores against CPU i8 scores

It does not:

- generate candidates on GPU
- keep index buffers resident across queries
- perform top-k on GPU
- expose public `gpu=True` search
- avoid full payload copy and Python-object decode in the extension bridge

Reason: this isolates the first real-payload correctness and timing question
before deciding the internal backend ownership model.

Bridge environment finding:

- starting Python with `PYTHONPATH=python` can make Mojo GPU context creation
  fail locally with NVML error `9`
- the GPU tasks now rely on each script's explicit `sys.path` setup instead of
  task-level `PYTHONPATH`

Reason: this is a verified local failure mode; leaving `PYTHONPATH=python` in
GPU benchmark tasks would make the repo report false GPU unavailability.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_real_payload_rerank
```

Artifacts:

- report: `.cache/kayak/gpu_i8_real_payload_rerank/summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260425T204818Z`
- exported input manifest:
  `.cache/kayak/gpu_i8_real_payload_rerank/input/manifest.json`

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

- exact CPU query batch mean: `0.08336409466695234`
- i8 candidate generation mean: `0.0005140923331055092`
- i8 same-candidate score mean: `0.0003028680002898909`
- i8 recall@10 versus exact: `0.5`
- i8 build seconds: `0.034528254998804186`
- payload export seconds: `0.12685209600022063`

GPU real-payload row:

- status: `ok`
- target accelerator: `nvidia:sm_89`
- payload source: `real_kayak_i8_snapshot`
- score delta max abs: `4.57763671875e-05`
- two-pass score delta max abs: `4.57763671875e-05`
- serial kernel mean: `0.0008182270699300699`
- two-pass kernel mean: `2.7571585074626866e-05`
- host-to-device mean: `3.3218578999999997e-05`
- device-to-host mean: `4.195892902230327e-06`
- host-to-device + kernel + device-to-host mean:
  `6.498605697685718e-05`
- two-pass kernel scores/sec: `9284921.389433919`
- two-pass speedup versus serial GPU kernel: `29.67646102737324x`

GPU extension bridge row:

- status: `ok`
- target accelerator: `nvidia:sm_89`
- score delta max abs: `4.57763671875e-05`
- repeated measurement iterations: `3`
- host marshalling mean: `0.0038326423327816883`
- extension call mean: `0.47131670599992503`
- two-pass kernel mean: `2.843207403180797e-05`
- host-to-device mean: `3.3075953e-05`
- device-to-host mean: `3.4894390333333338e-06`
- host-to-device + kernel + device-to-host mean:
  `6.49974660651413e-05`

Comparison:

- GPU kernel seconds / CPU same-candidate score second:
  `0.09103498899928897`
- GPU h2d+kernel+d2h seconds / CPU same-candidate score second:
  `0.21456891092705602`
- GPU bridge h2d+kernel+d2h seconds / CPU same-candidate score second:
  `0.21460658109449926`

Interpretation:

- The GPU score primitive now agrees with real Kayak i8 payloads for this
  regular dim128 shape.
- Copy+kernel+readback for the score primitive is lower than the CPU
  same-candidate score timing on this shape.
- The in-process extension proves the accelerator-targeted Python/Mojo bridge
  is feasible, but its current `0.47s` extension-call mean is dominated by
  Python-object decoding and allocation, not GPU execution.
- End-to-end search is still not accelerated because CPU candidate generation,
  payload export, full payload copying, Python-object decode, and CPU top-k are
  outside the GPU row.

## Decision

Move next to resident prepared-index GPU buffers or a lower-overhead host buffer
interface.

Reason:

- correctness against real i8 payloads is now measured
- standalone file staging is no longer the only measured boundary
- the remaining artificial cost is not the score math but Python-object decode,
  repeated full-payload copying, and lack of buffer ownership
- keeping candidate generation on CPU remains the right next boundary because
  scoring has isolated evidence and candidate generation does not

## Validation

Ran:

```bash
pixi run python -m py_compile \
  python/kayak_bridge/mojo_gpu_i8_rerank.py \
  python/kayak_bridge/gpu_i8_real_payload_score.py \
  python/kayak_bridge/plaid_approx.py \
  python/scripts/profile_gpu_i8_real_payload_rerank.py \
  python/tests/test_gpu_i8_rerank_contract.py \
  python/tests/test_fastplaid_speed_track.py
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_fastplaid_speed_track.FastPlaidSpeedTrackTests.\
test_kayak_plaid_i8_payload_snapshot_exposes_flat_buffers -v
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_gpu_i8_rerank_contract.GpuI8RerankContractTests.\
test_real_payload_probe_input_manifest_records_binary_counts -v
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_gpu_i8_rerank_contract -v
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_fastplaid_speed_track -v
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_sdist_manifest -v
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_repo_semantic_inventory -v
pixi run profile_gpu_i8_real_payload_rerank_raw
pixi run profile_gpu_i8_real_payload_rerank
pixi run bench_gpu_i8_rerank_contract_raw
pixi run mojo format benchmarks/gpu_i8_real_payload_score_probe.mojo
pixi run mojo format python/kayak_bridge/_mojo_gpu_i8_rerank_bindings.mojo
pixi run mojo run --target-accelerator nvidia:sm_89 \
  -D data_root=.cache/kayak/gpu_i8_real_payload_rerank/input \
  -D query_count=2 \
  -D query_vector_count=8 \
  -D document_count=256 \
  -D document_vector_count=16 \
  -D candidate_k=128 \
  benchmarks/gpu_i8_real_payload_score_probe.mojo
git diff --check
```

Observed:

- Python compile check passed
- i8 payload snapshot test passed
- real-payload input manifest test passed
- GPU i8 rerank contract tests: `23/23` passed
- FastPlaid speed-track tests: `8/8` passed
- sdist manifest tests: `3/3` passed
- repo semantic inventory tests: `3/3` passed
- raw real-payload GPU profile status: `ok`
- quiet real-payload GPU profile status: `ok`
- raw GPU i8 rerank contract task saw the GPU and reported copy, single-score,
  and candidate-score probe status `ok`; production backend integration still
  correctly reports missing
- standalone real-payload Mojo probe after formatting reported status `ok`

## Follow-Up

The next step was completed in
`docs/traces/2026-04-26_gpu_i8_prepared_bridge.md`.

Result: a single-call prepared-session bridge keeps real Kayak i8 index payload
buffers resident on device for the profiled score path. On the same smoke
shape, score H2D + two-pass kernel + D2H moved from about `6.47e-05s` in the
full-copy bridge to about `3.88e-05s` with resident index buffers, while
preserving `score_delta_max_abs=4.57763671875e-05`.

Reason: this validates the resident-buffer hypothesis without claiming a public
GPU backend or settling Python-owned device-buffer lifetime.
- both real-payload executable runs reported `twopass_score_agreement_ok=True`
- the real-payload extension bridge reported status `ok`
- `git diff --check` passed

Open validation still needed:

- repeat real-payload profile across larger candidate windows
- support irregular document vector counts or explicitly reject them at the GPU
  boundary
- remove repeated full-payload copy by adding an internal prepared-index GPU
  buffer owner
- avoid element-wise Python-object decode at the GPU bridge
- measure CPU top-k cost after GPU score readback
