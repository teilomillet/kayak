# 2026-04-25: GPU I8 Rerank Plan

## Claim

GPU work should start as an internal dim128 i8 candidate-window rerank
primitive, not as full candidate generation and not as a public search backend.

Reason:

- the CPU i8 lane is already explicit enough to serve as the implementation
  reference
- same-candidate rerank makes score correctness measurable
- copy, kernel, and readback time can be reported separately
- a public GPU API would imply stability the repo has not measured

## Evidence Checked

Code and docs inspected:

- `kayak/search/plaid_i8_approx_dim128.mojo`
- `kayak/search/plaid_approx_dim128.mojo`
- `kayak/numeric/scalars.mojo`
- `kayak/runtime/exact_backend.mojo`
- `python/kayak_bridge/plaid_approx.py`
- `python/scripts/bench_fastplaid_cpu_pareto.py`
- `docs/traces/2026-04-25_i8_plaid_cpu_matrix.md`
- `docs/product_direction.md`

Repo search:

```bash
rg -n "\b(gpu|GPU|cuda|CUDA|metal|Metal|accelerator|Accelerator)\b" \
  kayak benchmarks tests python/kayak_bridge python/kayak_engine \
  -g '*.mojo' -g '*.py'
```

Observed:

- no production Mojo GPU runtime pattern exists in the repo today
- `kayak/runtime/exact_backend.mojo` only leaves a future-GPU comment on the
  exact backend trait
- Python encoder/reference utilities mention CUDA through Torch, but those are
  not a Mojo GPU runtime boundary

Local toolchain checks:

```bash
pixi run mojo --version
pixi run gpu-query
```

Observed:

- Mojo version: `0.26.3.0.dev2026041405`
- PCI/procfs checks found an NVIDIA GeForce RTX 4070 Ti at `0000:01:00.0`
- with host device access, `nvidia-smi` works and `/dev/nvidia0`,
  `/dev/nvidiactl`, `/dev/nvidia-uvm`, and `/dev/dri/renderD*` are visible
- with host device access, `gpu-query` sees the RTX 4070 Ti through CUDA:
  compute capability `8.9`, target accelerator `nvidia:sm_89`, and reported
  memory about `10.15 GB`

Those device results are local-environment facts only.

## Decision

Added the architecture contract:

- `docs/gpu_architecture.md`

Added a benchmark scaffold:

- `python/kayak_bridge/gpu_i8_rerank_contract.py`
- `python/kayak_bridge/gpu_device_capability.py`
- `python/kayak_bridge/gpu_copy_roundtrip.py`
- `python/kayak_bridge/gpu_probe_output.py`
- `python/kayak_bridge/gpu_i8_profile_contract.py`
- `python/kayak_bridge/gpu_i8_candidate_score.py`
- `python/kayak_bridge/gpu_i8_real_payload_score.py`
- `python/kayak_bridge/gpu_i8_single_score.py`
- `python/kayak_bridge/_mojo_gpu_i8_rerank_bindings.mojo`
- `python/kayak_bridge/mojo_gpu_i8_rerank.py`
- `python/scripts/bench_gpu_i8_rerank_contract.py`
- `python/scripts/compare_gpu_i8_fastplaid.py`
- `python/scripts/gpu_i8_cpu_reference.py`
- `python/scripts/profile_gpu_i8_real_payload_rerank.py`
- `python/scripts/profile_gpu_i8_candidate_score.py`
- `benchmarks/gpu_i8_copy_roundtrip_probe.mojo`
- `benchmarks/gpu_i8_candidate_score_probe.mojo`
- `benchmarks/gpu_i8_real_payload_score_probe.mojo`
- `benchmarks/gpu_i8_single_doc_score_probe.mojo`
- `pyproject.toml` tasks:
  - `bench_gpu_i8_rerank_contract_raw`
  - `bench_gpu_i8_rerank_contract`
  - `profile_gpu_i8_candidate_score_raw`
  - `profile_gpu_i8_candidate_score`
  - `compare_gpu_i8_fastplaid_raw`
  - `compare_gpu_i8_fastplaid`
  - `compare_gpu_i8_fastplaid_cuda_raw`
  - `compare_gpu_i8_fastplaid_cuda`
  - `profile_gpu_i8_real_payload_rerank_raw`
  - `profile_gpu_i8_real_payload_rerank`

Added the same-candidate CPU i8 reference boundary:

- `kayak/search/plaid_i8_approx_dim128.mojo`
- `kayak/search/__init__.mojo`
- `python/kayak_bridge/_mojo_exact_cpu_bindings.mojo`
- `python/kayak_bridge/plaid_approx.py`

Reason:

- GPU score correctness needs CPU i8 scores for the same candidate ids
- candidate generation must stay separate from rerank scoring while the GPU
  primitive is being validated
- real-payload GPU probes need a narrow benchmark-only snapshot of prepared i8
  token codes, token scales, and document offsets

Added focused tests for the scaffold contract:

- `python/tests/test_gpu_i8_rerank_contract.py`
- `python/tests/test_fastplaid_speed_track.py`

Added benchmark-only GPU score probes, but did not add
`kayak/search/plaid_i8_gpu_dim128.mojo`.

Reason:

- `benchmarks/gpu_i8_single_doc_score_probe.mojo` verifies launch semantics,
  typed device buffers, score math, and readback before the production module
  exists
- `benchmarks/gpu_i8_candidate_score_probe.mojo` verifies `global_idx`
  indexing, doc offsets, candidate positions, and one score per GPU thread on
  deterministic flat tensors. It also compares that serial GPU strategy with a
  two-pass token-parallel strategy.
- `benchmarks/gpu_i8_real_payload_score_probe.mojo` verifies the same scoring
  path on exported Kayak i8 payload snapshots and CPU-generated candidate
  windows.
- `_mojo_gpu_i8_rerank_bindings.mojo` verifies a separate accelerator-targeted
  Python/Mojo extension boundary for the same real-payload score primitive.
- a production Mojo GPU module would still force ownership, bridge data
  transfer, and backend-boundary choices before those choices have been
  measured
- the Python scaffold can already enforce explicit vector-count reporting, no
  silent GPU fallback, and score-agreement gating

## Scaffold Contract

The scaffold records:

- host GPU inventory
- Mojo GPU capability probe output
- Mojo `DeviceBuffer` copy roundtrip status and copy timings for `Float32`,
  `Int8`, and `Int64`
- one-document GPU i8 score status, score delta, and timing fields
- benchmark-only batched candidate GPU score status, max score delta, and
  timing fields
- benchmark-only real-payload GPU score status, max score delta, exported input
  byte counts, and copy/kernel/readback timing fields
- in-process real-payload GPU extension status, host marshalling time,
  extension-call time, max score delta, and copy/kernel/readback timing fields
- candidate-score profile rows that derive candidate-score rates, serial versus
  two-pass ratios, scalar CPU reference ratios, and copy-to-kernel ratios
- explicit shape and vector counts
- dim128 tensor boundary shapes and dtypes
- CPU exact, CPU i8 full-search, and CPU i8 same-candidate reference rows
  unless skipped
- FastPlaid CPU/CUDA comparison rows when requested by the dedicated
  comparison task, with GPU candidate-score rows labelled as primitive-only
- one quiet-wrapper `Mean:` section for CPU i8 same-candidate rerank time
- blocked GPU backend-integration status until the benchmark-only score kernel
  is connected to real Kayak i8 payloads behind an internal backend boundary
- empty GPU timing fields for build/copy/kernel/readback/end-to-end
- a profiling contract that marks future GPU rows as exploratory until
  timings, agreement metrics, explicit vector counts, and repeated runs are
  present

Default gate behavior:

- missing GPU returns a non-zero exit unless `--allow-missing-gpu` is passed
- missing production backend integration returns a non-zero exit unless
  `--allow-missing-kernel` is passed
- missing or timed-out probe tooling still returns a non-zero exit
- if Mojo sees a GPU but the copy roundtrip fails, the scaffold exits non-zero
  and reports `blocked_gpu_copy_probe_failed`
- if Mojo sees a GPU and copy works but the one-document score probe fails, the
  scaffold exits non-zero and reports `blocked_gpu_single_score_failed`
- if Mojo sees a GPU and the benchmark-only candidate score probe fails, the
  scaffold exits non-zero and reports `blocked_gpu_candidate_score_failed`

Reason:

- this makes no-GPU, failed-probe, and missing-backend-integration states
  explicit
- benchmark rows cannot silently substitute CPU timings for GPU timings

Environment correction:

- GPU Pixi tasks no longer set `PYTHONPATH=python`.
- The scripts add the repo `python/` directory to `sys.path` themselves.
- Local verification showed that task-level `PYTHONPATH=python` can make
  `gpu-query`, `mojo run --target-accelerator`, and embedded `DeviceContext`
  creation fail with NVML error `9`.

Reason: GPU availability must be measured from the actual Mojo runtime, not
from an environment variable that poisons the runtime.

## Current Measurements

Latest quiet scaffold run:

- log directory: `.cache/kayak/bench_quiet/20260425T210525Z`
- report path: `.cache/kayak/gpu_i8_rerank_contract/summary.json`
- report status: `blocked_gpu_backend_integration_missing`
- Mojo GPU capability status: `available`
- copy roundtrip status: `ok`
- one-document GPU i8 score probe status: `ok`
- one-document GPU i8 score delta: `0.0`
- one-document GPU i8 score timings:
  - host-to-device mean: `8.419537795608107e-06`
  - kernel mean: `8.16478298162015e-05`
  - device-to-host mean: `3.4643046e-06`
- benchmark-only candidate-score GPU probe status: `ok`
- benchmark-only candidate-score GPU score delta max abs: `0.0`
- benchmark-only candidate-score GPU timings:
  - host-to-device mean: `2.0588047077082257e-05`
  - serial kernel mean: `0.0005140764034334764`
  - two-pass kernel mean: `1.5341061310512197e-05`
  - device-to-host mean: `4.125636247848537e-06`
  - scalar CPU reference mean: `0.0022286610188679244`
- copy probe element counts:
  - `float32_count=3136`
  - `int8_count=131072`
  - `int64_count=129`
- same-candidate CPU i8 reference:
  - `candidate_position_count_total=64`
  - `candidate_score_count_total=64`
  - `query_batch_mean_seconds=0.00014694399942527525`
  - `recall_at_k_vs_kayak_exact=0.55`

For this tiny deterministic shape, the serial benchmark-only GPU kernel is
slower than the CPU same-candidate reference. The two-pass benchmark-only GPU
kernel is faster than the serial GPU probe and faster than the CPU
same-candidate reference for kernel-only time, but this is not a production
speedup claim because the probe uses deterministic generated tensors and is not
connected to real Kayak i8 payload transfer yet.

Latest raw scaffold rerun after removing `PYTHONPATH=python` from GPU tasks:

- report path: `.cache/kayak/gpu_i8_rerank_contract/summary.json`
- Mojo GPU capability status: `available`
- copy roundtrip status: `ok`
- one-document GPU i8 score probe status: `ok`
- benchmark-only candidate-score GPU probe status: `ok`
- report status remains `blocked_gpu_backend_integration_missing`

Reason: the scaffold now proves the local GPU is usable while still refusing
to pretend that a production backend exists.

Latest candidate-score profile smoke rerun:

- report path: `.cache/kayak/gpu_i8_candidate_score_profile/summary.json`
- status: `ok`
- shape `default_2q_8qv_64d_16dv_32k`:
  - candidate scores: `64`
  - two-pass kernel mean: `1.5358767097021603e-05`
  - serial GPU kernel mean: `0.0005561794575471698`
  - two-pass versus serial GPU ratio: `36.212506774389794`
  - score delta max abs: `0.0`
- shape `candidate_window_2q_8qv_256d_16dv_128k`:
  - candidate scores: `256`
  - two-pass kernel mean: `2.757245736612273e-05`
  - serial GPU kernel mean: `0.0007796183790849672`
  - two-pass versus serial GPU ratio: `28.27525921004255`
  - score delta max abs: `0.0`

Exploratory full profile run:

- report path: `.cache/kayak/gpu_i8_candidate_score_profile/profile_summary.json`
- status: `ok`
- all four profile rows reported `twopass_score_delta_max_abs=0.0`
- two-pass versus serial GPU ratios ranged from about `19.24x` to `34.17x`
- two-pass kernel means ranged from `1.5380724526369688e-05` to
  `0.00013566163235294117`

FastPlaid comparison:

- trace: `docs/traces/2026-04-25_gpu_i8_fastplaid_compare.md`
- CPU-device FastPlaid report:
  `.cache/kayak/gpu_i8_fastplaid_compare/summary.json`
- CUDA-device FastPlaid report:
  `.cache/kayak/gpu_i8_fastplaid_compare/cuda_summary.json`
- both reports status: `ok`
- both GPU candidate-score primitive rows reported `score_delta_max_abs=0.0`
- CPU FastPlaid full-search query batch mean:
  `0.0036393629998201504`
- CUDA FastPlaid full-search query batch mean:
  `0.0016935300009208731`
- GPU primitive h2d+kernel+d2h mean beside the CPU FastPlaid run:
  `6.485110803984101e-05`
- GPU primitive h2d+kernel+d2h mean beside the CUDA FastPlaid run:
  `6.513659430293093e-05`

These ratios are scope-labelled profiling context only because FastPlaid is
full search and the current GPU row is still only a candidate-score primitive.

Real-payload GPU rerank profile:

- trace: `docs/traces/2026-04-25_gpu_i8_real_payload_rerank.md`
- report: `.cache/kayak/gpu_i8_real_payload_rerank/summary.json`
- quiet log: `.cache/kayak/bench_quiet/20260425T204818Z`
- report status: `ok`
- payload source: `real_kayak_i8_snapshot`
- candidate scores: `256`
- CPU i8 same-candidate score mean: `0.0003028680002898909`
- GPU two-pass kernel mean: `2.7571585074626866e-05`
- GPU h2d+kernel+d2h mean: `6.498605697685718e-05`
- GPU extension bridge two-pass kernel mean: `2.843207403180797e-05`
- GPU extension bridge h2d+kernel+d2h mean: `6.49974660651413e-05`
- GPU extension bridge call mean: `0.47131670599992503`
- GPU extension bridge host marshalling mean: `0.0038326423327816883`
- two-pass score delta max abs: `4.57763671875e-05`
- GPU h2d+kernel+d2h seconds per CPU score second:
  `0.21456891092705602`
- GPU bridge h2d+kernel+d2h seconds per CPU score second:
  `0.21460658109449926`

This is the first run where the GPU score primitive consumes exported Kayak i8
payload buffers instead of only formula-generated deterministic tensors. The
in-process extension proves the bridge can run against the same real payload,
but the immediate bottleneck is Python-object decode and repeated full-payload
copy, not the two-pass kernel.

## Open Questions

- Which Mojo GPU APIs should own device buffers and kernel launch?
- Should candidate positions remain `Int64` on device or narrow to `Int32`
  after shape validation?
- What tolerance is justified for CPU i8 versus GPU i8 score agreement?
- At what candidate window does CPU top-k readback become a bottleneck?
- Can Mojo expose or own a resident prepared-index GPU buffer object cleanly
  from Python, or should the next bridge use a lower-overhead host buffer
  interface first?

## Validation

Ran:

```bash
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_gpu_i8_rerank_contract -v
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_fastplaid_speed_track -v
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_sdist_manifest -v
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_repo_semantic_inventory -v
pixi run python -m py_compile \
  python/kayak_bridge/gpu_copy_roundtrip.py \
  python/kayak_bridge/gpu_device_capability.py \
  python/kayak_bridge/gpu_i8_candidate_score.py \
  python/kayak_bridge/gpu_i8_real_payload_score.py \
  python/kayak_bridge/gpu_i8_rerank_contract.py \
  python/kayak_bridge/gpu_i8_profile_contract.py \
  python/kayak_bridge/gpu_i8_single_score.py \
  python/kayak_bridge/gpu_probe_output.py \
  python/kayak_bridge/mojo_gpu_i8_rerank.py \
  python/kayak_bridge/plaid_approx.py \
  python/scripts/bench_gpu_i8_rerank_contract.py \
  python/scripts/compare_gpu_i8_fastplaid.py \
  python/scripts/gpu_i8_cpu_reference.py \
  python/scripts/profile_gpu_i8_real_payload_rerank.py \
  python/scripts/profile_gpu_i8_candidate_score.py \
  python/tests/test_gpu_i8_rerank_contract.py \
  python/tests/test_fastplaid_speed_track.py
pixi run mojo run --target-accelerator nvidia:sm_89 \
  -D float32_count=810 \
  -D int8_count=4096 \
  -D int64_count=19 \
  benchmarks/gpu_i8_copy_roundtrip_probe.mojo
pixi run mojo run --target-accelerator nvidia:sm_89 \
  benchmarks/gpu_i8_candidate_score_probe.mojo
pixi run mojo run --target-accelerator nvidia:sm_89 \
  benchmarks/gpu_i8_single_doc_score_probe.mojo
pixi run bench_gpu_i8_rerank_contract_raw
pixi run bench_gpu_i8_rerank_contract
pixi run profile_gpu_i8_candidate_score
pixi run compare_gpu_i8_fastplaid
pixi run compare_gpu_i8_fastplaid_cuda
pixi run profile_gpu_i8_real_payload_rerank_raw
pixi run profile_gpu_i8_real_payload_rerank
pixi run python \
  python/scripts/profile_gpu_i8_candidate_score.py \
  --shape-set profile \
  --allow-missing-gpu \
  --emit-quiet-mean \
  --output .cache/kayak/gpu_i8_candidate_score_profile/profile_summary.json
pixi run compare_gpu_i8_fastplaid_raw
pixi run compare_gpu_i8_fastplaid_cuda_raw
pixi run mojo format \
  python/kayak_bridge/_mojo_gpu_i8_rerank_bindings.mojo \
  benchmarks/gpu_i8_real_payload_score_probe.mojo
```

Observed:

- GPU i8 rerank contract tests: `23/23` passed
- FastPlaid speed-track tests, including same-candidate i8 bridge test:
  `8/8` passed
- sdist manifest tests: `3/3` passed
- repo semantic inventory tests: `3/3` passed
- standalone Mojo copy roundtrip probe passed for `Float32`, `Int8`, and
  `Int64`
- standalone Mojo benchmark-only candidate score probe matched CPU with
  `score_delta_max_abs=0.0` and `twopass_score_delta_max_abs=0.0`
- standalone Mojo one-document i8 score probe matched CPU with
  `score_delta_abs=0.0`
- Python compile check passed
- raw scaffold task wrote
  `.cache/kayak/gpu_i8_rerank_contract/summary.json`
- quiet scaffold task passed and wrote
  `.cache/kayak/bench_quiet/20260425T210525Z`
- raw candidate-score smoke profile task passed and wrote
  `.cache/kayak/gpu_i8_candidate_score_profile/summary.json`
- FastPlaid CPU raw comparison task passed after removing `PYTHONPATH=python`
- FastPlaid CUDA raw comparison task passed after removing `PYTHONPATH=python`
- quiet real-payload GPU profile task passed and wrote
  `.cache/kayak/bench_quiet/20260425T204818Z`
- FastPlaid CPU and CUDA raw comparison tasks passed after removing
  `PYTHONPATH=python`
- exploratory full candidate-score profile task passed without
  `PYTHONPATH=python` and wrote
  `.cache/kayak/gpu_i8_candidate_score_profile/profile_summary.json`
- summary status after host-device-access run:
  `blocked_gpu_backend_integration_missing`
- Mojo GPU capability status after host-device-access run: `available`
- GPU copy roundtrip status after host-device-access run: `ok`
- GPU one-document i8 score status after host-device-access run: `ok`
- GPU benchmark-only candidate-score status after host-device-access run: `ok`
- CPU exact, CPU i8 full-search, and CPU i8 same-candidate reference rows were
  emitted for the small explicit shape

The CPU, copy, one-document kernel, and benchmark-only candidate kernel
timings are scaffold sanity checks only. They are not GPU backend performance
results.
