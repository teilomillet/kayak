# GPU I8 Rerank Architecture

Status: measured scaffold, benchmark-only GPU score probes, and an internal
GPU-targeted Python/Mojo extension probe; no production GPU backend integration
yet.

Date: `2026-04-27`

## Verified Local Facts

- The current compressed PLAID-style CPU lane is
  `kayak/search/plaid_i8_approx_dim128.mojo`.
- That lane stores `Int8` document token codes, one `ScoreScalar` scale per
  token, document offsets, and sampled-centroid posting metadata.
- Scalar aliases are centralized in `kayak/numeric/scalars.mojo`:
  `VectorScalar = Float32`, `ScoreScalar = Float32`, and
  `MetricScalar = Float64`.
- `kayak/runtime/exact_backend.mojo` has an exact-scoring backend trait with a
  comment about a future GPU scorer, but repo search found no production Mojo
  GPU runtime pattern under `kayak/`, `benchmarks/`, `tests/`, or
  `python/kayak_bridge/`.
- The local Mojo toolchain is `0.26.3.0.dev2026041405`.
- On this host, the OS/driver metadata exposes an NVIDIA GeForce RTX 4070 Ti at
  PCI bus `0000:01:00.0`.
- With host device access, `nvidia-smi` works, `/dev/nvidia0`,
  `/dev/nvidiactl`, `/dev/nvidia-uvm`, and `/dev/dri/renderD*` are visible, and
  `gpu-query` sees the card through CUDA.
- The GPU capability observed through Mojo is: device `NVIDIA GeForce RTX 4070
  Ti`, driver `595.58.03`, compute capability `8.9`, target accelerator
  `nvidia:sm_89`, and reported memory about `10.15 GB`.
- The first Mojo `DeviceBuffer` copy probe succeeds locally for `Float32`,
  `Int8`, and `Int64` buffers sized from the explicit scaffold shape.
- The first Mojo GPU scoring probe launches one GPU thread, scores one
  deterministic dim128 i8 candidate document, and matches the CPU scalar
  reference with `score_delta_abs=0.0`.
- The first benchmark-only batched candidate-score probe uses one GPU thread per
  candidate score over the flat tensor contract and matches the CPU scalar
  reference for all reported candidate scores with `score_delta_max_abs=0.0`.
- A separate GPU-targeted Mojo Python extension can be built and imported when
  the repo path is inserted inside Python. Starting Python with
  `PYTHONPATH=python` can make Mojo GPU context creation fail locally with NVML
  error `9`, so GPU benchmark tasks must not use that environment variable.

Reason for recording these facts: the GPU layer should start from observed
repo contracts and local capability probes, not from a FastPlaid-shaped
abstraction.

## Decision

The first GPU target is an internal dim128 i8 candidate-window rerank
accelerator.

It is not:

- full candidate generation
- a full search backend
- a public `search(..., gpu=True)` switch
- a silent CPU fallback

Reason: candidate-window rerank isolates the scoring kernel from search-plan
policy. It makes correctness measurable against the CPU i8 lane for the same
candidate document ids and keeps transfer costs visible instead of hiding them
behind end-to-end search.

## Primitive 1: Device Capability

Own one explicit capability probe before any buffer or kernel work.

Required fields:

- `status`: available, unavailable, tool missing, or error
- backend API and vendor when available
- target accelerator string when available
- device memory when a reliable Mojo/runtime probe exists
- supported dtypes when verified by a runtime probe
- raw probe command and output

Current scaffold: `python/kayak_bridge/gpu_device_capability.py` owns host GPU
inventory and Mojo GPU capability probing. The benchmark script records both,
which separates "hardware exists" from "the Mojo runtime can use it."

Open point: local `gpu-query` exposes memory and target accelerator, but not a
complete dtype support matrix. Supported dtypes remain unverified until a Mojo
device runtime probe is added.

Primary Mojo API basis: Modular documents
[`DeviceContext`](https://docs.modular.com/mojo/stdlib/gpu/host/device_context/DeviceContext/)
as the host-side GPU stream object for allocation, copy, compile, and launch
work, and
[`DeviceBuffer`](https://docs.modular.com/mojo/std/gpu/host/device_context/DeviceBuffer/)
as device-resident storage with explicit copy operations. Modular's
[`gpu` package docs](https://docs.modular.com/mojo/std/gpu/) describe
`global_idx`, `block_idx`, and related per-thread grid primitives used by the
candidate-score probe.

## Primitive 2: Device Buffers

The first real GPU code should introduce typed, owned buffers for:

- `Int8` token codes
- `ScoreScalar` / `Float32` query values, token scales, and candidate scores
- `Int64` host offsets and candidate positions, with an explicit decision if a
  narrower device type is used

Every copy boundary must time and report:

- host-to-device query copy
- host-to-device index/candidate copy
- device-to-host score readback

Reason: without separate copy timing, a faster kernel can still lose
end-to-end. Hidden global caches are excluded initially because they would make
first-run and warmed-run claims hard to compare.

Current scaffold: `benchmarks/gpu_i8_copy_roundtrip_probe.mojo` allocates
typed `DeviceBuffer`s, copies host-to-device and device-to-host, synchronizes,
and verifies roundtrip values. The Python wrapper records per-dtype element
counts and mean copy timings in `gpu_copy_roundtrip_probe`.

Reason: this validates that the local Mojo GPU runtime can do the minimum
buffer work needed by the future scoring kernel before any kernel code is
introduced.

## Primitive 3: Flat Dim128 Tensor Views

The GPU boundary uses flat dim128 views and explicit vector counts:

| Name | Shape | Dtype |
| --- | --- | --- |
| query values | `[query_count, query_vector_count, 128]` | `VectorScalar` / `Float32` |
| token codes | `[total_doc_vectors, 128]` | `Int8` |
| token scales | `[total_doc_vectors]` | `ScoreScalar` / `Float32` |
| doc offsets | `[document_count + 1]` | host `Int64`, device type unresolved |
| candidate positions | `[query_count, candidate_k]` | host `Int64`, device type unresolved |
| candidate scores | `[query_count, candidate_k]` | `ScoreScalar` / `Float32` |

Reason: these shapes match the CPU i8 payload and preserve vector count as a
first-class axis in the API, benchmark, and trace notes.

## Primitive 4: GPU Rerank

Proposed internal primitive:

```text
plaid_i8_gpu_rerank_candidates_dim128(
    query_values,
    token_codes,
    token_scales,
    doc_offsets,
    candidate_positions,
) -> candidate_scores
```

Initial behavior:

- score one query batch against provided candidate document ids
- return candidate scores
- keep top-k selection on CPU until GPU score correctness and transfer timing
  are stable

Reference:

- CPU i8 scores for the same query batch and the same candidate document ids
- exact MaxSim remains the broader quality reference, but CPU i8 is the direct
  implementation reference for this GPU primitive

Reason: score agreement against CPU i8 catches kernel/layout mistakes before
the design absorbs candidate generation or top-k complexity.

Current scaffold: the existing CPU i8 Mojo lane now exposes candidate
generation separately from same-candidate scoring. The benchmark report writes
`cpu_i8_same_candidate_reference`, including candidate positions, candidate
score count, rerank-only timing, and recall versus exact.

Current GPU probes:

- `benchmarks/gpu_i8_single_doc_score_probe.mojo` validates the smallest
  scoring unit. It copies one deterministic query and one candidate-document
  token block to the GPU, launches a one-thread kernel, reads back one score,
  and compares it with the same scalar CPU formula.
- `benchmarks/gpu_i8_candidate_score_probe.mojo` validates the flat candidate
  scoring boundary. It copies query values, token codes, token scales, doc
  offsets, and candidate positions to the GPU, compares a serial
  one-thread-per-candidate kernel with a two-pass token-parallel kernel, reads
  back candidate scores, and compares every score with the CPU scalar formula.
- `benchmarks/gpu_i8_real_payload_score_probe.mojo` consumes benchmark-exported
  Kayak i8 payload snapshots: query values, prepared token codes, token scales,
  document offsets, CPU-generated candidate positions, and CPU i8 reference
  scores. Candidate generation and top-k remain on CPU.
- `python/kayak_bridge/_mojo_gpu_i8_rerank_bindings.mojo` and
  `python/kayak_bridge/mojo_gpu_i8_rerank.py` build a separate GPU-targeted
  Python extension and run the same real-payload scoring boundary in-process:
  a full-copy bridge row, a prepared-session list bridge row, and a
  prepared-session ndarray bridge row, and a prepared-session address bridge
  row. The prepared sessions copy token codes, token scales, and document
  offsets to device once inside a single profiling call, then time
  query/candidate H2D, two-pass kernel, and D2H against those resident index
  buffers. The ndarray row passes contiguous NumPy arrays directly to the same
  Mojo function to isolate Python list materialization. The address row passes
  raw typed array addresses and reconstructs `UnsafePointer` values inside Mojo
  to test per-element `PythonObject` indexing overhead. The address serving row
  uses the same typed-address tensor boundary without internal `benchmark.run`
  sections to measure a serving-shaped extension call.

Reason: this checks kernel launch, pointer argument types, scalar i8 math,
copy boundaries, readback, `global_idx` indexing, doc offsets, and candidate
positions before production backend integration. It is not evidence of speedup.
The real-payload probe additionally verifies that the GPU score math agrees
with Kayak's prepared i8 payload representation rather than only synthetic
formula-generated buffers. The extension bridge verifies that a Python/Mojo GPU
boundary is feasible, but its current element-wise Python-object decode and
per-call allocation are much slower than the GPU copy+kernel+readback work and
should not be mistaken for a production backend. The prepared-session rows test
device residency and host-ingestion surfaces without committing to a
Python-owned prepared object.

Exploratory optimization result: the two-pass probe is consistently faster
than the serial GPU probe on the measured deterministic shapes while preserving
`score_delta_max_abs=0.0`. The profile task is:

```bash
pixi run profile_gpu_i8_candidate_score
pixi run profile_gpu_i8_real_payload_rerank
```

Reason: this isolates kernel-structure work from the future bridge integration.
The result supports carrying the two-pass structure forward, but not claiming a
production speedup yet.

Latest bridge finding: on the `2 queries x 8 query vectors x 256 documents x
16 document vectors x candidate_k 128` real-payload smoke shape, the full-copy
in-process extension reports about `6.47e-05s` for H2D + two-pass kernel + D2H
and score agreement within `4.57763671875e-05`, matching the standalone probe.
The prepared list bridge reports about `2.64e-05s` to prepare resident index
buffers and about `3.88e-05s` for score H2D + two-pass kernel + D2H with those
index buffers already resident. The prepared ndarray bridge reports about
`2.65e-05s` to prepare resident index buffers, about `3.85e-05s` for score
H2D + two-pass kernel + D2H, and about `2.48e-05s` of Python host marshalling
instead of the list bridge's about `3.22e-03s`. That validates device residency
as useful and validates direct ndarray input as useful for Python-side
materialization on this shape.

The prepared address bridge reports about `2.65e-05s` to prepare resident index
buffers, about `3.71e-05s` for score H2D + two-pass kernel + D2H, about
`1.19e-05s` of Python host marshalling, and the same
`score_delta_max_abs=4.57763671875e-05`. That validates typed address ingestion
for this internal probe shape.

It does not validate a production backend because the extension call remains
about `0.67s` for the address row. That debunks `.tolist()` and per-element
`PythonObject[index]` access as the only extension-call bottlenecks in the
profiled function; the remaining cost is likely dominated by the internal
`benchmark.run` harness, allocation, and session setup work inside the call.

Follow-up finding: the no-internal-benchmark address serving call reports
about `2.62e-04s` extension-call time, about `1.02e-05s` Python host
marshalling, and the same `score_delta_max_abs=4.57763671875e-05` on the same
`2 x 8 x 256 x 16 x candidate_k 128` shape. That validates the hypothesis that
the previous `0.67s` address profile was dominated by internal profiling
scaffolding, not by the Python extension boundary alone. It also gives a first
end-to-end serving-shaped measurement that is slightly faster than the CPU i8
same-candidate reference on this small shape: about `0.262ms` versus about
`0.296ms`.

This still does not validate a production GPU backend. The serving call
allocates buffers, copies the prepared index tensors, scores, reads back scores,
checks the CPU reference, and returns Python scores inside one call. It does
not reuse a prepared GPU index across calls, and candidate generation plus
top-k remain on CPU.

Shape-sweep finding: `profile_gpu_i8_address_serve_sweep` kept CPU i8 score
agreement across six explicit vector-count shapes. The one-shot serving call
remains useful as a fixed-overhead baseline, but after adding the explicit
prepared handle the one-shot serving call is no longer the best measured
implementation. The latest quiet sweep reports one-shot GPU score ratios from
about `0.719x` to `2.000x` of CPU same-candidate scoring, so the one-shot call
is still not uniformly faster.

Resident-session finding: an in-call repeated address session copies the
prepared index tensors once, then scores the same query and candidate window
four times against those resident buffers. Its per-iteration score time ranged
from about `0.282x` to `0.637x` of CPU same-candidate scoring, and the
CPU-candidate-generation-plus-resident-iteration envelope ranged from about
`0.701x` to `0.896x` of CPU candidate generation plus CPU score.

Multi-window resident finding: a second in-call resident probe copies the same
prepared index tensors once, then scores four different query/candidate windows
inside the call. Its latest per-window score time ranged from about `0.327x`
to `0.585x` of CPU same-candidate scoring per window, and the
CPU-candidate-generation-plus-multi-window-resident envelope ranged from about
`0.745x` to `0.884x` of CPU candidate generation plus CPU score per window.
This is evidence that prepared-index residency and allocation reuse are likely
higher leverage than immediate dim128 kernel rewrites on the swept shapes. It
also debunks the narrower hypothesis that the resident win only comes from
repeating one identical candidate window.

Ownership finding: a first Python-visible `PreparedGpuI8RerankDim128` object was
not kept because Mojo Python `module.add_type[...]` requires `Writable`, while
`DeviceContext` cannot derive `Writable`. A follow-up probe verified that a
heap-owned Mojo struct can still own `DeviceContext`, `DeviceBuffer`, and
`HostBuffer` fields and be controlled by an explicit integer handle with
`prepare -> score -> release`. This is not a public object and not a hidden
cache; it is unsafe internal ownership with explicit release.

Explicit-handle finding: the latest quiet
`profile_gpu_i8_address_serve_sweep` row keeps the prepared i8 index resident
across separate Python score calls. Prepared-handle score ratios ranged from
about `0.250x` to `0.371x` of CPU same-candidate scoring per window, and the
CPU-candidate-generation-plus-prepared-handle envelope ranged from about
`0.719x` to `0.817x` of CPU candidate generation plus CPU scoring. All six
cases preserved CPU i8 score agreement.

Top-k return finding: the prepared handle can return only `[query_count,
top_k]` positions and scores after Mojo-side host top-k selection. The latest
quiet sweep preserved `topk_position_agreement=1.0` for all six cases. Top-k
return ratios ranged from about `0.162x` to `0.309x` of CPU same-candidate
scoring, and the CPU-candidate-generation-plus-top-k envelope ranged from about
`0.412x` to `0.750x` of CPU candidate generation plus CPU scoring. Returning
top-k was faster than returning all candidate scores on every swept case.

Reason: this keeps measured evidence ahead of abstraction. The next
optimization targets are the measured candidate-generation envelope and an
internal search boundary around the explicit handle, not immediate public API
surface area.

No-reference top-k finding: the prepared handle now has a serving-shaped top-k
call that does not pass CPU reference scores into the Mojo extension. CPU scores
are used only after the call to validate returned positions. The default quiet
sweep preserved `topk_position_agreement=1.0` on all six cases, and the wide
sweep preserved `topk_position_agreement=1.0` on all five cases. On the default
sweep, no-reference top-k ranged from about `0.145x` to `0.264x` of CPU
same-candidate scoring; the candidate-generation-plus-no-reference-top-k
envelope ranged from about `0.146x` to `0.636x` of CPU candidate generation
plus CPU scoring. On the wide sweep, no-reference top-k ranged from about
`0.104x` to `0.225x` of CPU same-candidate scoring; the full envelope ranged
from about `0.105x` to `0.511x`.

Reason: this removes validation-only input from the serving call without hiding
correctness. The result also falsifies a narrow optimization hypothesis:
removing reference-score input halves host marshalling on some rows, but total
extension-call time is within noise of the validating top-k path on the measured
shapes.

Wide top-k finding: `profile_gpu_i8_address_serve_wide_topk` adds explicit
larger candidate-window and vector-count cases without changing the default
sweep. The quiet run on `nvidia:sm_89` preserved `topk_position_agreement=1.0`
on all five cases. Prepared-handle top-k ratios ranged from about `0.104x` to
`0.232x` of CPU same-candidate scoring, and the
CPU-candidate-generation-plus-top-k envelope ranged from about `0.331x` to
`0.560x` of CPU candidate generation plus CPU scoring. The first wide run also
debunked the fixed `1.0e-4` absolute score-delta threshold for
`query_vector_count=32`; GPU i8 score agreement now reports an explicit
vector-count-aware tolerance field.

Reason: the wider run validates the prepared-handle top-k boundary beyond the
original small windows, while the tolerance finding keeps correctness evidence
visible instead of hiding a widened threshold.

Candidate-generation cleanup finding: CPU candidate generation was the next
visible limiter after GPU scoring moved behind an explicit prepared handle.
Four CPU-side changes improved that boundary without adding GPU abstraction:
full candidate windows now return all document positions directly in Mojo,
`top_positions_by_score` uses a bounded worst-first heap instead of repeated
full scans, the Python bridge skips the Mojo candidate-generation call entirely
when `candidate_k >= document_count`, and non-full candidate generation can
ingest contiguous `float32` query tensors by typed address instead of Python
lists. The latest wide quiet run
measured full-window candidate generation at about `0.00000103s/window`
(`candidate512`) and `0.00000147s/window` (`candidate1024`). The remaining
non-full-window candidate-generation cases measured about `0.000427s/window`
(`doc_vectors64`), `0.000923s/window` (`query_vectors32`), and
`0.000774s/window` (`query_batch4`).

Reason: candidate-window identity is exact when `candidate_k >= document_count`,
so proxy sorting cannot improve correctness. The heap keeps the prior
descending-score, lower-position tie order while reducing selection work from
`O(k * n)` to bounded heap maintenance plus ordered extraction. The typed
address bridge is justified because the query tensor is already contiguous and
it avoids Python list materialization without changing candidate semantics.

Candidate-generation follow-up finding: local hot-path edits after the
no-reference top-k checkpoint did not produce a decision-quality win. A dense
reset variant, streamed centroid top-selection variant, and direct query-pointer
candidate bridge were all removed after quiet sweeps showed default or wide
envelope regressions. Reserve-only preallocation was also removed because the
measured effect was mixed and small. The final reverted-code sweeps remained
`ok`: the default CPU-candidate-plus-no-reference-top-k envelope ranged from
about `0.145x` to `0.671x`, and the wide envelope ranged from about `0.103x`
to `0.510x`.

Reason: the evidence now points to a missing profiling boundary, not another
small loop edit. The next candidate-generation step should break timing and
work counts into centroid scoring, centroid selection, posting accumulation,
and final candidate top-k before moving any of that work onto GPU.

Candidate-generation breakdown finding: the dedicated benchmark-only
breakdown now profiles full candidate generation, centroid scoring, centroid
selection, posting accumulation, and final candidate top-k from a prepared i8
index handle. It records query vector count, document vector count, selected
centroid count, posting visits, touched documents, and output candidate count.
The quiet wide run was `ok` on all five cases. On non-full candidate windows,
centroid selection and posting accumulation are the leading measured costs:
`query_vectors32` measured about `0.000206s/batch` for centroid selection and
`0.000213s/batch` for posting accumulation; `doc_vectors64` measured about
`0.000100s/batch` for posting accumulation and `0.000063s/batch` for final
top-k; `query_batch4` measured about `0.000103s/batch` for centroid selection,
`0.000112s/batch` for posting accumulation, and `0.000127s/batch` for final
top-k. The full-window rows remain shortcut rows, so their decomposed timings
are explanatory only.

Reason: the next GPU work should target a measured primitive with explicit
posting fanout and vector-count fields. The current evidence points to
reducing or restructuring candidate-generation work before treating GPU as a
direct port of the CPU loop.

Centroid-budget finding: a benchmark-only sweep now varies
`centroids_per_query_vector` while holding `centroid_count`, `candidate_k`,
document vectors, query vectors, and i8 payload fixed. On the wide non-full
case set, the fastest no-loss budgets versus the `32`-centroid baseline were
`4` for `query_vectors32`, `16` for `doc_vectors64`, and `8` for
`query_batch4`. On the default non-full case set, the fastest no-loss budgets
were `4`, `4`, `16`, `4`, and `24` across the five cases. Lower budgets
preserved or improved the baseline exact-reference recall on these synthetic
rows, while reducing posting visits substantially. The winning budget is
shape-dependent.

Reason: this debunks `32` as a universally justified fixed budget, but it does
not justify a new static default. The next optimization should be a measured
shape-aware budget policy or calibration step, followed by a same-shape
FastPlaid comparison.

FastPlaid comparison finding: on the explicit wide `candidate1024` shape
(`document_count=1024`, `document_vector_count=16`, `query_count=2`,
`query_vector_count=8`, `candidate_k=1024`, `top_k=10`), the latest quiet
comparison measured CPU candidate generation plus GPU no-reference top-k at
about `0.000166s/window` against FastPlaid CPU full search at about
`0.007614s/batch`, and about `0.000166s/window` against FastPlaid CUDA full
search at about `0.002212s/batch`. This remains a scope comparison, not a
public backend claim, because Kayak starts from CPU-provided candidate windows
and FastPlaid is timed as full search. On this synthetic shape, Kayak i8 recall
was `1.0`; the observed FastPlaid recall was `0.4` on CPU and about `0.45` on
CUDA.

Reason: the comparison is now strong enough to justify building an internal
search boundary around the primitive, but not strong enough to claim a public
GPU backend speedup.

## Primitive 5: Measurement Contract

Every GPU row must report:

- build or prepare time
- host-to-device copy time
- kernel time
- device-to-host readback time
- end-to-end time
- query count
- query vector count
- document count
- document vector count
- total document vector count
- candidate window size
- recall and order agreement versus CPU i8
- recall versus Kayak exact when exact reference is part of the run

Every profiling row starts as `exploratory`. It can become decision-quality
only after the timing fields, agreement fields, explicit vector counts, and at
least three repeated runs are present. It is falsified immediately when score
agreement versus CPU i8 exceeds the stated tolerance.

The default benchmark command must use the quiet wrapper for decision-quality
comparisons:

```bash
pixi run bench_gpu_i8_rerank_contract
```

Reason: the repo already treats repeated quiet runs as the default for
performance-sensitive claims.

Current scaffold note: the quiet-wrapper `Mean:` section is the CPU i8
same-candidate rerank reference query-batch time. GPU scoring timings remain
`null` in the production backend row until the production rerank integration
exists. The copy probe and benchmark-only score probes record their timings
separately; they are scaffold checks, not backend speedup results.

## FastPlaid Comparison Boundary

FastPlaid remains a required competitive baseline, but it must be compared at
the right scope.

Current comparison task:

```bash
pixi run compare_gpu_i8_fastplaid
pixi run compare_gpu_i8_fastplaid_cuda
pixi run compare_gpu_i8_fastplaid_wide_candidate1024
pixi run compare_gpu_i8_fastplaid_wide_candidate1024_cuda
```

The report intentionally keeps these surfaces separate:

- Kayak exact CPU: full-search correctness reference
- Kayak i8 CPU: full-search i8 candidate-window baseline
- FastPlaid CPU or CUDA: full-search external system baseline
- Kayak GPU i8: benchmark-only candidate-score primitive
- Kayak GPU i8 prepared-handle top-k: real Kayak i8 payload, CPU-provided
  candidate windows, explicit GPU handle, top-k positions/scores returned
- Kayak GPU i8 prepared-handle no-reference top-k: the same boundary, but CPU
  reference scores are not passed into the Mojo serving call and are used only
  for post-call validation

Reason: FastPlaid search includes indexing, candidate generation, approximate
search, and top-k output. The Kayak GPU rows now include both a deterministic
flat dim128 candidate-score primitive and a real-payload prepared-handle top-k
boundary, but candidate generation still starts on CPU and the handle boundary
is internal. Putting those numbers in one report is useful profiling context,
but treating the ratio as a backend speedup claim would be wrong until the GPU
primitive is integrated into a complete search path.

Wide FastPlaid finding: on the explicit `1024 x 16` documents,
`2 x 8` queries, `candidate_k=1024` shape, the prepared-handle top-k boundary
remained correct with `topk_position_agreement=1.0`. The latest no-reference
top-k comparison measured the CPU-candidates-plus-GPU-top-k envelope at about
`0.0218x` of CPU FastPlaid full-search batch time and about `0.0752x` of CUDA
FastPlaid full-search batch time.

Reason: the wide comparison shows the current internal rerank/top-k boundary is
competitive on this explicit synthetic shape, while still requiring a complete
search-path integration before any public backend speedup claim.

Required fields for every FastPlaid comparison row:

- FastPlaid device: `cpu`, `cuda`, or another explicit device string
- FastPlaid version and indexing knobs
- query count and query vector count
- document count, document vector count, and total document vector count
- candidate score count for the GPU primitive
- top-k return count and top-k agreement for prepared-handle rows
- recall@k versus Kayak exact for full-search rows
- a scope warning when GPU primitive timings are shown beside full-search rows

Reason: this prevents a GPU implementation from looking successful only because
it was compared against a larger operation than it actually performs.

## Primitive 6: Backend Boundary

Keep the first GPU entry point internal. Do not add public SDK or service
surface until the primitive is correct, timed, and useful on at least one
explicit vector-count shape.

Allowed first integration:

- an internal benchmark-only path
- a narrow `plaid_i8_gpu_rerank_candidates_dim128` module once the Mojo GPU API
  choice is verified

Blocked until evidence exists:

- public API flags
- planner/runtime automatic device dispatch
- CPU fallback inside GPU-labelled rows

Reason: public backend selection implies stability and support. The current
evidence only justifies a measured primitive.

## Validation Ladder

1. Capability probe writes an explicit blocked report on no-GPU hosts.
2. CPU i8 same-candidate reference row runs on the same explicit shape.
3. Device buffer roundtrip copies `Int8`, `Float32`, and offsets without score
   computation.
4. One query, one candidate document score matches CPU i8 within a stated
   tolerance. Current local result: `score_delta_abs=0.0`.
5. Batched candidate scores match CPU i8 for multiple query vector counts,
   document vector counts, and candidate windows. Current first local result:
   `score_delta_max_abs=0.0` for `query_count=2`,
   `query_vector_count=8`, `document_count=64`,
   `document_vector_count=16`, and `candidate_k=32`.
6. Compare serial GPU scoring with token-parallel GPU scoring across an
   explicit profile shape set.
7. Connect the winning benchmark-only structure to exported real Kayak i8
   payload snapshots while candidate generation and top-k stay on CPU. Current
   local result: `twopass_score_delta_max_abs=4.57763671875e-05` for
   `query_count=2`, `query_vector_count=8`, `document_count=256`,
   `document_vector_count=16`, and `candidate_k=128`.
8. Move from snapshot export to an internal prepared-index GPU rerank boundary
   that avoids file staging. Current local result: a single-call prepared
   session keeps index buffers resident and reduces score H2D + kernel + D2H
   from about `6.47e-05s` to `3.88e-05s` on the smoke shape, while preserving
   `score_delta_max_abs=4.57763671875e-05`.
9. Replace the single-call prepared session with a reusable internal ownership
   boundary or lower-overhead host buffer path. Current local result: direct
   ndarray input reduces Python host marshalling from about `3.22e-03s` to
   about `2.48e-05s`, but does not reduce the extension call, which remains
   about `0.72s`.
10. Test typed address ingestion. Current local result: address input preserves
   CPU i8 agreement and reduces Python host marshalling to about `1.19e-05s`,
   but the profiled extension call remains about `0.67s`.
11. Add a no-internal-benchmark serving-style address call to separate
   extension-call overhead from profiler harness overhead. Current local
   result: about `0.262ms` on the smoke shape, preserving
   `score_delta_max_abs=4.57763671875e-05`.
12. Sweep the no-internal-benchmark address call across explicit vector-count
   shapes. Current quiet result: one-shot GPU score ratios range from about
   `0.524x` to `1.364x` of CPU same-candidate scoring, so the one-shot call is
   not uniformly faster.
13. Test in-call prepared-index residency without introducing a hidden global
   cache. Current quiet result: resident-session per-iteration score ratios
   range from about `0.282x` to `0.637x` of CPU same-candidate scoring across
   the six swept cases, but this repeats the same candidate window inside one
   extension call and does not prove cross-call ownership.
14. Test different query/candidate windows inside the same in-call resident
   session. Current quiet result: multi-window per-window score ratios range
   from about `0.327x` to `0.588x` of CPU same-candidate scoring per window
   across the six swept cases, while preserving CPU i8 score agreement.
15. Test explicit cross-call prepared-index ownership without a public GPU
   object or hidden global cache. Current quiet result: prepared-handle
   per-window score ratios range from about `0.250x` to `0.371x` of CPU
   same-candidate scoring, and CPU candidate generation plus prepared-handle
   GPU scoring is faster than CPU candidate generation plus CPU score in all
   six swept cases.
16. Return only top-k positions and scores from the explicit handle. Current
   quiet result: top-k return ratios range from about `0.162x` to `0.309x`
   of CPU same-candidate scoring, top-k order agreement is `1.0` in all six
   cases, and top-k return is faster than returning all candidate scores on
   every swept case.
17. Add a no-reference top-k serving call and validate it outside the Mojo
   extension. Current quiet result: no-reference top-k preserves
   `topk_position_agreement=1.0` on the default and wide sweeps, while total
   extension-call timing remains within noise of the validating top-k path.
18. Reduce CPU candidate-generation overhead before adding more GPU
   abstraction. Current quiet result: full-window candidate generation skips
   proxy sorting and Python-to-Mojo candidate generation, while bounded heap
   selection and typed-address query ingestion improve the non-full-window wide
   cases.
19. Compare the same wide candidate1024 shape against FastPlaid CPU and CUDA.
   Current quiet result: CPU candidates plus GPU no-reference top-k is faster
   than FastPlaid full search in this scope-limited synthetic comparison, with
   explicit recall and vector-count fields recorded.
20. Add a dedicated non-full candidate-generation breakdown before more local
   loop edits. Current quiet result: dense reset, streamed centroid selection,
   direct pointer candidate scoring, and reserve-only preallocation did not
   produce a decision-quality win across default and wide sweeps. The new
   breakdown shows centroid selection, posting accumulation, and final
   candidate top-k as the dominant non-full-window substeps.
21. Sweep centroid budgets as the first policy lever on posting fanout. Current
   quiet result: lower budgets preserve or improve the `32`-centroid baseline
   recall on measured default and wide non-full synthetic cases, but the
   fastest no-loss budget is shape-dependent.
22. Quiet benchmark compares copy, kernel, readback, CPU candidate generation,
   CPU top-k, and end-to-end times after the resident ownership boundary
   exists.
23. Only after a measured win, consider public API design.

## Falsification Conditions

This direction should be revised if:

- transfer time dominates end-to-end on the target candidate windows
- score deltas versus CPU i8 are unstable or unexplained
- Mojo cannot expose reliable device memory or dtype capability information
- useful shapes require candidate generation on GPU before rerank wins
- top-k readback becomes the real bottleneck before scoring is solved

These are not failures of the project. They are evidence that the primitive or
boundary is wrong.
