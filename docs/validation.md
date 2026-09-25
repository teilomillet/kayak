# Validate the full model on your hardware

Initial preparation ran on a Linux machine with no GPU and about 8 GB RAM.
The later Mac checks below cover small-model execution and a full 8B MPS/BF16
smoke test. Full-model runs require sufficient available memory. The same
procedure covers CUDA, MPS, and CPU; automatic
selection uses the available PyTorch backend, not a preferred hardware platform.
Passing the small model tests does not establish the released encoder's numerical equivalence,
task quality, latency, or memory requirements.

The observations below are dated preparation evidence. They do not establish
the status of every checkout or machine. The [readiness audit](records/readiness-audit.md)
records which evidence was freshly exercised on Linux and which remains a
historical record or an open requirement.

## Evidence available here

- The selected 75,557,149-byte projection checkpoint was downloaded at the
  pinned revision and its published SHA-256 verified.
- Its actual configuration and tensors load with `weights_only=True`.
- Head projections and scores were compared against the pinned upstream
  `HeadPair` on deterministic inputs. The matching matrix-vector scoring path
  had zero observed maximum absolute score difference on this CPU.
- The standalone validator, using the default CPU thread configuration, later
  reported `5.72e-6` maximum absolute difference from that recorded fixture,
  within the unchanged `1e-5` tolerance. Both observations are retained; zero
  difference in the first comparison was not a cross-configuration guarantee.
- A first comparison using NumPy matrix-matrix multiplication differed by about
  `1.53e-5`. That was not the upstream engine's scoring operation. The recorded
  fixture uses upstream's matrix-vector operation; tolerance was not increased
  to hide the difference.
- Contract tests and an actual small, randomly initialized Qwen3 exercise local
  loading, padding, batching, Choice inference, and HTTP parity. Its weights
  establish plumbing/numerical behavior, not language understanding.
- The complete suite has passed here with the released heads enabled; the
  current release check results are recorded in [release readiness](release.md).
  A live loopback HTTP round trip exercised the portable validator's
  server/client path. CPU head-only validation completed successfully; its
  full-model and encoder-reference results remain `not_run`.

The fixture in `tests/fixtures/released_heads.json` records the source revision,
input construction, expected scores, and absolute tolerance (`1e-5`, zero
relative tolerance). It was generated using upstream's `HeadPair`, not Kayak's
projection implementation. Both states and candidates were sampled in sequence
with NumPy `default_rng(20260924)`, cast to FP32 and normalized before projection.

## Portable head validation

On an Apple M4 Pro with PyTorch 2.14.0, both Kayak and the pinned upstream
`HeadPair` produced identical FP32 scores, but differed from the original Linux
snapshot by up to `2.48e-5`. The same difference occurred with one and eight CPU
threads. The checkpoint hash matched. This was a cross-platform reference-check
failure, not a difference between Kayak and upstream on the Mac.

The validator now uses two checks, both retaining the original `1e-5` absolute
tolerance and zero relative tolerance:

- An independent functional implementation of the pinned released architecture
  runs in FP64 against the unchanged recorded fixture. Its observed maximum
  error on the Mac was `5.07e-6`.
- Kayak's actual FP32 heads are compared with that independent implementation in
  FP32 on the same CPU backend. The observed maximum error was zero. The reference
  itself was checked against the unmodified pinned upstream implementation using
  three input seeds in both FP32 and FP64, with zero difference in all six checks.

The report labels both reference precisions and retains the historical FP32
snapshot difference as a diagnostic. Model inference and projection precision
are unchanged. Regression tests reject runtime score drift, non-finite scores,
and altered checkpoint files. This validates the released heads on CPU; it does
not establish full-encoder or cross-device equivalence.

## Full 8B smoke test on the local Mac

On 2026-09-24, the full pinned encoder passed on an Apple M4 Pro with 24 GB
unified memory, Python 3.13.5, PyTorch 2.14.0, and Transformers 4.57.6.
Closing unused apps increased available memory from about 6.7 to 13.1 GiB.
The model used MPS and its stored BF16 precision, with no integer quantization.
All weights were cached before loading; the test used `--local-files-only`.

Loading took 10.38 seconds. The tides and billing examples selected `moon` and
`billing`, respectively, in all three repeats, with identical scores within
each case. Warm medians were 0.663 and 0.450 seconds (only two warm samples each).
Input overflow was rejected and the live HTTP result matched direct inference.
MPS tensor allocation was 14.20 GiB, while driver snapshots were about 15.20 GiB.
Sampled system swap peaked at 3.57 GiB. Memory pressure briefly reached warning
level and returned to normal; the monitor observed no critical pressure.

The tested process used `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0` and
`PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9`; global system settings were unchanged.
These caps constrain this process's GPU allocator and do not reserve RAM.
The model process exited after validation to release its allocations.

The working service command from this checkout is:

```sh
PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 \
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9 \
uv run --extra serve kayak serve --device mps --dtype bfloat16 \
  --cache-dir validation/model-cache --local-files-only
```

The local cache must already contain the pinned artifacts. Raw evidence is in
`validation/full-mps-bf16.json`, `validation/full-mps-bf16-memory.json`, and
`validation/full-mps-bf16.log`. These short cases establish execution and parity
with the local service, not application quality, independent encoder equivalence,
CUDA behavior, default FP16 behavior, or memory sufficiency for larger inputs.

## Run on your machine

Use this checkout or its locally built `kayak-0.5.0.tar.gz` source distribution.
The archive includes the validator, reference fixture, tests, and examples.
Use uv to create a Python 3.11+ environment and install the runtime:

```sh
uv sync --extra serve --extra test
uv run --no-sync kayak --version
uv run --no-sync pytest -q
```

Use the PyTorch build appropriate to the machine. The full encoder download is
about 16 GB; runtime and loading also need memory for temporary weights,
activations, and the operating system. Hardware sufficiency is not established
by the weight size alone. No paid GPU allocation is created by these commands.

```sh
# Automatic selection: CUDA, then MPS, then CPU.
uv run --extra serve scripts/validate_model.py --device auto --repeats 21 --output validation/auto.json

# Explicit alternatives, using the same validator:
# --device cuda --output validation/cuda.json
# --device mps  --output validation/mps.json
# --device cpu  --output validation/cpu.json
```

The command downloads the pinned artifacts into the Hugging Face cache, checks
the released heads, loads the actual 8B model, runs two illustrative decisions
21 times each with the command above (default: three), checks overflow rejection,
then starts a loopback HTTP service
and compares client results with direct results. The service reuses the loaded
model, so this check does not allocate a second encoder. It closes the model
when done. `--cache-dir` controls storage and `--local-files-only` prohibits
downloads when all artifacts are already cached.

Use `--heads-only` to check the released projections without loading Qwen3.
Reports explicitly leave full-model and HTTP checks as `not_run` in that mode.
Reports are written even on failure and include version, device, precision,
scores, individual timings, and observed disagreements. Each case separates its
first call from subsequent calls and reports the warm median, minimum, and
maximum. The first case includes initial inference overhead; later cases do
not represent a fresh process. The report also includes process peak RSS and
allocator observations after loading and each case: CUDA peak tensor bytes, or
MPS current tensor and driver bytes. MPS snapshots are not peak measurements.
Process peak RSS is measured on Linux/macOS and reported as `null` elsewhere.
These counters overlap; do not add them together or call them total machine
memory. Monitor operating-system memory and swap separately during the run.
Exact repeatability is
checked on the chosen configuration; a failure is evidence to investigate,
not a reason to suppress a run. These few examples are not a quality benchmark
or a p95 latency estimate.

## Compare with an independent encoder

To evaluate the Transformers/vLLM boundary, run an independent upstream-style
encoder on a suitable host, pinned to the manifest revision:

```sh
vllm serve Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --served-model-name qwen3-8b --runner pooling \
  --max-model-len 2048 --port 8090
```

Record the vLLM version, precision, hardware, and pooler configuration. Avoid
running two full encoders on one machine unless its memory is sufficient.
Pass the independent embedding endpoint to the validator:

```sh
uv run --extra serve scripts/validate_model.py --device auto \
  --reference-emb-url http://REFERENCE_HOST:8090/v1/embeddings \
  --reference-atol YOUR_ACCEPTED_ABSOLUTE_SCORE_TOLERANCE \
  --output validation/reference.json
```

The reference uses the same prepared texts and the independently checked heads.
It requires matching selections and reports every reference score and maximum
absolute difference. Select the tolerance from the precision requirements and
characterized variation; there is deliberately no guessed default for this
cross-backend comparison. The endpoint's encoder revision is an operator
assertion, recorded as an assumption. Without a reference endpoint, the report
keeps `encoder_reference` as `not_run`.

For a release claim, retain these hardware reports, independently labeled task
evaluations, and any failures or near-tie differences. The current implementation
does not claim calibrated probabilities or measured 8B CUDA/MPS parity.

## Decide whether this configuration is ready

Inspect the JSON report before treating a run as successful:

- No `error`, and both `full_model` and `http` are `passed`.
- `model.device` and `model.dtype` match the configuration you intend to use.
  If auto selected CPU unexpectedly, check your PyTorch installation or select
  the intended backend explicitly; do not infer accelerator support from a CPU run.
- All raw repeat timings and scores are retained. Investigate inconsistent
  scores, slow outliers, swap growth, and near-tie selection changes.
- `encoder_reference` is `passed` only after an independent reference comparison.
  Without that comparison, it remains `not_run` even if local/HTTP parity passes.

Once the cache is warm, repeat the command with `--local-files-only` and a new
output filename to check offline operation. Run the local, ranking, and JSONL
examples too, using the same device and cache. Preserve the report alongside
machine RAM/VRAM, accelerator/chip model, and any relevant driver information.

Before calling the model useful for your application, evaluate representative,
independently labeled inputs, including ambiguous cases and near ties. Establish
acceptable quality, warm latency, and memory use for that application before
accepting the results. The two illustrative validator cases cannot supply those
criteria. A pass on one machine does not validate other hardware or precision.

The [evaluation guide](../examples/evaluations/README.md) provides runnable
Choice and ranking checks through a service. For fixed-question suites, use the
[Python evaluation workflow](evaluation-python.md). Supply reviewed application
labels; the starter cases illustrate the formats and do not establish model
quality.
