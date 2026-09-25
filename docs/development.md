# Development and profiling

This guide measures code around inference: request validation, serialization,
the Python client, and HTTP handling. All benchmarks use fixed fixtures and
local test transports. They load no model, download nothing, and contact no
external service. There is no production monitoring endpoint or telemetry SDK.

## Code boundaries

The package lives directly in `kayak/`. Setuptools explicitly includes only
`kayak` packages; tests and benchmarks remain development tools.

| Path | Responsibility |
| --- | --- |
| `kayak/decisions.py` | Decision contracts, independent snapshots of caller data, and score transformations |
| `kayak/client.py` | Sync/async HTTP effects with shared response validation and error translation |
| `kayak/ranking.py` | Ranking values and pure conversion to/from the canonical Choice decision |
| `kayak/server.py` | Body limits, admission, timeouts, and ownership of the resident model |
| `kayak/bundle.py` | Artifact identity and loading boundary |
| `kayak/runtime/_preparation.py` | Pure text recipe and token-budget policy |
| `kayak/runtime/_loading.py` | Validated model loading and resource allocation |
| `kayak/runtime/_model.py`, `kayak/_heads.py` | Serialized execution, consumed encoder/tokenizer interfaces, numerical recipe, and close |

The [engineering conventions](engineering.md) establish Go as the main design
reference, adapted to Python and its functional core. Structural checks enforce
acyclic explicit imports and the 1,000-line source-file limit in both ordinary
pytest and the non-inference suite.

Data flows from caller-owned values through validation into a request that owns
its dictionaries, then into transport or execution. A response is validated
before the client returns it. Contract objects forbid field reassignment, but
their nested dictionaries are mutable. `request_from` therefore revalidates and
copies typed inputs too; `revalidate_instances="always"` is essential to that
ownership guarantee. Validation remains enabled at these boundaries.

Model lifetime is stateful: startup loads and checks readiness; admission owns
one active task; timeout or caller cancellation leaves that task occupying
capacity; shutdown stops admission, drains it, and closes the model. Explicit
event-controlled tests cover these histories without depending on model speed.

## Correctness checks

```sh
pixi run --locked -e test test-code
pixi run --locked -e test lint
pixi run --locked -e test format-check

# Equivalent uv environment; these checks need no PyTorch.
uv sync --extra test
uv run --no-sync pytest -q -m 'not inference'
HYPOTHESIS_PROFILE=ci uv run --no-sync pytest -q -m 'not inference'
```

`--no-sync` uses the environment just prepared by `uv sync`. CI uses `uv venv`
and `uv pip install` where it needs a specific CPU build or independently
installed packages, then runs checks with `uv run --no-sync`.

Hypothesis generates Unicode text and ordered dictionaries to challenge
round-trip preservation, input ownership, revalidation of mutated inputs, and
untrusted response handling. Limit tests exercise values on both sides of the
question, candidate, and aggregate character boundaries. HTTP tests exercise
streamed byte limits, error recovery, admission races, cancellation, and shutdown.

The default Hypothesis profile runs 100 examples per generated test; the `ci`
profile allows 300 and disables wall-clock deadlines. Timing belongs in the
benchmark suite. Hypothesis retains failures locally in `.hypothesis/`; keep its
reproduction output when reporting a failure. This generated coverage supports
the tested contracts; it is not an exhaustive proof.

The regular CI workflow checks the surrounding code without inference packages
and retains the existing CPU inference regression jobs. Full local regression
checks use `uv run --extra local --extra test pytest` with the optional inference dependencies.

## Ruff and static typing

Every project function has annotated parameters and a return type, including
fixtures, async helpers, scripts, and benchmarks. Ruff enforces its annotation
rules (`ANN`, including `ANN401` for explicit `Any`) alongside imports, basic
correctness, Bugbear (`B`), Python modernization (`UP`), redundant `else` after
`return` (`RET505`), and unused suppressions (`RUF100`). Formatting is checked
separately with `ruff format --check`.

Mypy runs in strict mode over `kayak/`, `tests/`, `scripts/`, `benchmarks/`, `examples/`, and `stubs/`.
It rejects untyped functions, bare generic containers, explicit `Any`, types lost
through missing imports, and dynamic return values escaping typed functions.
The Pydantic plugin checks constructors using actual field types. Invalid-input
tests may deliberately bypass a static type error to exercise runtime rejection;
these targeted suppressions do not weaken the production contracts.

```sh
# Full static checking needs the libraries whose model interfaces it analyzes.
# Use the appropriate CPU or accelerator PyTorch build; no weights are downloaded.
uv sync --extra local --extra test --extra bench
uv run --no-sync ruff check kayak tests scripts benchmarks examples stubs
uv run --no-sync ruff format --check kayak tests scripts benchmarks examples stubs
uv run --no-sync mypy
```

The Python 3.11/3.13 CI jobs run these checks before tests. Lightweight development
can still run Ruff and the non-inference tests through the test Pixi environment.
Type checking follows the active Python environment so it uses that version's
installed dependency annotations.

Protocols describe the operations consumed from the encoder, tokenizer, and
service model without importing inference packages during a normal `import kayak`.
Structured dictionaries describe checkpoint options, reference fixtures, and
embedding responses. Unknown JSON values are narrowed at validation boundaries;
heterogeneous diagnostic reports use `object` values rather than unchecked
dynamic types. Small casts at external factories and deserialization boundaries
state the expected third-party contracts; they do not replace runtime validation.

`stubs/` describes only the untyped pyperf and Rust tokenizer calls used by our
development tools. The stubs are included in source distributions, excluded from
the wheel, and exercised by benchmark smoke checks and the real tiny-model
fixture. Keep them aligned when changing those calls or upgrading dependencies.
Third-party internals retain their own typing; the policy applies to our source
and interfaces, without trying to re-annotate PyTorch or Transformers.

## Timing measurements

For actual encoder execution, use the [local inference eval loop](inference-evals.md).
The benchmarks below intentionally isolate the surrounding code.

Use the same environment and machine for before/after measurements. Keep other
work idle. The benchmark environment has its own locked Pixi dependencies:

```sh
mkdir -p .benchmarks
pixi run --locked -e bench benchmark -o .benchmarks/before.json
# Make a change, run the correctness checks, then repeat:
pixi run --locked -e bench benchmark -o .benchmarks/after.json
pixi run --locked -e bench pyperf compare_to \
  .benchmarks/before.json .benchmarks/after.json --table
pixi run --locked -e bench pyperf check .benchmarks/after.json
```

Alternatively, use `uv run --extra bench -m benchmarks.bench_overhead`.
`--workload wide.client` selects one workload; `--fast` gives a rougher, quicker
measurement. Default pyperf settings use calibrated loops, warmups, and multiple
worker processes. Reports retain samples, variation, environment/dependency
versions, fixture sizes, and hashes of the measured Kayak source files. Reports
in `.benchmarks/` are ignored by Git; retain the files with a performance review.

The small fixture has one question and two candidates. The wide fixture reaches
the supported limits of 32 questions and 256 candidates. The padded fixture has
four 64,000-character Unicode texts with edge spaces, just below the character
and HTTP byte limits. It challenges orchestration allocations, not tokenizer
limits or model quality. All three include Unicode text. Each fixture measures:

| Operation | Included work |
| --- | --- |
| `validate_python` | Snapshot and validate typed Python inputs |
| `validate_json` | Decode and validate the request body |
| `prepare_texts` | Prepare question states and verbatim candidates; no tokenization |
| `build_result` | Current typed-answer construction and enclosing result validation, starting from Python score lists |
| `build_result_once` | Experimental assembly of fresh fields followed by full result validation; benchmark only |
| `decode_result` | Validate the full response contract |
| `encode_dict`, `encode_json` | Compare dictionary-based and direct JSON response encoding |
| `client` | Public client call through MockTransport, including request and response validation |
| `http` | FastAPI request through TestClient and a prebuilt stub result, including thread scheduling |

Transport fixtures are checked against their expected results before timing.
Setup, model loading, tokenization, and tensor computation are outside the timed
region. HTTP numbers include test-harness overhead and exclude a real network.
These are repeated-operation timings, not production request percentiles or
concurrent-load capacity estimates. Never subtract percentiles to estimate a
component's overhead.

See the [orchestration optimization measurements](records/orchestration-performance.md)
for the validation and serialization changes, preserved behavior, and resource
measurements across these workloads.

The manual **Code benchmarks** GitHub workflow stores timing reports as artifacts
for 30 days. Shared runners are useful for diagnostics, but their timing noise is
not an absolute speed gate. Investigate variation with `pyperf stats`, `hist`,
and `check`; repeat material comparisons on a controlled machine. The workflow
does not deploy or start an external service.

## CPU and allocation profiling

Use separate runs for profiling because instrumentation changes timing:

```sh
uv run --extra bench -m benchmarks.bench_overhead --workload wide.client \
  --fast --profile .benchmarks/client.prof
uv run python -c "import pstats; pstats.Stats('.benchmarks/client.prof').strip_dirs().sort_stats('cumulative').print_stats(25)"

uv run --extra bench -m benchmarks.bench_overhead --workload wide.client \
  --tracemalloc --loops 100 -p 2 -n 3 -w 1 -o .benchmarks/client-memory.json
uv run --extra bench pyperf show .benchmarks/client-memory.json
```

`cProfile` identifies Python call costs. Profile the direct validation/client
workloads; a caller-thread profile does not include work executed on TestClient's
other threads. `tracemalloc` measures traced Python allocation peaks over the
benchmark worker workload, including harness allocations. It is neither bytes
allocated per request nor total process/GPU memory. Keep the same loop count and
setup when comparing allocation reports. Compare retained measurements before claiming an allocation improvement.

These tools use the maintained [pyperf measurement and profiling facilities](https://pyperf.readthedocs.io/en/latest/runner.html)
and Python's [allocation tracer](https://docs.python.org/3/library/tracemalloc.html),
without introducing an instrumentation framework into library code.

## Recorded measurements

[Historical code measurements](records/performance-history.md) retain the dated
validation, profiling, and diagnostics comparisons. They describe their measured
source and environment; use a fresh baseline when evaluating a current change.
