# Engineering conventions

Prefer code that a reader can follow from inputs through validation and effects
to the result. Use Python's idioms, with Go's emphasis on explicit dependencies,
simple interfaces, and clear resource ownership. Use a functional core where it
makes domain rules easier to understand and test.

The [architecture](architecture.md) describes the current components. The
[model contract](model-contract.md) and [compatibility policy](compatibility.md)
define behavior that changes must preserve.

## Structure and interfaces

- Group code by responsibility. Contracts and pure transformations belong with
  the domain they describe; network, filesystem, model loading, and scheduling
  effects have explicit owners.
- Keep dependencies acyclic. Qualify calls to shared domain operations, such as
  `decisions.request_from(...)`, so their ownership is visible. Public imports
  retain the convenient `kayak.Choice` and `kayak.load` interface.
- Define small protocols beside the code that consumes them. An abstraction
  needs a current caller or a boundary that benefits from substitution. Prefer
  composition over a backend hierarchy or speculative extension framework.
- Choose domain names over generic `utils` or `helpers` modules. A module
  docstring states its responsibility; comments explain invariants and decisions
  that the code does not make apparent.
- Review long functions for mixed responsibilities. Keep a coherent numerical
  operation or resource lifetime together when splitting it would obscure its
  invariants.
- Every first-party `.py` and `.pyi` file must contain at most 1,000 physical
  lines, including comments and blank lines. This applies to repository-root
  files and `kayak`, `tests`, `scripts`, `benchmarks`, `examples`, and `stubs`.
  Split by responsibility instead of compressing code to meet the limit.

## Types, validation, and data ownership

Annotate parameters and return values. Strict mypy and Ruff reject explicit
`Any`, untyped functions, and bare generic containers. Keep narrow development
stubs aligned with the external calls they describe.

Validate unknown data at the boundary. A cast records an expected type; it does
not validate a value. Reject invalid input with runtime checks and the documented
error type, including when Python runs with `-O`.

Preserve caller data. Frozen contract objects can contain mutable dictionaries,
so typed inputs still need revalidation and independent snapshots. Pure
transformations return owned values. Local mutation is appropriate when it keeps
control flow simple and does not change caller-owned inputs.

Keep the normal path visible. Guard invalid cases early, avoid redundant `else`
after a return, and reserve exception handling for operations that can fail.
Use broad catches only at documented isolation or translation boundaries.

## Effects and lifetime

Keep acquisition and cleanup together with context managers or `try/finally`.
The loader allocates resources, the resident model serializes execution, and its
owner closes it. HTTP clients own connections; provider adapters borrow objects
whose caller retains configuration and cleanup.

Expose the material concurrency boundaries: admission, cancellation, retained
work after timeout, and shutdown. Use event-controlled tests for histories that
would otherwise depend on timing.

Applications own retrieval, generation, action execution, and backend settings.
Evaluation code keeps recorded observations separate from independent judgments.
Missing labels, execution failures, and measured incorrect answers are distinct
outcomes. Trace hashes bind reviews to values; they do not authenticate data or
establish causality.

## Verification

For each change, exercise the requested behavior and relevant failure boundaries.
Use public contracts and independent expectations to choose assertions. Preserve
accepted validation rules and compatibility fixtures. A failing test is evidence
to investigate, not a reason to weaken the contract.

Use the existing tools:

```sh
uv sync --extra local --extra test --extra bench
uv run --no-sync ruff check kayak tests scripts benchmarks examples stubs
uv run --no-sync ruff format --check kayak tests scripts benchmarks examples stubs
uv run --no-sync mypy
HYPOTHESIS_PROFILE=ci uv run --no-sync pytest -q
```

Use the appropriate PyTorch build for the machine. The
[contribution guide](../CONTRIBUTING.md) includes a smaller environment for work
that does not require inference libraries.

`tests/test_structure.py` checks the source-file limit and explicit imports,
including deferred and type-checking imports. Dynamic imports and third-party
internals are outside that graph check. A separate fresh-process test verifies
that importing Kayak leaves optional inference and server dependencies unloaded.

Ruff checks formatting, imports, annotations, common bugs, and supported Python
syntax. Mypy runs strictly with the Pydantic plugin across all first-party
source. Hypothesis adds generated boundary cases to fixed contract examples.
Keep skipped checks and untested configurations visible in review notes.

## Performance and dependencies

Investigate a measured operation before introducing optimization complexity.
Retain the workload, source, environment, raw samples, and variation. Compare
correct implementations under the same conditions; smoke timings only show
that a benchmark executes. See [development and profiling](development.md).

Prefer existing dependencies and tools. Additional abstractions, libraries,
caches, concurrency, and numerical changes need a present requirement or
measurement. Model quality and hardware support require the separate evidence
in the [release checklist](release.md).

## Design references

These references inform the conventions above; Python's contracts and runtime
remain authoritative for this implementation.

- [Go at Google](https://go.dev/talks/2012/splash.article): explicit dependencies, composition, and consistent tools.
- [Go code review guidance](https://go.dev/wiki/CodeReviewComments): meaningful names, consumer-owned interfaces, and direct control flow.
- [Jane Street: Effective ML](https://blog.janestreet.com/effective-ml/): useful types, readable interfaces, and pragmatic purity.
- [Google code review guidance](https://google.github.io/eng-practices/review/reviewer/looking-for.html): concrete requirements, complexity, tests, and concurrency.
- [Python's functional programming guide](https://docs.python.org/3/howto/functional.html): isolating effects and testing transformations.
