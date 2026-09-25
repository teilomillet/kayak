# Contributing to Kayak

Kayak 0.5.0 is developed on `main`. Base changes and pull requests on that
branch. Start with the
[architecture](docs/architecture.md) and [engineering conventions](docs/engineering.md).

## Repository layout

| Path | Contents |
| --- | --- |
| [kayak/](kayak/) | Installable package: contracts, clients, runtime, service, adapters, and evaluation |
| [docs/](docs/README.md) | Current guides and contracts; dated evidence lives in [docs/records/](docs/records/README.md) |
| [examples/](examples/README.md) | Runnable integration recipes and sample inputs |
| [tests/](tests/) | Regression checks and independent contract fixtures |
| [benchmarks/](benchmarks/) | Measurement tools, fixed workloads, and retained reports |
| [scripts/](scripts/) | Package, compatibility, provider, and model validation commands |
| [stubs/](stubs/) | Narrow typing definitions for development dependencies |
| [.github/](.github/) | CI, release workflows, and the pull request template |

Project and tool configuration lives in [pyproject.toml](pyproject.toml); the
optional Pixi environments use [pixi.lock](pixi.lock). Generated build output,
environments, caches, and local measurements are ignored by Git. Local
`.benchmarks/` and `validation/` results can contain retained evidence; preserve
them when cleaning disposable build output and caches.

## Report a problem

Use the repository's [issue tracker](https://github.com/teilomillet/kayak/issues).
Include the package version or commit, Python version, operating system, a
minimal reproduction, and the expected and observed behavior. For inference
issues, include the model fingerprint, device, precision, and relevant input
sizes. For HTTP errors, include the status, error code, and request ID when
available. Use synthetic inputs when the original data is private.

For a proposed feature, describe the application task and the limitation of the
current API. A concrete calling example helps establish the smallest useful change.

## Set up a development environment

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/):

```sh
git clone --branch main https://github.com/teilomillet/kayak.git
cd kayak
uv sync --extra test
```

The base test environment checks contracts and HTTP behavior without model
weights or an accelerator:

```sh
uv run --no-sync ruff check kayak tests scripts benchmarks examples stubs
uv run --no-sync ruff format --check kayak tests scripts benchmarks examples stubs
uv run --no-sync pytest -q -m 'not inference'
```

For strict typing and the complete local suite, install the inference and
benchmark dependencies. Choose the appropriate PyTorch build for your machine.
These tests use tiny model fixtures; they do not download the full 8B encoder.

```sh
uv sync --extra local --extra test --extra bench
uv run --no-sync mypy
HYPOTHESIS_PROFILE=ci uv run --no-sync pytest -q
```

Released-head checks additionally require `KAYAK_TEST_HEADS` to point to the
checkpoint described in [hardware validation](docs/validation.md). Record skips
and unavailable hardware in your validation notes.

## Prepare a pull request

Keep each change focused on one observable outcome. Preserve existing public
contracts, request validation, caller data, and resource ownership. Add or update
tests for changed behavior and meaningful failure cases; avoid tests tied only
to a private implementation detail.

Update affected documentation and examples when the calling contract changes.
Run the checks relevant to the change. The description should explain the
problem, resulting behavior, validation performed, and material limitations.
Performance claims need measurements with the source, environment, workload,
and variation retained; see [development and profiling](docs/development.md).

Changes to existing `/v1` bodies require the review described in the
[compatibility policy](docs/compatibility.md). Model recipe changes additionally
require independent reference evidence. Packaging and release checks are in the
[release checklist](docs/release.md).
