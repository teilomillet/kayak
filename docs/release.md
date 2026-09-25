# Release readiness

Version 0.5.0 is in development on `main` and has not been published or tagged.
Its initial commit preserves the typed SDK, `/v1` contract, and model recipe.
Version 0.4.0 was the first CLM-focused release, replacing the previous retrieval
implementation. The [changelog](../CHANGELOG.md) describes that change; 0.4.0
was not a drop-in upgrade of the earlier retrieval package.

## 0.4.0 publication scope

This release makes the typed SDK, HTTP service, provider adapters, and evaluation
APIs available as an installable package. The reviewed candidate passed the
Linux/macOS code checks, Python 3.11/3.13 tests, client/server compatibility matrix,
distribution checks, and isolated wheel checks. The publishing workflow repeats
these checks before uploading its artifacts.

Version 0.4.0 is [published on PyPI](https://pypi.org/project/kayak/0.4.0/).
The [release-commit checks](https://github.com/teilomillet/kayak/actions/runs/36167059203)
and [publishing run](https://github.com/teilomillet/kayak/actions/runs/36167059648)
completed successfully for `b6b665344a199f72a2301d4baec2569f2b719454`.

Model evidence remains limited to the configurations and observations recorded
in [hardware validation](validation.md). Independent full-encoder equivalence,
representative application quality, CUDA behavior, and default FP16 full-model
behavior remain open. Publication accepts this limited scope; it does not mark
those checks as passed. The deployment acceptance requirements below still apply.

Historical [BANKING77 development evidence](records/clm-development-results.md)
also records weak task accuracy: 58/770 correct (7.53%) for the unchanged model,
versus 276/770 (35.84%) for description word overlap. The raw Mac experiment
artifacts are not present in this Linux checkout; that record is retained
evidence, not a fresh reproduction. It does not measure support-ticket routing.
The next application acceptance exercise is the [support pilot](support-routing.md),
with provisional targets and human review for every suggestion.
The [readiness audit](records/readiness-audit.md) separates current installation
and code evidence from historical measurements and open model claims.

## Before publishing a release

- Review the final candidate, including documentation and example changes, as
  one installable checkout. Resolve stale commands, links, and API descriptions.
- Require successful Linux/macOS code checks, supported Python checks, and the
  client/server compatibility matrix for that commit. The test workflow runs
  on `main` and release-branch pushes and pull requests; local results supplement it.
- Build the wheel and source distribution from that commit. Exercise the wheel
  in an isolated environment and confirm the source archive includes the guides,
  examples, tests, and validation tools.
- Review the remaining model evidence below and record the configurations and
  workload results accepted for release. Unknown results remain open.
- Verify that installation instructions, package links, and the changelog match
  the release stage.

## Deployment acceptance

For each supported deployment, retain the following before directing application
traffic to it:

| Boundary | Required evidence |
| --- | --- |
| Build and model identity | Reviewed commit, wheel, pinned model artifacts, precision, and device |
| Hardware | Successful startup, representative input sizes, memory headroom, and offline restart |
| Application behavior | Reviewed labels, accepted quality criteria, failure cases, and explicit application fallback rules |
| Capacity | Measured latency and load for the target workload; the service has one active inference slot and no queue |
| Integration | Readiness, authentication, error handling, overload, timeout, and shutdown exercised by the calling application |
| Recovery | Previous build/configuration retained and a restart or rollback procedure exercised |

The [serving guide](serving.md) defines the operational contract. Deployment
configuration and application acceptance criteria belong to the operator.

## Reproduce code checks

Use the appropriate PyTorch build for your machine. No model download is needed
for these checks unless you explicitly enable the separately downloaded heads.

```sh
uv sync --extra local --extra test --extra bench
uv run --no-sync ruff check kayak tests scripts benchmarks examples stubs
uv run --no-sync ruff format --check kayak tests scripts benchmarks examples stubs
uv run --no-sync mypy
HYPOTHESIS_PROFILE=ci uv run --no-sync pytest -q
uv run --extra bench -m benchmarks.bench_overhead --debug-single-value --loops 1
uv build
uv run --no-sync twine check --strict dist/*
```

To include the actual released-head comparison, set
`KAYAK_TEST_HEADS=/path/to/CLM_v0.1-8B.pt` when running pytest. Without that file,
the released-head test skips. The small Qwen3 tests remain real inference, but
their random weights do not establish language understanding.

Check the wheel separately from the checkout and optional dependencies:

```sh
uv venv /tmp/kayak-wheel-check
uv pip install --python /tmp/kayak-wheel-check/bin/python dist/kayak-0.5.0-py3-none-any.whl
/tmp/kayak-wheel-check/bin/python -I scripts/check_wheel.py
```

Use a fresh environment path if that directory already exists. This verifies
the installed version, packaged manifest and typing marker, client round trip,
CLI, and operation without inference or serving dependencies. The `-I` flag is
required: importing the source checkout would not test wheel contents.

## Automated release boundary

The [test workflow](../.github/workflows/test.yml) checks the surrounding code
on Linux and macOS, and actual tiny-model inference on Linux with Python 3.11
and 3.13. Both full jobs run strict typing, benchmark smoke checks, distribution
checks, and an isolated wheel exercise. These jobs do not download the 8B model.

The compatibility job also checks all four pairings of independently installed
baseline/current clients and servers over loopback HTTP. It verifies package
hashes and retains reports and logs on failure. See the [compatibility policy](compatibility.md)
for the pinned development baseline, scope, and local reproduction commands.

The [publishing workflow](../.github/workflows/publish-pypi.yml) calls those checks
from the same commit before building its release artifact. It then checks that
exact wheel in a fresh base-only environment. A failed check prevents the build
and publish jobs from proceeding. A manual run produces checked distributions;
only a matching version tag enters the existing PyPI publishing environment.

The release decision must record the accepted evidence and remaining limits,
as the 0.4.0 scope above does. The workflow cannot infer model readiness from
passing code tests. Hardware and application acceptance remain required before
deployment or claims about those configurations and workloads.

## Remaining full-model evidence

Follow [hardware validation](validation.md) on an adequately sized machine:

```sh
uv run --extra serve scripts/validate_model.py --device auto --repeats 21 \
  --output validation/auto.json
```

The command is hardware-neutral. An explicit `--device cuda`, `mps`, or `cpu`
selects a backend without changing the procedure. Retain evidence for every
configuration you intend to claim as tested.

- Successful loading, repeated full-model inference, overflow rejection,
  local/HTTP parity, and a second run using the offline cache.
- Actual load time, warm timings, memory observations, and operating-system
  memory/swap behavior on the target machine.
- Independent encoder/reference comparison with a justified score tolerance;
  the historical encoder revision used during training remains an assumption.
- Representative labeled task evaluation with accepted quality and performance
  criteria. The two built-in smoke cases do not measure application accuracy.

Keep failures and variations with the reports. `not_run` means unknown. A Mac
report does not establish CUDA parity, and a CUDA report does not establish MPS
parity. No calibrated-confidence claim is part of this release.

## Validation records

[Preparation records](records/release-validation.md) retain the earlier Linux and macOS
checks. The [hardware guide](validation.md) describes the full-model MPS/BF16
smoke run and its limits. [Provider validation](records/provider-validation.md) records
separate Laya/Jev checks. None substitutes for checks of the final candidate.

## Package and publish after validation

Transfer the source archive from `dist/` to a validation machine if working from
an unpublished checkout. Extract it and follow the commands above. Examples,
docs, tests, and the reference fixture travel with the archive.

For a subsequent release, record the accepted model evidence, update the
changelog date and README release status, commit the reviewed candidate, and
create its matching version tag.
Publish the accepted artifact through the existing trusted PyPI identity.
