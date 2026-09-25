# Documentation and application readiness audit — 2026-09-25

This audit covers the documentation reorganization, a clean installation of the
published package, the evidence behind release claims, and the first
support-ticket evaluation. The user selected support routing and authorized
provisional targets with starter data. It does not accept a production deployment.

The base checkout is `b6b665344a199f72a2301d4baec2569f2b719454`, with the
documentation moves and support-pilot changes in the working tree. The 42
production Python modules and model manifest match the published 0.4.0 source
byte-for-byte. This change adds examples, tests, and documentation; it preserves
the model recipe and `/v1` contract.

## Claim and evidence map

| Claim | Evidence and status | Limit |
| --- | --- | --- |
| 0.4.0 can be installed from PyPI | **Observed:** `uv add 'kayak==0.4.0'` in a new project; isolated wheel exercise passed | Linux/Python 3.13.15 in this run |
| Published source contains runnable guides/examples | **Observed:** downloaded source archive, `uv sync`, quickstart, JSON validation, eight model-free example invocations, and the support classifier benchmark passed | Model/service/Ollama examples require their documented dependencies; their controlled integration tests are separate |
| Release automation passed | **Observed:** [release-commit tests](https://github.com/teilomillet/kayak/actions/runs/36167059203) and [publishing](https://github.com/teilomillet/kayak/actions/runs/36167059648) completed successfully for the base commit | Linux/macOS code checks, Python 3.11/3.13 tests, distributions, and development-baseline compatibility; no full 8B model |
| Reorganized documentation remains navigable | **Supported:** local file/heading links and documented example-module targets checked in the checkout and rebuilt source archive; moved records retain their text with relative-link corrections | External URLs were unchanged by the moves; this is not a check of every external service or historical local artifact |
| Current code and connected examples work under the tested contracts | **Supported:** Ruff, formatting, strict mypy, and 931 pytest/Hypothesis tests passed | Five skips: four released-head tests without the checkpoint, one MPS-only test; existing Starlette warning retained |
| Released heads match the independent reference | Earlier CPU/Mac comparisons and unchanged fixtures are recorded in [hardware validation](../validation.md) | The real checkpoint is absent here, so those four checks were not rerun; tiny-model tests do not replace them |
| Full model executes on MPS/BF16 | Earlier [Mac smoke record](../validation.md#full-8b-smoke-test-on-the-local-mac) reports loading, repeated decisions, and local/HTTP parity | Raw Mac artifacts are absent here; not freshly reproduced or evidence for CUDA, FP16, or larger workloads |
| CLM is useful for classification | **Unknown for the support task.** The retained [BANKING77 record](clm-development-results.md) reports 58/770 correct (7.53%) versus word overlap's 276/770 (35.84%) | Different 77-intent development task, older source snapshot, raw Mac artifacts unavailable here; no support-task accuracy estimate follows |
| Full encoder matches the reference runtime or historical training encoder | **Unknown:** the [input-boundary investigation](input-recipe-investigation.md) identifies the remaining comparison; historical training revision remains an assumption | Head parity, input-text agreement, and HTTP parity do not establish encoder equivalence |
| Probabilities are calibrated or review is guaranteed abstention | No such claim is accepted; [model contract](../model-contract.md) and [support protocol](../support-routing.md) retain this boundary | Relative shares and a review candidate do not authorize application actions |
| A new user understands the product | Commands were exercised from fresh environments and their evidence scope is explicit | Independent human onboarding/comprehension has not been measured |

## Published artifacts and clean-install exercise

The [PyPI release](https://pypi.org/project/kayak/0.4.0/) was fetched directly,
and both downloaded artifacts matched PyPI's SHA-256 metadata:

| Artifact | SHA-256 |
| --- | --- |
| `kayak-0.4.0-py3-none-any.whl` | `45c8b972d0ca837c9b2eace903fec3d4acec6a87212fa0af55ddf7a58a823c88` |
| `kayak-0.4.0.tar.gz` | `a8027c479119fb37ea0a8daaaae9eefbec3e9fa831e83ba2ce1b0eb327b6b9b5` |

The independent new project installed only the base package and dependencies.
Running `python -I scripts/check_wheel.py` through its interpreter exercised the
installed package, manifest, typed clients, CLI, and evaluation interfaces.
The published source archive was extracted separately, then its documented
`uv sync` setup was run. From that archive:

```sh
uv run -m examples.mock_integration
uv run kayak validate examples/decision.json --pretty
uv run kayak validate --ranking examples/ranking.json --pretty
uv run -m examples.evaluate_use_cases examples/evaluations/*.json --validate
uv run -m examples.evaluate_use_cases examples/evaluations/*.json --simulate
uv run -m examples.benchmark_classifiers --suite examples/suites/support.json \
  --output .benchmarks/support-demo
uv run -m examples.evaluate_late_interaction
uv run -m examples.evaluate_rag
uv run -m examples.evaluate_rag_pipeline
uv run -m examples.evaluate_rag_async --max-concurrency 2 --repeats 2
uv run -m examples.evaluate_rag_http
```

All completed successfully. The support classifier example produced the
expected eight-case comparison: constant prediction 3/8, word overlap 7/8.
This demonstrates a usable evaluation workflow with constructed data; it does
not measure CLM. The default HTTP example above uses its mock endpoint.

The current source archive and wheel were also rebuilt and checked with strict
Twine and an isolated base-only wheel environment. Every current guide, example,
and example input was compared byte-for-byte with the rebuilt source archive.
The strengthened documentation check ran inside that extracted archive. The
new support evaluator ran there with the separately installed wheel and base
dependencies. These local builds retain version 0.4.0 for verification; they
were not published or substituted for PyPI's immutable artifacts.

## Support pilot result

The [pilot](../support-routing.md) defines billing, shipping, account, and manual
review as outcomes. Its 20 fictional tickets include ambiguous, unrelated, and
multiple-team requests. It uses the public evaluator and a label-independent
word-overlap baseline. Targets and their exact observed values are retained
together in `acceptance.json`.

The documented simulation completed with 5/20 matching labels, compared with
word overlap's 10/20 on the same starter suite. Quality and coverage gates failed
as expected. `evidence_scope` is `integration_only`,
`provisional_gates_passed` is false, and `deployment_accepted` is false.
The simulation's measured time is transport-fixture overhead, not model latency.

Focused tests exercise request-only inference inputs, 503/504/401 responses,
connection failures, malformed responses, failed warmups, the 2-second latency
boundary, valid numeric-gate passes, invalid suites, and preservation of existing
evidence. A failed ticket labeled `review` still counts as incorrect; its
application outcome retains no model suggestion and requires manual review.
Every successful suggestion also requires human confirmation. No retry or
ticket-system action occurs.

Before judging usefulness, collect and independently review held-out application
tickets, freeze the provisional targets, run the full model on the intended
hardware, and inspect every error alongside the lexical baseline. The
[deployment acceptance requirements](../release.md#deployment-acceptance) retain
memory, capacity, integration, and recovery checks as separate obligations.

## Reproduction and retained evidence

Use the [contribution checks](../../CONTRIBUTING.md#set-up-a-development-environment),
[release build checks](../release.md#reproduce-code-checks), and the
[pilot commands](../support-routing.md#run-the-checks). The full suite here used
`HYPOTHESIS_PROFILE=ci`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`; no model
weights were downloaded. An initial sandboxed asynchronous test run stalled and
was stopped; it is not counted as passed. The bounded run outside that sandbox
completed in 54.34 seconds with the results above.

Raw logs, JUnit output, PyPI metadata/artifacts, simulation and baseline reports,
the final local distributions, and a source-hash receipt are retained locally
under `validation/readiness-20260925/`. Like other local validation evidence,
these files are ignored by Git. The source-hash receipt identifies the tested
working tree; hashes detect changes and do not authenticate execution.
