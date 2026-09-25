# Release preparation records

These records describe the dated development snapshots below. Test counts,
local environments, and unverified configurations belong to those runs; they
are not a current CI status. Use the [release checklist](../release.md) and the
candidate commit's workflow results for the release decision.

## Local preparation evidence

Observed on Linux during preparation on 2026-09-24:

| Check | Result |
| --- | --- |
| Complete pytest/Hypothesis suite, released heads enabled | Passed on Python 3.11 and 3.13 |
| Locked Pixi environment without inference dependencies | Code tests passed on Python 3.14 |
| Ruff lint/format and strict mypy | Passed; examples and validation tools included |
| Example execution | Local/offline/ranking/JSONL with actual tiny Qwen3; simulated HTTP success and failure |
| Hardware-report tool | Tiny-model HTTP exercise, repeat summaries, unavailable-memory case, and retained failure report |
| Code benchmark smoke | All 20 workloads completed; this is not a latency comparison |
| Packaging | Source archive and wheel built; strict Twine and isolated base-only wheel checks passed |
| Workflow structure | Actionlint passed; publishing depends on reusable code checks |

The subsequent SDK/CLI pass added public `DecisionRequest`, `Client.model_info`,
request-file/stdin commands, and migration guidance for Jev/Laya users. The full
Python 3.13 suite passed 135 tests. Python 3.11 and the inference-free environment
also passed their checks, including a separate-process CLI round trip through a
real loopback service. The existing model loading and projection files remain
identical to the `4c26ed8` model-test baseline.

The subsequent [API evaluation](api-evaluation.md) exercised local/HTTP parity
and input-error recovery with typed and dictionary inputs. It also compared two
isolated client snapshots against fixed and perturbed responses: unchanged
responses worked, while additive metadata was rejected. Forward compatibility
and first-time-user comprehension are not established by these checks.

The existing Starlette test-client deprecation warning remains visible. The
earlier result-assembly speed experiment stays in benchmarks; production
inference and scoring code were not changed for this release preparation.

Local macOS verification of the SDK/CLI update (`8fbaad4`) and follow-up fixes
uses Python 3.13.5, PyTorch 2.14.0, and an Apple M4 Pro. It covers actual tiny
Qwen3 inference on CPU and MPS, repeated MPS decisions, and live HTTP parity.
The released-head validator now separates a stable recorded reference check
from same-platform FP32 parity; see [the numerical evidence](../validation.md#portable-head-validation).
Indexed `mps:0` devices retain their memory counters in validation reports.
Invalid non-ASCII client API keys produce a plain input error without echoing
the credential. Regression tests cover both fixes and the numerical checker.
All 144 tests passed with the released checkpoint enabled; Ruff lint/format and
strict mypy passed. All 20 benchmark smoke workloads, distribution checks, and
the isolated base-only wheel exercise passed as well.

The subsequent full-model run passed on the same 24 GB Mac after closing unused
apps, with about 13.1 GiB available before loading. The actual pinned 8B encoder
ran on MPS in BF16, using only cached artifacts. Both illustrative cases repeated
exactly across three runs; overflow rejection and live HTTP parity passed. Loading
took 10.38 seconds. Warm medians were 0.663 and 0.450 seconds; these small samples
are not performance guarantees. MPS driver allocation was about 15.2 GiB and
sampled system swap peaked at 3.57 GiB. See [hardware validation](../validation.md)
for the exact configuration and measurement boundaries.

Hosted GitHub jobs, independent full-encoder parity, application quality, CUDA,
and default FP16 full-model behavior remain unverified. The local BF16 smoke
test does not establish those claims.
