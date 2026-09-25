# Client and server compatibility

Kayak clients and servers may be upgraded separately within the `/v1` contract.
Existing request and response bodies stay fixed, including nested fields.
The Python contracts intentionally reject unknown fields. Even an optional
metadata field added to an existing response would break installed clients.

This policy separates wire compatibility from behavior: accepting the same JSON
is insufficient if its meaning changes. That distinction follows
[Google's API compatibility guidance](https://google.aip.dev/180).
Package version `0.5.0` and HTTP API version `/v1` are separate identifiers.
The compatibility baseline is the published CLM release `0.4.0`; the earlier
retrieval-focused Kayak project is outside this contract.

## What stays stable

| Boundary | Contract |
| --- | --- |
| `POST /v1/decide` | `state` and named `questions`; each Choice has `type`, `instructions`, and `criteria` |
| Decision response | `model`, `answers`, `input_tokens`, `calibration`; each answer has `type`, `choice`, `scores`, `probabilities` |
| `GET /v1/model` and nested `model` | `id`, `revision`, `fingerprint`, `encoder`, `encoder_revision`, `input_recipe`, `device`, `dtype` |
| Field types and meaning | Required fields, defaults, accepted enum values, limits, and validation rules described in the [API reference](api.md) |
| Ordering | Submitted question/candidate IDs are preserved; score and probability mappings preserve candidate order; ties select the first candidate |
| Scores | Finite scaled cosine similarities; probabilities are their softmax, not calibrated confidence; calibration remains `none` |
| Errors | `{"error": {"code": "...", "message": "..."}}`; codes, HTTP statuses, and SDK translations in [serving](serving.md) and [the API](api.md#handle-failures) |
| Client defaults | No API key, 120-second network timeout, no redirects or automatic retries; invalid requests fail before I/O |
| Service defaults | No API key, 120-second inference timeout, diagnostics off; one active inference and no queue |
| Resource lifetime | A response timeout does not free active inference capacity; shutdown drains work and closes the model |

Error message wording can improve; branch on the error class, status, and code.
Proxy/network errors remain subject to the documented fallback behavior.
Model identity and runtime settings are values, not constants: choosing another
model revision or device need not yield identical scores. Hardware parity and
application quality need their own [validation](validation.md).

New endpoints and optional informational headers can add capabilities if
existing calls retain their behavior. Clients must tolerate absent or unknown
optional headers. Request IDs illustrate this: newer servers send them without
changing the response body; older servers may omit them.

Changing an existing body's fields, field types, enum range, interpretation, or
defaults requires a new API version and migration guidance. Keep `/v1` available
for its supported clients. Do not loosen validation or rewrite compatibility
fixtures simply to make a breaking change pass.

## Automated evidence

`Model.judge`, `Client.judge`, and `AsyncClient.judge` are Python adapters for
mixed Choice/Noul/Score questions. They compile into one existing Choice request
and decode the checked result locally. `JudgmentRequest`/`JudgmentResult` are
new Python values; neither replaces the `/v1` request or response. The current
client compatibility checks exercise Noul/Score against both server versions.

`Model.rank`, `Client.rank`, and `AsyncClient` are compatible Python extensions.
Ranking becomes one normal Choice request to `/v1/decide`; its ordered
`RankingResult` is assembled by the caller. The async client uses the same
routes, serializers, response checks, and error policy. No existing body,
capability enum, default, or model recipe changed. `RankingRequest`/`RankingResult`
are new Python/CLI values and require a new SDK; the service needs no upgrade.

The current-client pairings also exercise ranking and async calls against both
installed server versions, including an async input failure followed by recovery.
The original request/result fixtures remain unchanged.

The [compatibility runner](../scripts/check_compatibility.py) exercises all four
pairings of baseline/current clients and servers. Each version is installed
non-editably in its own environment. Workers run with Python `-I`, outside the
checkout, and the runner verifies installed source hashes. A version string
alone cannot distinguish development snapshots.

The [baseline record](../tests/fixtures/api_v1/baseline.json) pins the published
0.4.0 wheel URL, wheel SHA-256, and installed-package SHA-256. The initial 0.5.0
checkout replaces the earlier development baseline with that released artifact,
as required after publication. The wheel digest was checked against PyPI, and its
package contents match the reviewed 0.4.0 release. The matrix now compares the
released 0.4.0 package with the current checkout, independently of Git history.
Replacing this baseline again requires an explicit compatibility review;
retain the released baseline while it is supported.

Each pairing starts a real loopback HTTP server with a controlled model. It
checks the hand-authored request/result fixtures, model inspection, typed and
dictionary inputs, malformed and oversized requests, authentication, local and
remote input errors, inference failures, timeout, overload, capacity recovery,
single HTTP attempts, and model cleanup. No weights are downloaded and no
inference libraries are imported. Current-checkout tests separately challenge
unknown JSON fields and confirm tolerance for optional headers.

The [test workflow](../.github/workflows/test.yml) runs this on Linux/Python 3.13
and retains `report.json` plus per-pair logs for 14 days, including on failure.
The reusable workflow is a prerequisite of publishing. The report records
installed package hashes, Python/dependency versions, fixture hashes, checks,
and shutdown outcomes. A failed pair returns a nonzero process status.

To reproduce on Linux or macOS from this checkout or its source distribution:

```sh
compat_work=$(mktemp -d)
baseline_requirement=$(uv run --no-project python - <<'PY'
import json
from pathlib import Path

baseline = json.loads(Path("tests/fixtures/api_v1/baseline.json").read_text())
print(f"kayak[test] @ {baseline['wheel_url']}#sha256={baseline['wheel_sha256']}")
PY
)
uv venv --python 3.13 "$compat_work/baseline"
uv venv --python 3.13 "$compat_work/current"
uv pip install --python "$compat_work/baseline/bin/python" "$baseline_requirement"
uv pip install --python "$compat_work/current/bin/python" '.[test]'
uv run --no-project scripts/check_compatibility.py \
  --baseline-python "$compat_work/baseline/bin/python" \
  --current-python "$compat_work/current/bin/python" \
  --output "$compat_work/results"
```

Use fresh environments after source changes; stale installs intentionally fail.
The source archive includes the runner and fixtures; installing the baseline
needs access to its pinned PyPI artifact, without earlier Git commits or tags.
The controlled responses do not establish
full-model behavior, every input combination, or compatibility with untested
versions. Keep the ordinary pytest/Hypothesis suite and hardware checks alongside
this gate.

## Published-baseline verification — 2026-09-25

All four pairings passed between independently installed published 0.4.0 and
candidate 0.5.0 packages on Linux/Python 3.13.15. The runner executed from an
extracted 0.5.0 source archive without a `.git` directory. Each pairing retained
its request/result fixture hashes, installed package hashes, and model cleanup
result. A deliberately incorrect wheel checksum was rejected during baseline
installation. The original request and response fixture files are unchanged.

The combined 0.5.0 candidate passed Ruff, formatting, strict mypy, actionlint,
distribution checks, an isolated base-only wheel exercise, and 940 tests.
Five tests skipped: four need the separately downloaded released heads, and one
needs MPS. The existing Starlette warning remains. No model weights were
downloaded, and these checks do not establish model quality or hardware parity.

## Historical local verification — 2026-09-24

On Linux/Python 3.13.15, all four pairings passed with Pydantic 2.13.5,
HTTPX 0.28.1, FastAPI 0.141.1, and Uvicorn 0.53.0. Each pairing completed
12 grouped checks and closed its model. In a separate temporary installation,
adding a metadata field to `/v1/model` made both clients reject that server and
made the runner exit with failure; baseline-server pairings still passed.
The original package was restored and the complete matrix passed again.

The full Python 3.13 pytest/Hypothesis suite passed 227 tests with released heads
enabled; one MPS-only case skipped on Linux. The Python 3.14 environment without
inference dependencies passed 200 tests, with 28 inference cases deselected.
The existing Starlette deprecation warning remains. These are local observations;
the new hosted GitHub job has not yet been observed running.
