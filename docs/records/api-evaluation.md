# API evaluation

Observed on Linux on 2026-09-24 against the API at `8fbaad4`. This evaluation
checks the documented Python workflows and challenges response compatibility.
It does not establish whether a new user finds the API intuitive.

## User workflows

The [workflow tests](../../tests/test_api_workflows.py) use the public API, an actual
small Qwen3 encoder with randomly initialized weights, and a real authenticated
loopback HTTP service. Only artifact resolution is redirected to local fixture
files; model execution and HTTP requests are real. Both typed `Choice` objects
and dictionary specifications go through the same scenario.

| Task | Observed result |
| --- | --- |
| Load once and make a decision | Passed; typed answers retain question and candidate IDs |
| Switch from `Model.decide` to `Client.decide` | Passed; identical inputs produce equal complete results in this CPU fixture |
| Inspect the remote model | Passed; remote metadata matches the local model |
| Correct a blank input and reuse the model/client | Passed; both reject it with `InputError` and accept corrected input |
| Correct a tokenizer-limit failure | Passed locally and over HTTP; server capacity remains available for the corrected call |
| Close resources and retain results | Passed; results still serialize, input values are preserved, and the closed model rejects further calls |

These observations establish API behavior within this setup. Random weights
do not establish the quality of the selected decision. They also do not
establish full 8B loading, accelerator behavior, or language understanding.

## Response compatibility

The [fixed request](../../tests/fixtures/api_v1/request.json) and
[fixed response](../../tests/fixtures/api_v1/response.json) represent the current
wire contract. The response is hand-authored: tied zero scores, probabilities
of 0.5, and the first candidate as the winner. Production result assembly did
not generate this expected output.

Two separately installed base-only clients were exercised under Python 3.13.15,
HTTPX 0.28.1, and Pydantic 2.13.5. Their installed client/contract source files
were checked against `4c26ed8` and `8fbaad4`. Both are development snapshots
labelled 0.4.0, not two published releases. Controlled HTTP responses isolated
the parsing and compatibility boundary; this comparison did not run two
independently deployed server versions.

| Response supplied to `Client.decide` | `4c26ed8` | `8fbaad4` |
| --- | --- | --- |
| Unchanged fixture | Accepted | Accepted |
| Added result metadata field | Rejected | Rejected |
| Added model metadata field | Rejected | Rejected |
| Added answer metadata field | Rejected | Rejected |
| Missing required token count | Rejected | Rejected |
| Renamed question ID | Rejected | Rejected |
| Wrong winner for tied scores | Rejected | Rejected |
| Probabilities inconsistent with scores | Rejected | Rejected |
| Unknown input recipe | Rejected | Rejected |
| Candidate order changed, with otherwise valid scores and winner | Rejected | Rejected |

Each case made exactly one HTTP attempt. Rejections became `TransportError`.
The current client's `model_info()` also rejects an added metadata field.

**Additive response fields break these clients.** The subsequent
[compatibility policy](../compatibility.md) therefore freezes existing `/v1` bodies
and permits new endpoints and optional informational headers. It preserves
strict validation. Changes to existing body fields require a new API version.

The [compatibility tests](../../tests/test_api_compatibility.py) retain the baseline
and additive-field checks for the current checkout. Their rejection assertions
enforce that policy; a green run does not mean added body fields are compatible.
The original two-snapshot comparison above was a separate local experiment.
The maintained runner now exercises independently installed clients and servers
in all four baseline/current pairings; its scope and evidence are documented
in the compatibility policy.

## Repeat the maintained checks

From the source directory:

```sh
uv run --extra local --extra test pytest -q tests/test_api_workflows.py tests/test_api_compatibility.py
```

Seven checks passed on Python 3.11 and 3.13. These require no model download;
the workflow tests need permission to bind a loopback socket. The five contract
checks also passed in the inference-free Python 3.14 environment using
`-m 'not inference'`; the two inference cases were deliberately deselected there.

The complete Python 3.13 pytest/Hypothesis suite passed 142 tests with released
heads enabled and the CI Hypothesis profile (300 generated examples per property).
Ruff lint/format and strict mypy also passed. The existing Starlette test-client
deprecation warning remains. Follow [release checks](../release.md) to repeat the
full suite. This evaluation added tests and documentation; production code did
not change.

## What remains unknown

A fresh Jev/Laya user's ability to complete the three tasks without assistance
has not been observed. Neither SDK was executed during this evaluation. A
usability session should record where users look for methods, misunderstand
inputs or results, or need explanations. Automated execution cannot supply
that evidence. Full-model testing remains in [hardware validation](../validation.md).
