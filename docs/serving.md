# Local HTTP service

The service owns one resident CLM. The Python client uses the same Choice and
result types as local inference. Version 0.5.0 is in development;
start from the [source checkout](../README.md#quickstart):

```sh
uv run --extra serve kayak serve --device auto
```

For the Python client, run `uv add /path/to/kayak` in your application project.
For a standalone server command, run `uv tool install '.[serve]'` in the
checkout, then `kayak serve --device auto`.

The default model is CLM-v0.1-8B and the default address is `127.0.0.1:8000`.
Device selection is shared with `kayak.load()`. Use `--device cuda`, `--device
mps`, or `--device cpu` to select explicitly. `kayak serve --help` lists all
options; `kayak --version` prints the installed package version.

## Readiness and requests

```sh
curl --fail http://127.0.0.1:8000/health
kayak info --pretty
# Run these examples from a source checkout:
kayak decide examples/decision.json --pretty
uv run -m examples.http_client
```

The server loads the model and completes a readiness inference before accepting
requests. Startup can take time on the first download. A connection failure
during startup is not a ready response.

| Endpoint | Successful response | Authentication |
| --- | --- | --- |
| `GET /health` | `{"ready": true}` | None |
| `GET /v1/model` | `ModelInfo` JSON | Bearer token when configured |
| `POST /v1/decide` | `DecisionResult` JSON | Bearer token when configured |

A request body has `state` and `questions`. Each question has
`type: "choice"`, `instructions`, and `criteria`. The Python client constructs
this JSON and validates the response. For a direct request:

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/decide \
  -H 'Content-Type: application/json' \
  -d '{"state":"Charged twice", "questions":{"team":{"type":"choice",
       "instructions":"Which team?", "criteria":{"billing":"Invoices and refunds",
       "technical":"Bugs and outages"}}}}'
```

## Admission, timeout, and shutdown

Only one inference request is active at a time. There is no inference queue;
another request receives 503. The CLI runs one worker, avoiding duplicate model
allocations. Body size is capped at 1 MiB; decision limits are in the
[API reference](api.md).

The server's inference timeout is 120 seconds, configurable with `--timeout`.
A 504 response does not stop an executing accelerator operation. Timed-out or
disconnected requests retain capacity until their computation finishes.
Shutdown drains active inference before closing the model. The client performs
no automatic retries; retry policy belongs to the calling application.

| HTTP status | Error code | Meaning |
| --- | --- | --- |
| 401 | `unauthorized` | Missing or invalid configured bearer token |
| 413 | `request_too_large` | Body exceeds the byte limit |
| 422 | `invalid_request` | Invalid contract or input/token limits exceeded |
| 500 | `inference_failed` | Model execution failed |
| 503 | `not_ready` | Model is not available |
| 503 | `overloaded` | Model is busy; no inference was queued |
| 504 | `timeout` | Response timed out; computation may still be running |

Error bodies use `{"error": {"code": "...", "message": "..."}}`. A web server
or intermediary can produce other statuses or formats; the client reports
these with its fallback `http_error` code.

Existing `/v1` bodies, error codes, and semantics stay stable. New capabilities
use separate endpoints or optional headers; see [compatibility](compatibility.md)
for the policy and independently installed client/server checks.

## Configure authentication and caching

For access outside loopback, the CLI requires a key from the environment:

```sh
export KAYAK_API_KEY='replace-with-your-local-secret'
kayak serve --host 0.0.0.0 --device auto
```

Pass that value as `api_key` to `Client`; the HTTP example reads the same
environment variable. A bearer token alone does not encrypt traffic. Remote
access needs your existing TLS and access-control boundary. Kayak does not log
request contents. `/health` discloses readiness only.

Use `--cache-dir /path/to/cache` to choose artifact storage. After a successful
load with that cache, `--local-files-only` requires cached artifacts. To serve
your own compatible bundle, pass its directory as the positional model argument.

Use `kayak serve --diagnostics json` for optional request and lifecycle events;
`--diagnostics off` is the default. Response headers carry request IDs, also
available on SDK exceptions. See [debugging a service](diagnostics.md) for
configuration, timeout histories, and logging limits. Code measurements live
in [development](development.md).
