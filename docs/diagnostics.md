# Follow a request through the service

Turn on structured events with one flag:

```sh
uv run --extra serve kayak serve --diagnostics json
```

Turn them off with `--diagnostics off`, which is the default. This adds no
dependencies, collector, network exporter, or configuration file. Changing the
flag takes effect on the next service start.

Kayak events go to stderr, one JSON object per line. Uvicorn's process messages
still use its normal format. In JSON mode, its ordinary access log is disabled
to avoid duplicating requests and recording raw URLs. With diagnostics off,
Uvicorn retains its normal logging and Kayak creates no diagnostics thread or
duration measurements.

## Find the request

Responses handled by Kayak carry `X-Request-ID`, even with diagnostics off.
The server generates a fresh ID; caller-supplied IDs and tracing headers are
not used. Decision JSON and error bodies retain their existing contracts.

SDK failures caused by a response expose `request_id`. For example:

```python
from kayak import Client, KayakError

with Client(base_url="http://127.0.0.1:8000") as client:
    try:
        info = client.model_info()
        print(info.id)
    except KayakError as error:
        print(type(error).__name__, error.request_id)
        raise
```

The CLI prints `[request_id=...]` with remote error messages. Search your
service logs for that value. No ID is available when local validation fails,
no response arrives, or a server/intermediary omits the header. The SDK rejects
malformed IDs and limits accepted values to 128 ASCII letters, digits, dots,
underscores, or hyphens. The ID is a diagnostic reference, not authentication.

## Read the history

Each ordinary event includes `event`, UTC `timestamp`, and `kayak_version`.
Applicable events also include `request_id`, `duration_seconds`, `status_code`,
`code`, `route`, and `model_fingerprint`. Fingerprints refer to the existing
[`ModelInfo`](api.md#read-results), not a second model identity.

| Event | What it establishes |
| --- | --- |
| `startup.started` → `startup.ready` or `startup.failed` | Loading and readiness inference succeeded or failed; duration covers both |
| `inference.started` → `inference.completed` | Admitted work began and finished, with its own result status |
| `http.completed` | The application handed the final response body to the HTTP server; this does not prove the client received it |
| `http.cancelled`, `http.disconnected`, `http.failed` | The HTTP task ended without a complete application response; status is absent if none was sent |
| `shutdown.started` → `shutdown.completed` or `shutdown.failed` | Admission stopped, active inference drained, and model cleanup finished or failed |
| `diagnostics.dropped` | The log buffer or sink lost records; `records` is the number lost |

A timed-out request can have this history under one ID:

```text
inference.started
http.completed       status_code=504 code=timeout
inference.completed  status_code=200
```

The later event describes computation, not a second HTTP response. While it
runs, another request receives `503 overloaded` with a different ID. Cancelling
an HTTP task can produce a similar history. A network disconnect is recorded
only when the ASGI server delivers it to the application; socket closure alone
does not necessarily cancel inference or the HTTP task.

| Observation | Next step |
| --- | --- |
| `401 unauthorized` | Check the caller's key and the service's configured key |
| `413 request_too_large` or `422 invalid_request` | Correct input size/shape; local `kayak validate` checks the JSON contract, while token limits require the loaded model |
| `503 overloaded` | Reduce concurrent submissions; no work was queued for this request |
| `504 timeout` | Look for later inference completion before deciding whether to retry |
| `500 inference_failed` | Find `inference.completed` by ID, inspect exception types and frame locations, and reproduce with the recorded model revision |
| No `startup.ready` | Inspect `startup.failed` and process output; a connection refusal alone does not identify the loading failure |
| `diagnostics.dropped` | Check stderr collection/backpressure; the observed history may be incomplete |

## What is recorded and bounded

Events exclude request/response content, candidate IDs, credentials, raw URLs,
headers, exception messages, and local variables. Errors retain up to four
exception types with the last eight stack locations each: filename, function,
and line number. Known routes use fixed names; other paths become `other`.
Keep reproducing inputs in application-owned fixtures with appropriate access.

The CLI buffers at most 1,024 records plus the record being written. If stderr
stalls, new records are dropped instead of making inference wait for the sink.
Loss is reported when that sink works again. Shutdown gives the writer up to
one second to drain. A permanently blocked sink or forced process exit can lose
records, including its loss report; these logs are not a durable audit trail.

Responses rejected before Kayak runs, including reverse-proxy or Uvicorn
admission failures, have no Kayak event or ID. An unexpected framework failure
can produce an outer 500 response without the header; `http.failed` records its
exception context when it passes through Kayak's middleware. Uvicorn, model
dependencies, and application code own their own logs and privacy settings.

## Embed in an existing Python service

`create_app(loader, diagnostics=True)` emits JSON messages on the standard
`kayak.diagnostics` logger. The application configures its level and handlers;
Kayak changes no root logging configuration. For example, after configuring
your application's logging:

```python
import logging

import kayak
from kayak.server import create_app

logging.getLogger("kayak.diagnostics").setLevel(logging.INFO)
app = create_app(lambda: kayak.load(device="auto"), diagnostics=True)
```

Use a message-only formatter if the destination expects JSON lines. Handler
buffering, output, and shutdown belong to the embedding application. A custom
handler can block its caller; the bounded writer described above belongs to
`kayak serve`. Set `diagnostics=False` to stop Kayak event generation.

This increment implements request correlation and logs. Metrics, distributed
traces, automatic retries, and feedback storage are not added. The
[research and design](records/diagnostics-design.md) describes the choices and possible
next increments; [development](development.md) covers local code measurements.
