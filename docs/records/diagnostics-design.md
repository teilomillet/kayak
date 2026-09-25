# Debugging a deployed Kayak service

Research and design, 2026-09-24. The first increment—request IDs, SDK error
context, and opt-in structured logs—is implemented; see the
[configuration guide](../diagnostics.md). Metrics, tracing, and exporters below
remain proposals, not current capabilities.
This addresses operational feedback: identifying a failed request, finding its
execution history, and knowing how to recover. Model-quality feedback requires
separately reviewed examples and labels.

## Outcome and current gaps

A developer receiving an error should be able to find the corresponding server
events, identify the model/software revision, distinguish rejection from work
that continued after a timeout, and reproduce the failure with an appropriate
fixture. Request text and secrets should remain outside routine telemetry.

At `4331ebe`, [the server](../../kayak/server.py) has bounded admission, readiness
inference, request limits, and shutdown draining. Its inference error logs
contain generic messages without exception context or request correlation.
[The client](../../kayak/client.py) preserves status and error code for remote
errors, but exposes no server request ID. A timed-out inference retains capacity
until completion. HTTP status alone cannot describe its eventual outcome.

The [API evaluation](api-evaluation.md) also found that additive result metadata
breaks existing clients. Diagnostic transport metadata must respect that boundary.

## Research decisions

| Source | Application to Kayak |
| --- | --- |
| [Google SRE: monitoring distributed systems](https://sre.google/sre-book/monitoring-distributed-systems/) | Measure traffic, latency, errors, and saturation. Keep successful and failed request latency distinguishable. Alerts should identify actionable user impact; dashboards support diagnosis. |
| [OpenTelemetry context propagation](https://opentelemetry.io/docs/concepts/context-propagation/) | Correlate client/server traces and logs using standard W3C context. Treat incoming context as untrusted; request identity is not authorization. |
| [OpenTelemetry Python status](https://opentelemetry.io/docs/languages/python/) | Traces and metrics are stable; logs are marked development when inspected. Use optional tracing/metrics integration and standard Python logging for application events. |
| [Prometheus instrumentation guidance](https://prometheus.io/docs/practices/instrumentation/) | Use bounded labels and distribution measurements. Request IDs, input IDs, exception messages, and raw URLs do not belong in metric labels. Measure instrumentation overhead. |
| [RFC 9457](https://datatracker.ietf.org/doc/html/rfc9457) | Keep machine-readable failure identity separate from human guidance and internal debugging detail. Retain Kayak's existing error envelope; adopting these principles does not make it an RFC 9457 implementation. |
| [AWS: timeouts, retries, and backoff](https://d1.awsstatic.com/builderslibrary/pdfs/timeouts-retries-and-backoff-with-jitter.pdf) | A timeout does not establish that remote work stopped. Retain explicit retry ownership; any future retry facility needs bounds and jitter, with completion uncertainty preserved. |
| [OpenTelemetry error handling](https://opentelemetry.io/docs/specs/otel/error-handling/) | Export failures must not turn successful decisions into failures. Bound buffering and export time; expose dropped telemetry and exporter trouble. |
| [Jane Street: getting from tested to battle-tested](https://blog.janestreet.com/getting-from-tested-to-battle-tested/) | Test histories involving failures, disconnects, restarts, and version skew. Turn an incident into a reproducible regression case. The lesson does not require buying a testing platform. |

## First increment: follow one failed request

Generate an opaque request ID at the service boundary and return it in a
`X-Request-ID` response header on responses produced by Kayak. Include it in
structured events and expose it on SDK exceptions caused by remote responses,
including malformed responses and server-side input errors. Preserve current
exception classes, error codes, and decision JSON. A connection failure or
upstream rejection may have no Kayak ID; absence must remain explicit.

Keep request IDs and trace IDs distinct: a distributed trace can contain several
requests. Request IDs work even when tracing is disabled or a trace is sampled
out. A future tracing integration should use existing OpenTelemetry propagators;
the first increment does not parse or propagate tracing headers.

Use named Python loggers; application code owns handlers. `kayak serve` can
configure JSON output explicitly. Record event name, timestamp, request ID,
outcome/code, elapsed time, software version, and model fingerprint. Link to
existing `ModelInfo` for artifact/device details instead of defining another
model identity. Log request shape as counts only when it helps diagnosis.

For internal failures, retain exception types and useful stack locations in
operator logs. Exclude local variable values, arbitrary exception messages,
request state, candidate text/IDs, credentials, and captured headers. Review
framework instrumentation too; custom application logs are not the only source
of collected data. This follows [OpenTelemetry's data-minimization guidance](https://opentelemetry.io/docs/security/handling-sensitive-data/).

Create a troubleshooting entry for each existing error code: its meaning,
whether work was admitted, what remains unknown, and the next useful check.
For example, `overloaded` means that request was not admitted; `timeout` means
inference may still be running. Retrying either is an application decision.

## Represent execution history accurately

The following is an illustrative event sequence, not observed timing:

```text
request r1: admitted; inference starts
request r1: HTTP response ends with 504; inference remains active
request r2: rejected as overloaded; no inference starts for r2
request r1: inference finishes; capacity becomes available
```

Measure HTTP duration and inference duration separately. The inference span and
active-inference count remain open until computation ends, even if the HTTP span
has ended. Attach the same request identity to both. A timeout followed by a late
success is one failed HTTP request and one successful computation; neither
measurement should erase the other. A disconnect is not an observed HTTP status.

Record one terminal event for each started operation. A process killed during
inference may have no terminal event; document that limit instead of inventing
a successful or cancelled completion. Startup, readiness failure, draining,
and model closure need explicit events too.

## Second increment: operational measurements

Use optional OpenTelemetry traces and metrics with explicit application-owned
configuration. The SDK/exporter owns collection and aggregation; do not maintain
a parallel set of counters for a second exporter. OTLP can feed a deployer's
collector, which can serve their chosen backend. Standard JSON logs remain
useful without a collector. Importing Kayak must not configure global logging,
start exporter threads, or send telemetry.

The serving configuration should keep slow log/export I/O off the inference
path with bounded buffering. Saturation must produce bounded diagnostic loss
instead of holding model capacity indefinitely. Applications installing their
own logging handlers retain responsibility for those handlers' behavior.

Start with these measurements:

| Measurement | Question answered |
| --- | --- |
| Request count and duration histogram by fixed route, status, and bounded outcome | Are calls succeeding, and how slow are successful and failed calls? |
| Inference duration and terminal outcome | How long does admitted computation actually take? |
| Active inference | Is the single inference slot still occupied after clients stop waiting? |
| Startup/warmup duration and outcome | Did the service load its model and become ready? |
| Dropped/export-failed telemetry | Are gaps in the diagnostics caused by instrumentation? |

Use the stable [HTTP duration convention](https://opentelemetry.io/docs/specs/semconv/http/http-metrics/)
where it applies. Keep Kayak-specific inference measurements explicitly named.
Histograms must cover the configured inference timeout and slower completions.
Overload and timeout rates come from bounded outcomes, not duplicate counters.
Request/trace IDs belong in logs and traces, never metric labels. Service and
model identity belong in deployment/resource metadata. Reuse the deployment's
host/device monitoring for resource pressure when available.

Choose reliability targets after measuring representative traffic and hardware.
Count authenticated decision attempts rejected for overload, and timeouts, as
failures. Track observed invalid and unauthenticated traffic separately;
overload can reject a request before its body is validated. A
synthetic decision check can test the client-visible path, but consumes model
capacity; frequent health probes should remain lightweight.

## Evidence required before shipping

Keep event construction and field filtering as typed, pure functions. Clock
reads, request context, logging, and export stay at explicit effect boundaries.
The inference slot remains owned by the existing server lifecycle code.

1. Exercise success, invalid input, authentication failure, overload, inference
   failure, and malformed responses; match SDK request IDs to server events.
2. With event-controlled tests, time out or disconnect a caller, reject a second
   call, complete the first computation, and admit a third. Check the full event
   history, timings, and capacity counts, including concurrent request isolation.
3. Generate malformed trace/request headers and payloads containing sentinel
   secrets with Hypothesis; inspect exported events as well as HTTP responses.
4. Disconnect or stall the exporter, fill its buffer, and trigger a failing log
   sink in the shipped configuration. Decisions must retain their original results and failure behavior;
   diagnostic loss must remain bounded and visible where an independent sink works.
5. Run actual process startup/shutdown tests and an installed older client against
   the instrumented server. Headers must not alter the current result contract.
6. Compare existing overhead benchmarks with instrumentation disabled, enabled,
   and export unavailable; retain latency variation and memory observations.

A model decision can be wrong despite a successful HTTP request. Link reviewed
application examples to request/model identity in an application-owned evaluation
dataset; routine operational telemetry is not a training-data collection system.

## First increment evidence

The implemented increment uses the standard library with no new dependencies.
Its 28 focused tests cover on/off behavior, error/header privacy, generated
malformed headers, timeout/cancellation histories, safe exception context,
blocked/broken sinks, CLI error output, and real process startup/shutdown.
The process test exposed Uvicorn closing a writer configured before its logging
setup; the CLI now owns the writer inside the service lifespan and drains it
before Uvicorn restores signal behavior.

Observed locally on Linux: all 170 tests passed on Python 3.13 with the tiny
Qwen3 and cached released heads; the 28 diagnostics tests passed on Python 3.11;
150 inference-free tests passed on Python 3.14. Ruff and strict typing passed.
These observations do not establish full-model or accelerator behavior.

Isolated clients built from `4c26ed8` and `8fbaad4` both accepted decisions and
retained `RemoteError(401, unauthorized)` against a diagnostics-enabled real
loopback service with a controlled model. Four requests had distinct IDs in
server events. These are development snapshots, not published release clients.
The local report is `.benchmarks/diagnostics-compatibility.json`.

[Overhead measurements](performance-history.md#optional-diagnostics-overhead--2026-09-24)
record timings with logging off, enabled, and unavailable, plus separate Python
allocation observations. A blocked sink has deterministic tests rather than a
claim about production latency under arbitrary I/O failures. Metrics, tracing,
and exporters require their own implementation and validation before shipping.
