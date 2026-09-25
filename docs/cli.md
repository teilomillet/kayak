# Use Kayak from the terminal

The CLI uses the same request, result, and error contracts as Python. The base
package provides `validate`, `info`, `decide`, and `rank`; only `serve` needs `kayak[serve]`.
Run `kayak` or `kayak --help` to discover commands. Every subcommand has `--help`.

Version 0.5.0 is in development. From the
[source checkout](../README.md#quickstart), install the command as a uv tool:

```sh
uv tool install .
kayak --help
```

Use `uv tool install '.[serve]'` instead to include the server dependencies.
If the tool directory is not on your PATH, run `uv tool update-shell` and open a
new terminal. Tool installs provide the CLI in an isolated environment; use
`uv add /path/to/kayak` in a Python project to import this checkout from your own code.

From a source checkout, use `uv run kayak` or `uv run --extra serve kayak serve`.
To install that checkout as the standalone tool, use `uv tool install .` or
`uv tool install '.[serve]'`.

`kayak eval` prepares BANKING77 and benchmarks local models or an existing HTTP
service. See [dataset evaluation](evaluation.md) for its commands, saved reports,
and evaluation-specific exit codes. HTTP evaluation needs only the base package;
local evaluation needs the `local` extra.
`kayak eval report PATH_TO_RUN` verifies saved artifacts and prints Markdown
without loading a model. See the [mock audit](records/benchmark-audit.md) to try it.

## One request, from validation to execution

From the source directory:

```sh
# No network, model, or inference libraries needed.
kayak validate examples/decision.json --pretty

# Start one resident model in a separate terminal.
kayak serve --device auto

# Inspect the model already owned by that service.
kayak info --pretty

# Send exactly the same request.
kayak decide examples/decision.json --pretty
```

`validate` checks JSON structure, supported question types, character limits,
and candidate counts. It emits a validated request, including default fields.
The loaded model still needs to check token limits before inference. Validation
does not download weights or assert language understanding.

`info` retrieves `/v1/model`: artifact identity, encoder, device, precision, and
input recipe. The GET itself performs no inference. The service's startup
readiness check remains part of loading a model.

`decide` submits one request to `/v1/decide` using the Python client. It uses the
running service and does not allocate another local model. It preserves
candidate IDs and returns the full `DecisionResult` JSON.

## Files and pipes

Use `-` as the filename to read UTF-8 JSON from stdin:

```sh
kayak validate - < examples/decision.json
kayak decide - < examples/decision.json > result.json
```

Request input is one JSON object, not JSONL. Use the
[JSONL example](../examples/process_jsonl.py) for sequential file processing.
The CLI reads at most 1 MiB plus one byte to detect overflow; oversized input is
rejected before parsing. The same byte limit is used by the HTTP service.

Successful `validate`, `info`, and `decide` commands write JSON only to stdout.
`--pretty` indents it; without that flag it is one compact JSON line. Errors go
to stderr and produce no partial JSON result. Shell `>` replaces the target
file, so use a fresh path when keeping multiple results.

## Rank a supplied set

Ranking input contains `state`, `instructions`, and an ordered `candidates`
mapping of IDs to explicit descriptions. Preview it before using a model:

```sh
kayak validate --ranking examples/ranking.json --pretty
kayak rank examples/ranking.json --pretty
kayak rank - < examples/ranking.json > ranking.json
```

`rank` accepts the same connection, authentication, timeout, and `--pretty`
options as `decide`. It submits one Choice through `/v1/decide`, then emits
`RankingResult` JSON: `model`, `ranked`, `input_tokens`, and `calibration`.
Every `ranked` entry contains `id`, `score`, and `probability`, in descending
score order with original order breaking ties. Shares refer to the whole set;
no tool is executed. The same byte/character/token bounds and exit/error policy
apply. Invalid ranking input fails before opening a client. Plain `validate`
continues to accept the original decision schema.

## Connection and authentication

The default address is `http://127.0.0.1:8000`. Set it explicitly for another
service. Keys are read from an environment variable, keeping them out of the
command's argument list:

```sh
export KAYAK_API_KEY='your-service-key'
kayak info --base-url https://your-service.example --pretty
kayak decide request.json --base-url https://your-service.example --timeout 60
```

Use `--api-key-env NAME` when your environment uses a different variable.
`--timeout` sets the client's network timeout, defaulting to 120 seconds. It
does not stop remote computation. A 503 reports unavailable or occupied capacity;
a 504 may leave computation running. Commands do not automatically retry or
follow redirects. See [service behavior](serving.md) for lifecycle details.

## Exit codes and recovery

| Code | Meaning | Next step |
| --- | --- | --- |
| 0 | Help/version or requested operation succeeded | Consume stdout |
| 1 | File access, network, server, or invalid response failure | Inspect stderr; check path, connection, or service state |
| 2 | Invalid arguments or request | Correct the indicated field or option |

Expected failures have plain messages without Python tracebacks. Unsupported
Noul/Score questions, missing descriptions, and structured state receive
guidance about the current Choice contract. The CLI does not silently invent
descriptions or serialize arbitrary objects for the model.
Noul/Score are available through the Python [`judge()` adapters](typed-judgments.md),
which compile to the existing Choice wire request.

Remote failures include `[request_id=...]` when the service supplies a valid
ID. Start the service with `--diagnostics json` to find that request in
structured logs; use `--diagnostics off` (the default) to disable those events.
See [service diagnostics](diagnostics.md) for event fields and privacy boundaries.

## Interface choices

For application-level RAG evaluation, `kayak eval rag` exports JSON schemas and
pipeline inputs, validates datasets/configuration, prepares external reviews,
scores recorded JSONL, and renders checked reports. These commands perform no
inference or network calls. See [RAG experiments](rag-experiments.md#use-json-from-any-language)
for the full exchange and its exit codes: a completed run without configured
gates is not a quality pass. Python callers use `evaluate_rag` or `aevaluate_rag`
to execute their own pipeline and reviewer callbacks.

The [Command Line Interface Guidelines](https://clig.dev/) inform discoverable
help, visible defaults, separate output/error streams, and useful exit statuses.
Python's [argparse](https://docs.python.org/3.11/library/argparse.html) supplies
those primitives without adding a CLI framework. The Python API remains the
canonical contract; the terminal calls it rather than implementing another
decision path. See [migration from Jev/Laya](migrating.md) for the SDK mapping.
