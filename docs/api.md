# Python API

This reference describes the **Kayak 0.5.0 development checkout**. This version
is not published yet. Add the checkout to your Python project:

```sh
uv add /path/to/kayak
```

Import public types and operations from `kayak`. The base installation provides
contracts and the HTTP client. Local inference and serving use the `local` and
`serve` extras, respectively.

For the source distribution and examples, follow the
[quickstart](../README.md#quickstart). From that directory, use
`uv sync --extra local` for local loading or `uv sync --extra serve` to include
serving. Run local code
with `uv run --extra local your_script.py` and client code with
`uv run your_script.py`. For a standalone CLI, run `uv tool install .` in the
checkout; see the [CLI guide](cli.md).

Existing `/v1` JSON bodies stay fixed, including nested fields. See the
[compatibility policy and cross-version checks](compatibility.md) before
changing the wire contract or upgrading clients and servers separately.

For caller-owned Laya or Jev execution, use `Laya(agent)`, `Jev(client)`, or
`AsyncJev(client)` from `kayak.adapters`. Their `judge` methods accept the same
typed questions and return provider-reported values in `ProviderResult`, separate
from CLM results. See [provider adapters](provider-adapters.md) for the contract.

## Use retrieved evidence and evaluate RAG output

For **typed answers from retrieved evidence**, use `Client.judge` or `Model.judge`
with the query and passages rendered as `state`, plus your named questions. The
[48-line example](../examples/rag_decisions.py) uses Choice/Noul/Score together.

Your application owns retrieval and any text generation. Record its actual
stage outputs with `kayak.eval.RAGTrace`; the
[standalone generation example](../examples/rag_quickstart.py) shows this with
lexical retrieval and an explicit Ollama call.

`RAGTrace.evaluate(*, expected_answer=None, expected_sources=None, k=5)` returns
the existing `RAGAssessment`. It composes `assess_rag` with stripped, case-sensitive
answer equality and one all-required source set. Missing labels remain unknown;
other sources remain unjudged and grounding is never inferred. `k` sets the
ranking metric cutoff independently of your application's context packing.
Use `assess_rag` for graded labels, alternative evidence sets, or independent
reviews. `evaluate_rag` and `aevaluate_rag` run your callable over labeled cases;
`score_rag` assesses recorded attempts. See the
[setup and experiment guide](rag-experiments.md).

## Load and own a model

```text
kayak.load(
    model="Contrastive-LM/CLM-v0.1-8B",
    *,
    device="auto",
    dtype="auto",
    batch_size=1,
    cache_dir=None,
    local_files_only=False,
) -> Model
```

| Argument | Accepted values and behavior |
| --- | --- |
| `model` | Released model ID, or `str`/`Path` directory containing `kayak.json` and its checkpoint |
| `device` | `auto`, `cuda`, `mps`, `cpu`; auto selects available CUDA, then MPS, then CPU |
| `dtype` | `auto`, `float32`, `float16`, `bfloat16`; explicit combinations still require device support |
| `batch_size` | Positive integer; maximum number of prepared sequences per encoder forward pass |
| `cache_dir` | `str`, `Path`, or `None` for the Hugging Face default |
| `local_files_only` | Require already cached artifacts; do not download missing files |

Auto precision is FP32 on CPU, BF16 on CUDA where supported, otherwise FP16.
Projection heads always use FP32. Availability does not establish sufficient
memory or validated numerical behavior. There is no fallback to a smaller model.
Increasing `batch_size` can change both memory use and numerical results.

Loading resolves and checks the small artifacts before allocating the full
encoder. Remote custom code is disabled. A `Model` owns its loaded resources:

```python
with kayak.load(device="auto") as model:
    print(model.info)
    result = model.decide(state=state, questions=questions)
```

Use the context manager or call `model.close()`. Closing is idempotent and waits
for active local inference. Concurrent local calls are serialized by a lock.
Calling a closed model raises `ModelClosedError`. Results remain usable after
the model closes. Resource release does not promise that an accelerator's
allocator immediately returns all cached memory to the operating system.

## Ask typed Noul and Score questions

`Model`, `Client`, and `AsyncClient` expose `judge(state=..., questions=...)`
for mixed `Choice`, `Noul`, and `Score` inputs, returning `JudgmentResult`.
Use `Noul(instructions="...")` for a true/false candidate share and
`Score(instructions="...", criteria=["Low", "Medium", "High"])` for an expected
rubric index. See the [typed judgment guide](typed-judgments.md) for complete
Python examples, distribution semantics, validation, and evidence boundaries.
These adapters use the existing Choice wire contract and require no server upgrade.

## Ask Choice questions

Both `Model` and `Client` expose:

```text
decide(
    *,
    state: str,
    questions: Mapping[str, Choice | Mapping[str, object]],
) -> DecisionResult
```

Prefer typed `Choice(instructions="...", criteria={"candidate-id": "description"})`.
Its `type` is always `"choice"`. The outer mapping assigns each question an ID;
the inner mapping assigns each candidate an ID. A request needs at least one
question and each question needs at least one candidate.

The model encodes `state.strip() + "\n\n" + instructions.strip()` for each
question and each candidate's exact description. IDs label results and are not
included in model text. There is no chat template or generated answer.

`decide()` validates and snapshots caller-owned dictionaries. Typed contracts
are frozen at the field level, but nested dictionaries remain mutable; treat
them as values and create new questions when editing. Construction directly
through `Choice(...)` raises Pydantic `ValidationError` on invalid fields;
request validation at `decide()` raises Kayak `InputError`.

`AsyncClient` accepts the same arguments through `await client.decide(...)`.
Validation and the input snapshot happen when the coroutine starts running,
before its first HTTP await. Mutations made before scheduling the coroutine
are therefore visible; mutations while waiting for the response are not.

| Limit | Value |
| --- | --- |
| Questions per request | 32 |
| Candidates across all questions | 256 |
| Question or candidate ID length | 1–128 characters, nonblank |
| Each state, instruction, or candidate description | 1–65,536 characters, nonblank |
| Total request characters | 262,144, including IDs; state counted once |
| Tokens per prepared sequence | Manifest limit, at most 2,048 |
| Total tokens across prepared sequences | 16,384; repeated state/question text counts each time |
| HTTP request body | 1 MiB |

Token counts include special tokens and exclude padding. Oversized inputs raise
an error before encoder execution; no truncation is applied.

## Read results

| `DecisionResult` field | Meaning |
| --- | --- |
| `answers: dict[str, ChoiceAnswer]` | Answers keyed by the submitted question IDs |
| `model: ModelInfo` | Identity and resolved runtime configuration |
| `input_tokens: int` | Total encoded tokens, counting repeated sequences |
| `calibration` | Always `"none"` in this release |

Each `ChoiceAnswer` has `type="choice"`, `choice: str`, `scores: dict[str, float]`,
and `probabilities: dict[str, float]`. Scores are scaled cosine similarities
after learned state and action projections. They are finite and preserve the
submitted candidate IDs and order. `choice` is the first highest-scoring ID.
`probabilities` contains the softmax of those scores, summing to one within
validation tolerance. These shares are not calibrated confidence or a basis for
comparing certainty across different candidate sets.

`ModelInfo` records `id`, heads `revision`, `fingerprint`, `encoder`,
`encoder_revision`, `input_recipe`, `device`, and `dtype`. The fingerprint is a
SHA-256 of the normalized manifest, including the checkpoint hash and pinned
encoder revision. It identifies artifacts and recipe, not runtime libraries or
hardware; preserve those separately for reproducible measurements.

Contracts support Pydantic's `model_dump()` and `model_dump_json()`. JSON output
can be read back using `DecisionResult.model_validate_json(...)`.

## Use an HTTP client

```text
kayak.Client(
    *,
    base_url: str,
    api_key: str | None = None,
    timeout: float = 120.0,
    transport: httpx.BaseTransport | None = None,
)
```

Use `with kayak.Client(...) as client:` or close it explicitly. The base URL
must be HTTP(S), with a host and no embedded credentials. A path prefix is
preserved. `timeout` must be finite and positive; it sets HTTPX's network
timeouts, not a guarantee that remote computation stops after that duration.
The client validates requests before sending and checks response types,
question IDs, candidate order, and distributions. It follows no redirects and
performs no automatic retries. `transport` supports tests without networking;
see the [simulated example](../examples/mock_integration.py).

Inspect a running service before submitting a decision:

```python
with kayak.Client(base_url="http://127.0.0.1:8000") as client:
    info = client.model_info()
    print(info.id, info.device, info.dtype)
```

`model_info() -> ModelInfo` performs one authenticated `GET /v1/model`, validates
the response, and runs no decision. It uses the same timeout, error translation,
and no-retry policy as `decide()`. For a local model, its identity is already
available as `model.info` without I/O.

## Rank supplied candidates

`Model`, `Client`, and (awaited) `AsyncClient` expose:

```text
rank(
    *,
    state: str,
    instructions: str,
    candidates: Mapping[str, str],
) -> RankingResult
```

This performs one Choice decision. Instructions are required and candidate
descriptions are passed verbatim. Candidate IDs are labels; dictionary order
breaks ties. Identical descriptions under different IDs remain separate
candidates. The normal Choice character, candidate, and token limits apply,
including the internal question ID `rank` in aggregate character accounting.

`RankingResult` contains `model`, `ranked`, `input_tokens`, and `calibration`.
`ranked` is a nonempty list of `RankedCandidate(id, score, probability)`, sorted
by descending score. Each score and probability is taken from the original
Choice result; there is no second scoring operation or normalization.
`result.ranked[0].id` selects the best supplied candidate; use
`result.ranked[:k]` to inspect a prefix. Its probabilities still refer to the
full supplied set. No candidate generation, search, refusal threshold, or
tool execution is implied. Scores are not comparable across arbitrary requests.

`RankingRequest(state=..., instructions=..., candidates=...)` validates JSON
files or Python values without a model. Like `DecisionRequest`, invalid direct
construction raises Pydantic `ValidationError`, while `rank()` raises
`InputError`. `request.as_decision()` returns an owned snapshot in the frozen
`/v1` format, with one Choice named `rank`. Both request and result support
`model_dump_json` and `model_validate_json`. Nested dictionaries/lists remain
mutable; treat them as values and revalidate edited inputs.

The HTTP clients implement ranking through `/v1/decide`, so an older compatible
server needs no new endpoint. `RankingResult` is a Python/CLI view, not a changed
HTTP response. See [CLI ranking](cli.md#rank-a-supplied-set) and the
[runnable example](../examples/rerank_documents.py).

## Use async HTTP

```text
kayak.AsyncClient(
    *,
    base_url: str,
    api_key: str | None = None,
    timeout: float = 120.0,
    transport: httpx.AsyncBaseTransport | None = None,
)
```

Use `async with kayak.AsyncClient(...) as client:` or `await client.aclose()`.
Await `client.decide`, `client.rank`, and `client.model_info`; their result types,
authentication, URL handling, validation, errors, and timeout defaults match
`Client`. The client uses HTTPX's async transport and does not load inference libraries.
Cancellation propagates as the async framework's cancellation exception.

One client may be reused within its application's async lifetime. Waiting for
network I/O yields to other tasks. This does not create a server queue or permit
parallel inference in the single-model service: overlapping decisions may
return `RemoteError` with HTTP 503. After a timeout or cancellation, remote
inference may still occupy capacity until completion; shutdown still drains it.
Neither client retries nor follows redirects. Use sequential awaited decisions
when processing a stream against one service, as in the
[async example](../examples/async_decisions.py).

## Validate a request value

`DecisionRequest` is available from `kayak`. Construct it with `state` and named
`Choice` questions, or use `DecisionRequest.model_validate(...)` and
`model_validate_json(...)` for dictionary and JSON inputs. Like `Choice`, direct
construction raises Pydantic `ValidationError`. It loads no model and checks no
tokenizer-dependent limits. Submit its values with
`model.decide(state=request.state, questions=request.questions)` or the same
client call; those entry points still validate and snapshot their inputs.

See [migration examples](migrating.md) and [CLI validation](cli.md).

## Handle failures

All errors below derive from `KayakError` and have `request_id: str | None`.
It identifies the server response when a valid `X-Request-ID` header is present,
including remote input errors and malformed responses. Local failures, network
failures without a response, and missing/malformed headers leave it `None`.
See [service diagnostics](diagnostics.md) to correlate errors with server logs.

| Error | Boundary and response |
| --- | --- |
| `InputError` | Invalid request or load options; correct the input; also a `ValueError` |
| `ModelLoadError` | Missing dependencies, unavailable device, or invalid/unloadable artifacts |
| `InferenceError` | Local model execution or non-finite scores failed |
| `ModelClosedError` | Attempt to use a closed resident model |
| `TransportError` | Network failure or invalid decision response; remote completion may be unknown |
| `RemoteError` | Non-success HTTP response; inspect `status_code` and `code` |

A server 422 with code `invalid_request` becomes `InputError`. Other HTTP
failures become `RemoteError`; malformed error bodies use code `http_error`.
See the [HTTP status table](serving.md) for service-specific failures.

## Supply a compatible model bundle

Start with the [canonical manifest](../kayak/models/clm-v0.1-8b.json). Put it at
`my-model/kayak.json` alongside the named checkpoint. Set the actual checkpoint
SHA-256, model identity, hidden size, and exact encoder Hub revision used in
training. The checkpoint filename must stay within the bundle directory.
Offline operation also needs that encoder revision in the Hugging Face cache.

Pass `kayak.load("./my-model")`. The loader currently supports only the
`clm-qwen3` family and `clm-choice-v1` recipe. A model still in training may need
a different loader once its contract is known; swapping a path cannot make an
incompatible architecture work. See the [model contract](model-contract.md).
