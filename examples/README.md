# Kayak examples

Start with one complete workflow. Each program keeps its inputs, API call,
result handling, and resource ownership visible, with setup commands in its
docstring. We keep examples that teach a distinct integration pattern; changing
the subject from email to groceries does not need another program.

## Start here

New to Kayak? Follow the [first-run walkthrough](../docs/getting-started.md) to
edit categories, check your input file, classify locally, and read the results.
Start by validating the included two-record file:

```sh
uv sync
uv run -m examples.classify_file examples/tickets.jsonl \
  --question examples/department.json --validate
```

Validation checks inputs with the base package and needs no model, network, or
service. The walkthrough then runs local Laya with progress and a result preview.

| Learn | Example | Requirement |
| --- | --- | --- |
| Exercise the client without a model | [Simulated integration](mock_integration.py) | Base package |
| Classify your own file with editable categories | [File classifier](classify_file.py), [question](department.json), [input](tickets.jsonl) | Local Laya, progress and result preview; validation needs only the base package |
| Load once and ask several questions | [Local decisions](local_decisions.py) | Local CLM |
| Ask Noul and Score questions | [Typed judgments](typed_judgments.py) | Kayak service |
| Handle service and connection failures | [HTTP client](http_client.py) | Kayak service |
| Call from an async application | [Async decisions](async_decisions.py) | Kayak service; sequential awaited calls |
| Select from permitted tools | [Tool routing](route_tools.py) | Kayak service; application filters permissions |
| Rerank a retrieved shortlist | [Document reranking](rerank_documents.py) | Kayak service; preserves source IDs and metadata |
| Process a file with one model load | [JSONL processing](process_jsonl.py), [sample input](tickets.jsonl) | Local CLM; preserves records and prior output |
| Retry only rejected busy requests | [Bounded retries](retry_busy.py) | Kayak service; no retry after uncertain completion |

Keep an existing Laya model or Jev client using the
[provider adapter examples](../docs/provider-adapters.md). They use the same
typed questions while preserving provider-reported results.

## Use retrieved evidence

| Learn | Example | Requirement |
| --- | --- | --- |
| Get Choice/Noul/Score answers from retrieved text | [RAG decisions](rag_decisions.py) | Kayak service; replace the retriever |
| Generate text and inspect its trace | [RAG quickstart](rag_quickstart.py) | Local Ollama with `gemma3:1b`; [setup](../docs/rag-experiments.md#try-a-small-experiment) |

Use `judge()` when you want Kayak decisions over retrieved evidence.
The standalone quickstart owns retrieval and the generator call, then records
and assesses a `RAGTrace`. Your application controls these effects.

## Measure a workflow

| Learn | Example | What it measures |
| --- | --- | --- |
| Evaluate editable Choice and ranking cases | [Use-case evaluation](evaluate_use_cases.py), [three starter datasets](evaluations/README.md) | Answer agreement and shortlist relevance; validation/simulation need no model |
| Evaluate support-routing suggestions | [Support pilot](evaluate_support.py), [starter suite](suites/support_pilot.json), [acceptance protocol](../docs/support-routing.md) | Provisional quality, review, failure, and latency gates; human confirmation remains required |
| Prepare reviewed support data | [Review commands](prepare_support.py), [review bindings](support_data.py), [offline walkthrough](../docs/support-review.md) | Prediction-free sheets, explicit adjudication, duplicate/group/split checks, and verified suites; base package only |
| Compare classifiers and add a custom metric | [Classifier benchmark](benchmark_classifiers.py), [support suite](suites/support.json) | Base-only classifier fixtures and checked reports |
| Compare Laya or Jev with a simple baseline | [Provider evaluation](evaluate_provider.py), [setup](../docs/provider-adapters.md#compare-on-the-same-cases) | Offline by default; opt-in provider calls with retained raw responses |
| Adapt an external token-level ranker | [Late interaction](evaluate_late_interaction.py) | MaxSim arithmetic on synthetic vectors, not learned-model quality |
| Find where evidence was lost | [RAG diagnosis](evaluate_rag.py) | Recorded stages and a separate context intervention |
| Evaluate an existing Python RAG | [Callable experiment](evaluate_rag_pipeline.py), [dataset](rag/dataset.json), [config](rag/config.json) | Configured execution, separate review, and quality gates |
| Evaluate async and multi-step retrieval | [Async experiment](evaluate_rag_async.py) | Bounded calls, rewritten queries, and retained repetitions |
| Evaluate an HTTP RAG service | [HTTP experiment](evaluate_rag_http.py) | Explicit client ownership; default mock, opt-in live URL |

The fixture modes run without downloading a model. The support pilot has
validation and simulation modes; provider evaluation defaults to a fixture.
These modes test
execution and evaluation mechanics. Replace them with real application outputs
and independently reviewed labels before making quality claims.

## Run the examples

Use this `main` checkout for the latest examples. Most client examples
read `KAYAK_BASE_URL` (default `http://127.0.0.1:8000`) and optional `KAYAK_API_KEY`.
Start the service once, then run a module:

```sh
uv run --extra serve kayak serve --device auto
# In another terminal:
uv run -m examples.rerank_documents
```

Provider evaluation and the support pilot were added after the 0.4.0 release
and are not included in its published source archive. Both use the existing
0.4.0 API.

Local examples use `uv run --extra local -m examples.local_decisions` and accept
`KAYAK_MODEL`, `KAYAK_DEVICE`, and `KAYAK_CACHE_DIR`. Full CLM loading downloads
about 16 GB of encoder weights plus 76 MB of heads and needs additional runtime
memory. Read the [hardware guide](../docs/validation.md) first.

For JSON exchange, [decision.json](decision.json) and [ranking.json](ranking.json)
are the canonical request samples. Validate them without loading a model:

```sh
uv run kayak validate examples/decision.json --pretty
uv run kayak validate --ranking examples/ranking.json --pretty
```

Candidate IDs belong to the application. Keep permission checks and action
execution in application code. A selected candidate or high relative share is
not calibrated correctness; `review` is an ordinary candidate, not guaranteed
abstention. Start with the [evaluation guide](evaluations/README.md) before
using decisions to drive a real workflow.
