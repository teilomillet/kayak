# Kayak documentation

Kayak provides typed AI decisions through a Python library and a service you
operate. Start with the [quickstart](../README.md#quickstart), then choose a guide
for your integration. These guides describe the 0.5.0 development checkout on
`main`. This version is not published yet; builds from this checkout include
the documentation and runnable examples in the source distribution.

## Build an integration

| Task | Guide |
| --- | --- |
| Classify your own file and inspect the results | [First-run walkthrough](getting-started.md) |
| Find a runnable example | [Example catalog](../examples/README.md) |
| Use local or HTTP inference | [Python API](api.md) |
| Ask Choice, Noul, or Score questions | [Typed judgments](typed-judgments.md) |
| Rank supplied documents or actions | [Ranking API](api.md#rank-supplied-candidates) |
| Feed retrieved evidence to Kayak | [RAG integration](rag-experiments.md) |
| Use a Laya or Jev provider | [Provider adapters](provider-adapters.md) |
| Migrate an existing integration | [Migration guide](migrating.md) |
| Work with JSON files and shell pipelines | [Command-line interface](cli.md) |
| Give context to a coding assistant | [Assistant guide](using-with-agents.md) · [llms.txt](../llms.txt) |

## Operate a service

| Task | Guide |
| --- | --- |
| Configure serving, authentication, and timeouts | [Serving](serving.md) |
| Investigate a failed request | [Diagnostics](diagnostics.md) |
| Upgrade clients and servers | [Compatibility contract](compatibility.md) |
| Select and validate hardware | [Hardware validation](validation.md) |
| Understand model artifacts and input preparation | [Model contract](model-contract.md) |
| Prepare an encoder diagnostic without inference | [Frozen probes and saved-output comparison](encoder-comparison.md) |
| Review release requirements | [Release checklist](release.md) |

## Evaluate behavior

| Task | Guide |
| --- | --- |
| Define success for an application | [Use-case evaluation map](../examples/evaluations/README.md) |
| Evaluate public customer-support intents and rejection | [HINT3 v2 workflow](hint3-evaluation.md) |
| Evaluate support-ticket routing | [Support pilot and provisional acceptance criteria](support-routing.md) |
| Prepare and review support tickets | [Review workflow and offline walkthrough](support-review.md) |
| Evaluate labeled data in Python | [Evaluation API](evaluation-python.md) |
| Run a classification benchmark | [Classification reports](classification-benchmarks.md) · [BANKING77](evaluation.md) |
| Compare Laya and Jev on your cases | [Provider comparison](provider-adapters.md#compare-on-the-same-cases) (example on `main`) |
| Evaluate retrieval, reranking, context, and answers | [RAG evaluation](rag-evaluation.md) |
| Run configured or repeated RAG experiments | [RAG experiments](rag-experiments.md) |
| Measure actual model latency and memory | [Inference evaluation](inference-evals.md) |
| Reproduce benchmark artifacts | [Benchmark protocol](benchmark-protocol.md) |
| Check metric arithmetic and benchmark comparability | [Independent reference checks](evaluation-reference-checks.md) |

Integration tests exercise contracts and failure handling. Model quality and
performance require labeled workloads and measurements on the target hardware.
The guides identify which kind of evidence each command produces.

## Understand and contribute

- [Architecture](architecture.md): current components, request flow, and resource ownership.
- [Contributing](../CONTRIBUTING.md): setup, checks, bug reports, and pull requests.
- [Engineering conventions](engineering.md): code structure, types, and validation rules.
- [Development and profiling](development.md): test environments and performance measurement.
- [Changelog](../CHANGELOG.md): changes prepared for each release.

## Validation and research records

The [record index](records/README.md) groups dated model, provider, integration,
release, and performance evidence. These records describe specific experiments
and their limits; they supplement the current API and operating guides.
