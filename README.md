# Kayak

**Classify, route, and rank text in Python with models you can run yourself.**
Define the allowed answers, get typed results, and evaluate them on your own data.
Kayak runs Contrastive Language Models locally or through a service you operate.

[Start here](docs/getting-started.md) · [Quickstart](#quickstart) · [Documentation](docs/README.md) · [Examples](examples/README.md)

[![PyPI](https://img.shields.io/pypi/v/kayak)](https://pypi.org/project/kayak/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB)](pyproject.toml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-16a085)](LICENSE)

Use Kayak when an application needs to select from known options: a support
department, a permitted tool, or a retrieved document. Supply text, a question,
and candidate descriptions. Kayak constructs a typed result from candidate scores;
the CLM path does not generate an answer string for you to parse.

Your application keeps control of permissions and actions. The same package can
evaluate decisions against reviewed labels and compare another classifier's
predictions, including errors and missing answers.

| Operation | Result |
| --- | --- |
| `decide` with `Choice` | A supplied candidate ID, scores, and relative shares |
| `judge` with `Choice`, `Noul`, or `Score` | Named typed answers, including binary shares and rubric averages |
| `rank` | Supplied candidates in score order, with stable ties |

The same operations are available on a local `Model`, `Client`, and `AsyncClient`.
Optional [Laya and Jev adapters](docs/provider-adapters.md) use the same named
questions while preserving each provider's result semantics.

**Current model evidence:** the recorded CLM run correctly classified 58 of 770
BANKING77 development examples (7.53%); a description word-overlap baseline
scored 35.84%. These are results from a specific development experiment, not an
official test-set benchmark or a fresh measurement of this release. Read the
[results and limitations](docs/records/clm-development-results.md) before choosing this
model for a task. Typed output does not establish a correct decision.

## Quickstart

**Kayak 0.5.0 is in development on `main`** and requires Python 3.11+.
Use [uv](https://docs.astral.sh/uv/) with the source checkout below.

The base package provides typed values, HTTP clients, and evaluation. Add the
`local` extra for local inference or `serve` for the HTTP service. See
[release validation](docs/release.md) for the tested scope and remaining model
and hardware checks.

For the runnable examples below, clone the development checkout:

```sh
git clone --branch main --depth 1 https://github.com/teilomillet/kayak.git
cd kayak
uv sync
```

The previous [0.4.0 release](https://pypi.org/project/kayak/0.4.0/) remains
installable with `uv add 'kayak==0.4.0'`; its source archive contains that release's
guides and examples. Check the current client integration without a model,
service, or accelerator:

```sh
uv run -m examples.mock_integration
```

This command uses a simulated HTTP response. To run actual inference, use a
local model or connect to an existing Kayak service as shown below.

The [first-run walkthrough](docs/getting-started.md) takes you from this check to
a readable evaluation report, then to real inference. Its first two steps need
only the base package and do not download model weights.

### Use Laya or Jev

Already using either provider? Keep your model or SDK client and wrap it with
`from kayak.adapters import Laya, Jev`. Both accept Kayak's `Choice`, `Noul`, and
`Score` questions through `judge()`.

| Provider | Execution | Setup |
| --- | --- | --- |
| Laya | A local model you load and own | [Laya quickstart](docs/provider-adapters.md#laya) |
| Jev | Hosted inference through your TypeSafe SDK client | [Jev quickstart](docs/provider-adapters.md#jev) |

The [provider comparison example](docs/provider-adapters.md#compare-on-the-same-cases)
on `main` runs offline by default, then accepts either provider on your own
labeled cases. Results retain provider semantics; adapters do not expose the
CLM-only `decide`, `rank`, or service contracts.

### Make a local decision

The default model is [CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B).
The first load downloads approximately 16 GB of encoder weights and 76 MB of
projection heads. Runtime memory exceeds the weight size; check the
[hardware guide](docs/validation.md) before loading.

Save this as `decide.py` in the checkout:

```python
import kayak
from kayak import Choice

questions = {
    "department": Choice(
        instructions="Which team should handle this request?",
        criteria={
            "billing": "Charges, invoices, and refunds",
            "technical": "Bugs and service outages",
        },
    )
}

with kayak.load(device="auto") as model:
    result = model.decide(
        state="I was charged twice for my subscription.",
        questions=questions,
    )

answer = result.answers["department"]
print(answer.choice)
print(answer.probabilities)
print(result.model.fingerprint)
```

```sh
uv run --extra local decide.py
```

Keep the model context open to reuse the loaded weights. `device="auto"` selects
available CUDA, then MPS, then CPU. Explicit device, precision, cache, and batch
settings are described in the [API reference](docs/api.md).

## Serve once, call from your application

Start a service on a machine with sufficient memory:

```sh
uv run --extra serve kayak serve --device auto
```

After loading and a readiness inference, it listens on `http://127.0.0.1:8000`.
In the quickstart program, replace the model context with a client context:

```python
with kayak.Client(base_url="http://127.0.0.1:8000") as client:
    result = client.decide(
        state="I was charged twice for my subscription.",
        questions=questions,
    )
```

The base client requires no inference libraries. In another Python project,
add this checkout with `uv add /path/to/kayak`. Async applications use
`async with kayak.AsyncClient(...)` and await the same operations.

The service admits one inference request at a time and returns 503 when busy.
Clients make one attempt per call. Configure authentication, TLS at your network
boundary, and timeouts using the [serving guide](docs/serving.md).

## Use retrieved evidence

Pass your query and retrieved passages as text `state`, then ask named questions
with `judge`. The [RAG decision example](examples/rag_decisions.py) shows
Choice, Noul, and Score together. Your application owns retrieval and any text
generation. [RAG evaluation](docs/rag-evaluation.md) checks recorded retrieval,
reranking, context, and answers; the [experiment guide](docs/rag-experiments.md)
covers configuration, repeated runs, and quality gates.

## Evaluate the result

CLM scores are scaled cosine similarities. Probabilities are relative shares
among the supplied candidates, not calibrated confidence. A candidate is selected
even when every option is unsuitable; application thresholds and fallback rules
need evaluation against reviewed labels.

Use the [evaluation API](docs/evaluation-python.md) for your data and the
[use-case evaluation map](examples/evaluations/README.md) for starter datasets
and application checks. Hardware validation, numerical conformance, and task
quality are separate evidence; their current scope is recorded in the
[release checklist](docs/release.md).

## Documentation

The [documentation index](docs/README.md) organizes guides by integration,
operation, evaluation, and contribution.

| Need | Reference |
| --- | --- |
| Types, inputs, results, and errors | [Python API](docs/api.md) · [Typed judgments](docs/typed-judgments.md) |
| Files, pipes, and JSON requests | [CLI](docs/cli.md) |
| Service operation and upgrades | [Serving](docs/serving.md) · [Diagnostics](docs/diagnostics.md) · [Compatibility](docs/compatibility.md) |
| Provider integration | [Laya and Jev](docs/provider-adapters.md) · [Migration](docs/migrating.md) |
| Implementation and model behavior | [Architecture](docs/architecture.md) · [Model contract](docs/model-contract.md) |
| Runnable integrations | [Examples](examples/README.md) |
| Coding assistant context | [Assistant guide](docs/using-with-agents.md) · [llms.txt](llms.txt) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, checks, bug reports,
and pull requests. Code changes follow the [engineering conventions](docs/engineering.md).

## License

Kayak is licensed under [Apache 2.0](LICENSE). It builds on
[Contrastive Language Models](https://github.com/Contrastive-LM/CLM) and the
Qwen3 encoder. See [NOTICE](NOTICE) for attribution. Model weights are downloaded
separately under their upstream licenses.
