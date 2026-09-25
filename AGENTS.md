# Working with Kayak

For integration work, start with the [documentation index](docs/README.md) and
[example catalog](examples/README.md). Read the closest example's source and use
its documented setup and run commands. These guides describe the 0.5.0 development
checkout. Run `uv sync` from `main`; 0.5.0 has not been published. The previous
0.4.0 release remains available on PyPI with its own source distribution.

| Task | Starting point |
| --- | --- |
| Use local or HTTP typed decisions | [Python API](docs/api.md) · [Typed judgments](docs/typed-judgments.md) |
| Supply retrieved evidence to Kayak | [RAG decisions](examples/rag_decisions.py): render query and source text as `state`, then call `judge` |
| Evaluate an application or use case | [Evaluation map](examples/evaluations/README.md) · [Python evaluation](docs/evaluation-python.md) |
| Evaluate external rankers, late interaction, or RAG stages | [Ranking and RAG evaluation](docs/rag-evaluation.md) |
| Configure and evaluate an existing RAG application | [RAG experiments](docs/rag-experiments.md): sync/async callables, HTTP, JSON exchange, and quality gates |
| Try retrieval and optional text generation | [Standalone example](examples/rag_quickstart.py): application code records a `RAGTrace` and checks explicit references |
| Use an existing Laya or Jev object | [Provider adapters](docs/provider-adapters.md) |

[llms.txt](llms.txt) indexes the contracts and guides.
[Assistant guidance](docs/using-with-agents.md) explains the integration checks.
Keep expected labels out of inference inputs. CLM scores are uncalibrated;
application permissions, thresholds, and action execution remain explicit.
Controlled responses establish integration behavior, not model quality.

For code changes, follow [CONTRIBUTING.md](CONTRIBUTING.md) and the
[engineering conventions](docs/engineering.md). Preserve the
[compatibility contract](docs/compatibility.md) and model input recipe. Keep
examples readable from inputs through the call to result use, with setup and run
commands in their docstrings. Update the catalog when recipes change.

Use the existing Ruff, strict mypy, and pytest checks. Do not download model
weights merely to validate requests or test HTTP integration. Use controlled
fixtures for those checks and the separate hardware and task evaluations for
model claims. Release requirements are in [release readiness](docs/release.md).
