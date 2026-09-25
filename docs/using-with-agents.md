# Use Kayak with a coding assistant

Point your assistant at [llms.txt](../llms.txt) or the root [AGENTS.md](../AGENTS.md).
Both link the public API, runnable examples, operating guides, and evaluation
paths. These files work as ordinary repository context; no assistant plugin is
required.

## Integration workflow

1. Find the closest task in the [example catalog](../examples/README.md) and read
   its source, setup, and run command. Identify the input, questions, result, and
   owner of each effect.
2. Use the API from this checkout. Version 0.5.0 is not yet published. Start with
   `uv sync` and `uv run -m examples.mock_integration` to check the client path
   without model weights or a service.
3. Adapt the example with explicit inputs and resource ownership. Use
   `with kayak.load(...)`, `with kayak.Client(...)`, or
   `async with kayak.AsyncClient(...)` as appropriate.
4. Exercise the integration with controlled responses, including a changed
   answer and a service error. Keep model quality separate from these checks.
5. Use the [evaluation map](../examples/evaluations/README.md) to find labeled
   cases and application checks. Evaluate reviewed data on the intended model
   and [hardware](validation.md) before choosing application thresholds.

Start a service with `uv run --extra serve kayak serve --device auto` on suitable
hardware. Client examples document `KAYAK_BASE_URL` and `KAYAK_API_KEY` where
supported. Reuse the service across calls; the client does not load local weights.

## Prompts to adapt

**Retrieved evidence and typed answers**

> Read examples/rag_decisions.py and docs/typed-judgments.md. Render my query and
> retrieved source IDs/text as state, then ask named questions with judge. Return
> the typed answers and distributions. Keep references out of inference inputs,
> handle empty retrieval, and retain the exact state for evaluation.

**Evaluate an existing RAG application**

> Read docs/rag-experiments.md. Adapt the closest callable, async, or HTTP example
> to my application. Apply backend settings explicitly, record observed stages,
> and evaluate independent references. Show a configuration change that affects
> actual output and a missing judgment that cannot pass a quality gate.

**Select an agent action**

> Find the task in examples/README.md for selecting a tool or action. Filter the
> permitted candidates in application code, preserve their IDs, and return a
> suggestion. Keep permissions and execution in my application. Include ordinary,
> ambiguous, and unsupported cases in the evaluation.

**Evaluate a new use case**

> Read examples/evaluations/README.md. Find the closest dataset, define the
> expected application outcome, and add reviewed cases before running inference.
> Show the integration check and the command for evaluating my service. Report
> failures and unknown outcomes alongside successful cases.

For generated text, [the standalone recipe](../examples/rag_quickstart.py) owns
retrieval and the generator call, then records and assesses a `RAGTrace`.

## Contracts to preserve

| Boundary | Contract |
| --- | --- |
| CLM output | Selects or scores supplied candidates; text generation belongs to the application |
| Retrieval | The application supplies passages; ranking orders supplied candidates |
| Noul and Score | Python `judge` compiles these to Choice; `/v1/decide` accepts Choice wire questions |
| Probability | Shares are uncalibrated and relative to the candidate set |
| Fallback | An `unknown` or `review` candidate can also be selected incorrectly |
| Concurrency | The service owns one active inference request and returns 503 when busy |
| Timeout | A timed-out response can leave model work running and capacity occupied |
| Evaluation | Configuration must affect execution; missing observations and judgments remain unknown |

Use the [API](api.md), [serving guide](serving.md), and
[compatibility policy](compatibility.md) as the detailed references. Repository
tests check local links, example commands, and controlled execution. They do not
establish external search indexing or automatic discovery by every coding assistant.
