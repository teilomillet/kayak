# Coming from Jev or Laya

To keep Laya or Jev as your inference provider, use the
[Python adapters](provider-adapters.md). The guide below covers moving execution
onto Kayak's CLM runtime instead.

Kayak keeps the familiar interaction: provide state, ask named questions, and
consume structured answers. Use `decide` for **Choice** and `judge` for
**Choice, Noul, and Score** through the pinned CLM transformations.
Start by migrating one representative Choice workflow and evaluating its
results before moving application decisions onto the new model.

The mappings below follow the [TypeSafe Python SDK for Jev](https://docs.typesafe.ai/sdk/python)
and [Laya's documented Python interface](https://github.com/NandhaKishorM/laya),
inspected on 2026-09-24. Those implementations were read as interface references;
their models were not run in these checks.

## The first call

Version 0.5.0 is in development. Use the
[source checkout](../README.md#quickstart), run `uv sync`, and run your client
code with `uv run your_script.py`. In another Python project, add that checkout
with `uv add /path/to/kayak`.

```python
import kayak
from kayak import Choice

questions = {
    "team": Choice(
        instructions="Which team should handle this request?",
        criteria={
            "billing": "Charges, invoices, and refunds",
            "technical": "Bugs and service outages",
        },
    )
}

with kayak.Client(base_url="http://127.0.0.1:8000") as client:
    result = client.decide(state="I was charged twice.", questions=questions)

print(result.answers["team"].choice)
```

Start the service from the checkout in a separate terminal with
`uv run --extra serve kayak serve --device auto`. For in-process use, run
your script with `uv run --extra local your_script.py`,
replace the client context with `with kayak.load(device="auto") as model:`, and call
`model.decide(state=..., questions=...)`. The question and result types stay the
same. Loading is explicit and the context manager owns the resources.

## Map concepts, then check their meaning

| Familiar operation | Kayak operation |
| --- | --- |
| Jev `TypeSafeClient().system_one(state=..., questions=...)` | `Client(base_url=...).decide(state=..., questions=...)` |
| Jev `response.choices[question_id]` | `result.answers[question_id]` |
| Laya `load(...).predict(state, questions)` | `load(...).decide(state=state, questions=questions)` |
| Laya `result["answers"][question_id]["choice"]` | `result.answers[question_id].choice` |
| Laya dictionary results | `result.model_dump()` or `result.model_dump_json()` |
| Named dictionary Choice specifications | Accepted directly, or construct `kayak.Choice` values |

The Jev names are from its [SDK quickstart](https://docs.typesafe.ai/sdk/python);
the Laya names are from its [Python API reference](https://nandhakishorm.github.io/laya/reference/agent/).
Kayak keeps one operation named `decide`; there are no parallel `predict` or
`system_one` aliases to learn or maintain.

For retrieved knowledge, keep the same pattern: your retriever supplies context,
then Kayak evaluates `state` with your named questions. The
[RAG decision example](../examples/rag_decisions.py) shows Choice, Noul, and Score
in one call. See the [RAG guide](rag-experiments.md) for the upstream examples and
the distinction between typed answers and an external generator's prose.

| Difference to resolve | What to do in Kayak |
| --- | --- |
| Structured state | Serialize it explicitly to text, for example `json.dumps(ticket, ensure_ascii=False)`; preserve that rendering in evaluation fixtures |
| Missing or `None` candidate descriptions | Write an explicit description for every candidate; IDs are labels and are not model input |
| Noul or Score questions | Use [`judge` with typed questions](typed-judgments.md); reevaluate task quality and thresholds. `decide` and raw `/v1` bodies remain Choice-only |
| Confidence thresholds | Reevaluate your policy on CLM outputs; softmax shares are uncalibrated and conditional on the supplied candidate set |
| Async calls | Use `AsyncClient` and await `decide`, `judge`, `rank`, or `model_info`; the service still admits one inference at a time |
| Model routing or framework hooks | Use one explicitly selected model; keep orchestration in the application |
| Existing checkpoints | Supply a Kayak-compatible bundle; Jev/Laya checkpoints do not become compatible by changing their filename |
| Laya's schema-driven `Agent.decide(...)` | Define named Choice questions explicitly; Kayak does not convert arbitrary JSON/Pydantic schemas into model tasks |

Jev documents Choice, Noul, Score, structured state, optional descriptions, and
separate sync/async clients in its [SDK guide](https://docs.typesafe.ai/sdk/python).
Laya documents additional judgment types and routing in its
[quickstart](https://github.com/NandhaKishorM/laya), and its separate schema-driven
operation in the [Agent reference](https://nandhakishorm.github.io/laya/reference/agent/).
These are capability
differences, not just spelling changes. Kayak does not claim wire compatibility,
matching accuracy, or matching calibration with either system.

For supplied action descriptions or best-of-N candidates, use
`model.rank(state=..., instructions=..., candidates={id: description})` or the
same client method. `result.ranked[0].id` is the best supplied candidate, and
`result.ranked[:k]` is a shortlist. Selecting an action and executing it remain
application decisions. The [comparison and verification report](records/sdk-validation.md)
records the pinned CLM recipes, newer upstream differences, and the Noul/Score
quality evaluation still required for application decisions. The Python adapters
are available without claiming matching accuracy or calibration with Jev/Laya.

## Validate before loading a model

Use the public request type to check existing dictionary specifications:

```python
from kayak import DecisionRequest

request = DecisionRequest.model_validate({
    "state": "I was charged twice.",
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "Charges and refunds", "technical": "Bugs and outages"},
        },
    },
})
```

`DecisionRequest` validates types, character limits, and candidate counts without
inference dependencies. It raises Pydantic `ValidationError`, just like `Choice`.
Actual token limits are checked later by the loaded tokenizer. A valid request
schema does not establish model quality or available device memory.

For structured state, render it explicitly with `json.dumps(ticket,
ensure_ascii=False)` before passing it as `state`. Keep that rendering fixed
while comparing providers. The [canonical request](../examples/decision.json)
can be validated without a model:

```sh
uv run kayak validate examples/decision.json --pretty
```

Keep input IDs and ordering unchanged during the first comparison. Record the
model identity, original inputs, scores, and selected IDs from both systems.
Use independently labeled examples and review disagreements and near ties.
Compare task behavior, not raw score scales across different models.

See [the CLI guide](cli.md) for pipes and errors and [the API](api.md) for the
complete current contract.
