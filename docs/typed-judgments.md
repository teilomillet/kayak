# Noul and Score in Python

For Laya or Jev execution with these question types, see
[provider adapters](provider-adapters.md). This page describes the native CLM path.

`Model.judge`, `Client.judge`, and `AsyncClient.judge` accept named `Choice`,
`Noul`, and `Score` questions together. They reuse the resident model or send one
ordinary Choice request to an existing Kayak server, then decode the result.
There is no separate model to load, generated answer, or automatic action.

```python
from kayak import Client, Noul, NoulAnswer, Score, ScoreAnswer

with Client(base_url="http://127.0.0.1:8000") as client:
    result = client.judge(
        state="I was charged twice, but my account still works.",
        questions={
            "duplicate": Noul(instructions="The customer reports a duplicate charge."),
            "impact": Score(
                instructions="Assess the reported impact.",
                criteria=["No disruption", "One operation affected", "Account unusable"],
            ),
        },
    )

duplicate = result.answers["duplicate"]
impact = result.answers["impact"]
assert isinstance(duplicate, NoulAnswer) and isinstance(impact, ScoreAnswer)
print(duplicate.noul, duplicate.probabilities)
print(impact.score, impact.legend, impact.probabilities)
```

Use `model.judge(...)` with an already loaded `Model`, or
`await client.judge(...)` inside an `AsyncClient` context. The
[standalone example](../examples/typed_judgments.py) includes setup and execution
commands. These operations have the same ownership, errors, cancellation, and
no-retry behavior as `decide`. The caller owns the model or client lifetime.

## What each value means

To ask these questions about retrieved knowledge, put the query and original
source text in `state`; see the [48-line RAG decision example](../examples/rag_decisions.py).
Your retriever and Kayak client/model keep their own lifetimes. The response is
the same native `JudgmentResult`, with no RAG-specific result wrapper.

| Question | Typed answer | Interpretation |
| --- | --- | --- |
| `Choice(instructions=..., criteria={id: description})` | `ChoiceAnswer` | Existing highest-scoring candidate and its full distribution. |
| `Noul(instructions=..., criteria=None)` | `NoulAnswer` | `noul` is the softmax share assigned to the prescribed true description, between 0 and 1. |
| `Score(instructions=..., criteria=[...])` | `ScoreAnswer` | `score` is the probability-weighted rubric index: `sum(i * p_i)`, starting at zero. |

All answers retain `scores` and `probabilities`. Score also retains `legend`,
mapping `"0"`, `"1"`, and so on to the supplied descriptions. The result retains
model identity, token use, and `calibration="none"`. Question and answer unions
are exported as `JudgmentQuestion` and `JudgmentAnswer`; narrow an answer by its
class or `type` before accessing fields specific to Noul or Score.

Noul does **not** return a boolean or apply an abstention threshold. A value of
0.8 is a relative share over these two descriptions, not an established 80%
chance that the claim is true. Missing evidence does not automatically produce
0.5. Application decisions need their own evaluated policy.

Score assumes equally spaced rubric indices, not measured distances between
concepts. For three levels, probabilities `[0.5, 0, 0.5]` produce score `1.0`;
that is not evidence that the middle description fits. Keep the distribution
when interpreting a score. Kayak does not expose upstream's distribution-gap
statistic as a generic `confidence` field.

## Inspect the request without inference

```python
from kayak import JudgmentRequest, Noul, Score

request = JudgmentRequest(
    state="Customer text",
    questions={
        "verified": Noul(
            instructions="The provided evidence verifies delivery.",
            criteria={"false": "Delivery is not verified.", "true": "Delivery is verified."},
        ),
        "impact": Score(instructions="Assess impact", criteria=["Low", "Medium", "High"]),
    },
)
decision = request.as_decision()
print(decision.model_dump_json(indent=2))

# With your own backend, keep the validated request unchanged until decoding:
# raw = backend.decide(state=decision.state, questions=decision.questions)
# result = request.decode(raw)
```

The recipe is the text-only subset of [CLM schema at `7956937`](https://github.com/Contrastive-LM/CLM/blob/7956937c58ed5839c06ddc4dc6b6b61c3a3e4094/src/clm/schema.py):

- State is stripped text, two newlines, then stripped instructions.
- Noul candidate order is false then true. Defaults are
  `false: No. This is false: {instructions}` and
  `true: Yes. This is true: {instructions}`. Custom text keeps the corresponding
  `false: ` or `true: ` prefix. Missing keys and empty strings use the defaults.
- Score has at least two descriptions, preserved verbatim, with consecutive
  zero-based string IDs. Ordering defines the rubric.

Nonblank state and instructions, explicit text rubric levels, and the existing
aggregate request/token limits still apply. A Noul question uses two candidates;
a Score uses one per level. Custom Noul values must be strings, with only
`false`/`true` keys. Structured state, null description values, and unknown
fields are rejected. Typed constructors raise Pydantic `ValidationError`;
`judge` raises Kayak `InputError` for invalid requests before inference or HTTP.
Dictionary questions must include `type` explicitly.

`judge` snapshots the original questions before I/O. Frozen contracts still
contain mutable dictionaries/lists: treat them as values. The separate
`as_decision`/`decode` workflow requires retaining the same request; `/v1` does
not echo descriptions and cannot detect a same-ID wording change between calls.

## Compatibility and evidence

This is an additive Python interface. `decide`, `DecisionRequest`, and
`POST /v1/decide` remain Choice-only. No new wire enum, field, endpoint, or tensor
operation is required, and the server needs no upgrade. Typed request/result
JSON is for Python-side storage, not a new server payload.

The [reference probe](../benchmarks/typed_judgments.py) checks all 12 fixed cases
against the exact SHA-256-checked upstream schema, including the public adapter's
compiled requests and decoding under tied and nonuniform logits. It performs no
inference unless you explicitly supply a service:

```sh
curl -fL https://raw.githubusercontent.com/Contrastive-LM/CLM/7956937c58ed5839c06ddc4dc6b6b61c3a3e4094/src/clm/schema.py -o /tmp/clm-schema-7956937.py
uv run -m benchmarks.typed_judgments --reference /tmp/clm-schema-7956937.py
```

This checks recipe conformance, not language understanding. Tests also compare
local and HTTP results with explicitly compiled Choice calls on a tiny random
Qwen model, check weighted averages independently, and retain bimodal and tied
distributions. These checks support the adapter implementation within the
tested scope. Full-checkpoint Noul/Score task quality and calibration remain
unverified; the encoder uncertainty in the [model contract](model-contract.md)
still applies. The existing synthetic probe labels are not a representative
application benchmark. Use reviewed cases for your task before choosing thresholds
or interpreting rubric averages as useful measurements.
