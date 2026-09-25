# Use Laya or Jev through Kayak

Keep your `Choice`, `Noul`, and `Score` questions and call `judge()` with a
different execution provider. Import adapters from `kayak.adapters`; the base
Kayak installation imports neither SDK and downloads no models.

The install commands below use the published 0.4.0 package, which already
contains these adapters. To use the 0.5.0 development version, substitute
`/path/to/kayak` for `kayak==0.4.0` after cloning `main`.

## Laya

Install Kayak and Laya into your application environment:

```sh
uv add 'kayak==0.4.0' 'laya==0.3.20' 'transformers<5'
```

```python
import laya
from kayak import Choice, Noul, Score
from kayak.adapters import Laya

with laya.load("convaiinnovations/laya", device="cpu") as agent:
    judge = Laya(agent)
    result = judge.judge(
        state="I was charged twice for one order. I can still use my account.",
        questions={
            "team": Choice(
                instructions="Which team should handle this?",
                criteria={"billing": "Charges and refunds", "support": "Technical problems"},
            ),
            "duplicate": Noul(instructions="The customer reports a duplicate charge."),
            "impact": Score(
                instructions="Assess the reported impact.",
                criteria=["No disruption", "One operation affected", "Account unusable"],
            ),
        },
    )

team = result.answers["team"]
if team.type == "choice":
    print(team.choice)
print(result.model_dump_json(indent=2))
```

Laya's loader downloads its own checkpoint when needed. Configure its device,
hooks, and other settings directly. In the pinned `laya==0.3.20` release, `load`
has no `revision` argument. To pin model weights, download a specific checkpoint
revision first and pass its local directory to `laya.load(...)`, as in the
[released-model validation](records/provider-validation.md#versions-and-retained-evidence).
`Laya` also accepts an already
constructed `laya.Router`; routing remains Laya's responsibility. The adapter
does not load, close, or reconfigure either object.

## Jev

Install the SDK and configure `TYPESAFE_API_KEY` in your environment:

```sh
uv add 'kayak==0.4.0' 'typesafe-sdk==0.7.1'
```

The caller owns the SDK client and its model, timeout, and retry settings:

```python
from typesafe_sdk import RetryPolicy, TypeSafeClient
from kayak import Noul
from kayak.adapters import Jev

with TypeSafeClient(timeout=30, retry=RetryPolicy(max_retries=0)) as client:
    judge = Jev(client)
    result = judge.judge(
        state="Please refund the duplicate payment.",
        questions={"refund": Noul(instructions="The customer requests a refund.")},
    )

answer = result.answers["refund"]
if answer.type == "noul":
    print(answer.noul)  # Provider-reported value; no automatic yes/no threshold.
```

For asynchronous code, construct `AsyncTypeSafeClient` in `async with`, wrap it
in `AsyncJev(client)`, and `await judge.judge(...)`. Cancellation reaches the
awaited SDK call directly. Kayak adds no retries or background tasks; SDK retries
still apply if you configure them. There is no thread-based async Laya wrapper.

For Jev through OpenRouter, configure the same SDK client explicitly. Set
`OPENROUTER_API_KEY` in your application environment, then use this client around
the same `judge.judge(...)` call:

```python
import os
from typesafe_sdk import RetryPolicy, TypeSafeClient
from kayak.adapters import Jev

with TypeSafeClient(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api",
    model="jev-1.13",
    timeout=30,
    retry=RetryPolicy(max_retries=0),
) as client:
    judge = Jev(client)
    result = judge.judge(
        state="Please refund the duplicate payment.",
        questions={"refund": {"type": "noul", "instructions": "A refund is requested."}},
    )
```

This follows [OpenRouter's TypeSafe SDK example](https://openrouter.ai/blog/insights/what-is-jev/).
The same configuration works with `AsyncTypeSafeClient` and `AsyncJev`.
The [2026-09-25 live validation](records/provider-validation.md) exercised both paths;
the credential is sent to OpenRouter.

## What stays the same, and what the result means

Adapters accept the existing validated **text-only question subset and request
size limits**. State, instructions, candidate IDs, order, and descriptions reach
the provider unchanged. Noul and Score remain native typed questions: the
adapters do not send CLM's compiled false/true descriptions. Each provider owns
its tokenizer, defaults, input rendering, truncation policy, and inference.
Equivalent questions do not establish equivalent model inputs or predictions.

`Laya`, `Jev`, and `AsyncJev` return `ProviderResult`:

| Field | Meaning |
| --- | --- |
| `provider` | `"laya"` or `"jev"` |
| `answers` | Named `ProviderChoice`, `ProviderNoul`, or `ProviderScore` values; narrow by `.type` before reading `.choice`, `.noul`, or `.score` |
| `model` | Provider-reported name, or `None`; this may be an alias, not an immutable checkpoint identity |
| `input_tokens` | Provider-reported usage, or `None`; accounting differs between providers |
| `calibration` | `"unknown"`: Kayak has not established calibration for your task |
| `raw` | Detached JSON response body, including provider-specific confidence, routing, action scores, usage, and other returned fields |

Probabilities are retained when supplied and otherwise remain `None`. Laya's
Noul normally supplies only the scalar. Kayak does not reconstruct a distribution,
invent logits, normalize rounded numbers, recompute a score, or change a provider's
selected option on a tie. Provider `confidence` remains in `raw` with its original
meaning; it is not a common cross-provider certainty measure. A middle Score can
still reflect probability on opposite extremes of a rubric.

Response checks require matching question types, IDs, candidate sets, and rubric
text and finite bounded values. Choices must select a reported maximum when
probabilities are present, while ties retain the provider's selection. Laya's
four-decimal rounding permits at most `K * 0.00005 + 0.000001` mass error for
`K` candidates. Its score check also accounts for independently rounded rubric
probabilities and the rounded scalar. Jev does not document output precision;
its probability mass and scalar/distribution arithmetic remain unverified rather
than imposing an invented tolerance. These checks do not establish accuracy or
calibration.

Invalid requests raise `InputError` before execution. Invalid returned bodies
raise `InferenceError` without including the body in the message. Provider SDK
exceptions propagate unchanged so callers can handle their native error types.
The adapters take independent request and response snapshots; nested result
containers remain mutable, as with Kayak's other Python contracts.

Native CLM `Model.judge` and `Client.judge` continue returning `JudgmentResult`
with their existing score contract. Existing `decide`, `rank`, and `/v1` bodies
are unchanged. Provider adapters expose only `judge`; they are not CLM bundle
loaders or drop-in `/v1` servers.

## Evaluate and reproduce

Use the same labeled cases across providers and retain each `ProviderResult`.
For Choice comparisons, transfer selected IDs to
[`PredictionSet` and `benchmark`](classification-benchmarks.md), recording the
actual provider settings, version, checkpoint revision where available, and input
recipe. The CLM-specific `evaluate` runner does not accept `ProviderResult`.
Laya's rounded probability vectors may fail the evaluator's stricter normalization
check: omit probability metrics for those vectors; do not silently normalize them.
Noul/Score task-quality evaluation remains application-specific.

### Compare on the same cases

The runnable [provider evaluation example](../examples/evaluate_provider.py) is
available on `main` after the 0.4.0 release. Start from a `main` checkout:

```sh
git clone --branch main https://github.com/teilomillet/kayak.git
cd kayak
uv run -m examples.evaluate_provider --output .benchmarks/provider-demo
```

The default uses controlled responses and needs only the base package. It always
selects the first candidate and compares those answers with word overlap on the
same eight fictional support cases. This checks the workflow, not model quality.

Select a real provider explicitly:

```sh
# Use an existing local Laya checkpoint; an online model ID can download weights.
uv run --with 'laya==0.3.20' --with 'transformers<5' -m examples.evaluate_provider \
  --provider laya --model /path/to/laya --output .benchmarks/laya

# Set TYPESAFE_API_KEY first. This sends case text to the hosted provider and
# uses your account's inference allowance/billing; SDK retries are disabled.
uv run --with 'typesafe-sdk==0.7.1' -m examples.evaluate_provider \
  --provider jev --model jev-1.13 --output .benchmarks/jev
```

Use `--suite your-suite.json` for reviewed labels in the
[Suite format](evaluation-python.md), or edit a copy of
[the support suite](../examples/suites/support.json). Each run makes one
sequential call per case. Only text and the question reach the provider; expected
labels stay in evaluation. The caller retains model/client ownership. To use an
already configured adapter (including an OpenRouter client), call the example's
`run(judge, suite, output=..., system=..., evidence_kind="provider_execution",
method="Pinned checkpoint; max_len=512; head_max_len=384; original input")`
inside its owner's context. Describe changed provider settings and input recipes
in `method`; supply supporting provenance through `metadata`. The comparator
checks `method`, not arbitrary metadata, when requiring an explicit recipe-change
comparison. The live CLI records SDK/model/configuration settings by default;
`--method` replaces that declaration with your complete recipe.

Each new output directory contains the full suite, `responses.jsonl` with raw
provider evidence, `predictions.json`, and `comparison/benchmark.md`. Raw responses
may contain application data; keep these local artifacts private. An exception or
Ctrl-C stops execution, saves partial results, and propagates. Failed and uncalled
cases remain in the scoring denominator. Existing output directories are refused.
Abrupt process termination or a storage failure can prevent final report writing.

This example compares selected IDs only. It retains reported probabilities in
the raw evidence without normalizing them or treating confidence as calibrated.
Model names are provider reports or requested aliases; immutable checkpoint
identity is not verified. Record pinned weights separately when reproducibility
requires them. Compare saved outputs without further provider calls:

```sh
uv run kayak eval benchmark \
  .benchmarks/laya/predictions.json .benchmarks/jev/predictions.json \
  --allow-recipe-change --output .benchmarks/laya-vs-jev
```

You can include a saved native CLM evaluation directory in the same command.
Paired comparisons require matching cases, labels, and candidate order, and a
prediction for every selected example. Different provider methods are explicit.

### Check the adapter contracts

The repeatable integration check is:

```sh
# In an environment with this checkout, the SDKs above, and transformers<5:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/check_provider_adapters.py
```

It constructs tiny random Laya weights locally, compares three mixed requests
with direct Laya execution, and exercises actual synchronous/asynchronous Jev
SDK clients through a controlled HTTP transport. No released model weights or
hosted inference are used. This supports integration behavior, not learned task
quality, production latency, or calibration.

The separate [released Laya and live Jev validation record](records/provider-validation.md)
documents real-provider checks, exact versions, observed variation, and limits.

Inspected upstream contracts:
[Laya source at 4066d5d](https://github.com/NandhaKishorM/laya/blob/4066d5d5fbf08b66c6757ddeedbd797bd7655bc0/laya/agent.py),
[TypeSafe SDK at 0ffd094](https://github.com/typesafe-ai/typesafe-sdk-python/tree/0ffd094c72ed9445223060b24ffd7a56aa781fb4/src/typesafe_sdk).
