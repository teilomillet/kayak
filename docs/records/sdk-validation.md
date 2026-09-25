# SDK validation — 2026-09-25

This historical record covers candidate ranking, asynchronous HTTP calls, and
CLM typed-question recipe probes. It records the tested snapshots and decisions
at that time. For the current interfaces, use the [Python API](../api.md),
[typed judgments](../typed-judgments.md), and [model contract](../model-contract.md).

## Direction and comparison

The starting worktree was clean at `31fba8dbb41c1db20270bd0aa1c0fb1a12bf33bd`.
The model contract, architecture, migration/compatibility policies, and existing
evaluation tools determined the scope. Go remains the primary readability
reference: direct control flow, explicit effects, useful types, and shared
validation policy, following the existing [engineering conventions](../engineering.md).

The correctness reference is CLM
[`7956937`](https://github.com/Contrastive-LM/CLM/tree/7956937c58ed5839c06ddc4dc6b6b61c3a3e4094).
Its [engine](https://github.com/Contrastive-LM/CLM/blob/7956937c58ed5839c06ddc4dc6b6b61c3a3e4094/src/clm/engine.py)
and [schema](https://github.com/Contrastive-LM/CLM/blob/7956937c58ed5839c06ddc4dc6b6b61c3a3e4094/src/clm/schema.py)
were inspected directly, alongside the
[Jev SDK guide](https://docs.typesafe.ai/sdk/python) and
[Laya source/usage](https://github.com/NandhaKishorM/laya).
Jev and Laya models were not run; their documentation supplies workflow ideas,
not CLM quality or performance evidence.

| Category | Observation | Decision |
| --- | --- | --- |
| CLM implemented, previously unexposed | Its `rank` composes a Choice, scores candidate descriptions, then orders them. | Expose `Model.rank`, `Client.rank`, `AsyncClient.rank`, and `kayak rank`. Retain explicit instructions and stable IDs. |
| CLM implemented, still unexposed as native judgments | Noul and Score use fixed transformations and distribution decoding over the same released heads. | Reproduce and check these in evaluation tooling. Defer a supported judgment API until full-checkpoint quality evidence exists. |
| CLM implemented, still unexposed | Prepared state/action vectors are reused in a bounded arena; roles and head namespaces distinguish representations. | Defer caching. This machine cannot establish full 8B cold/warm latency, memory costs, or numerical differences. |
| Existing Kayak, awkward | Reranking required a named Choice and a handwritten stable sort. | A single `rank` call returns IDs, scores, and shares in order; first item selects best-of-N. |
| Existing Kayak, awkward | Async applications had to call blocking SDK methods or manage a thread just for HTTP. | `AsyncClient` provides awaited calls and explicit async connection lifetime. |
| Competitor convenience that fits | Jev documents sync and async clients plus named typed questions. | Adopt async HTTP while retaining Kayak's strict text/Choice contracts. |
| Competitor convenience that fits | Laya documents direct loading, self-hosted serving, and a CLI. | Preserve Kayak's existing load/serve path; add ranking JSON/CLI and an async action-description example. |
| Different model or unsupported assumption | Laya's encoders, multilingual/router claims, alternate runtimes, and schema-driven decisions are not Kayak's checkpoint. | Do not transplant its runtime or claim its accuracy, latency, or language coverage. |
| Different training or evidence needed | Fine-tuned verifier results, calibrated truth probabilities, reliable refusal when all candidates are bad, generation, multimodal input, or context beyond the pinned recipe. | No such capability is inferred from a compatible interface or CLM's uncalibrated softmax. |

Newer upstream was inspected separately at
[`d43894a`](https://github.com/Contrastive-LM/CLM/compare/7956937c58ed5839c06ddc4dc6b6b61c3a3e4094...d43894a1c97c83d2bb1b5cfcaac5759f72ece8a6),
four commits ahead. The diff changes publishing, installation/fine-tuning
instructions, dependencies, and a best-effort Hub download-counting HEAD request
in the head downloader. It changes neither `schema.py` nor `engine.py`, nor the
head projection/scoring code. The pinned and current schemas have identical
SHA-256 `52cec58afbf49ad7b7aa6bdb7e7476ee42bf3fd7a2703d44319dc4b565987335`.
Kayak's revisions and downloader remain unchanged.

## Selected workflows

**Before:** build a Choice, choose an internal question ID, call `decide`, and
sort the returned score mapping. **After:** call `rank` on a model or client:

```python
import kayak

candidates = {
    "lookup_invoice": "Look up the invoice and payment history for a duplicate charge.",
    "reset_password": "Send a password reset link to restore account access.",
    "ask_customer": "Ask for missing details before taking an action.",
}
with kayak.Client(base_url="http://127.0.0.1:8000") as client:
    result = client.rank(
        state="I was charged twice for invoice 4411.",
        instructions="Which action should support consider next?",
        candidates=candidates,
    )
print(result.ranked[0].id)  # Best supplied candidate; application policy decides what to do.
print(result.ranked[:2])   # Shortlist; shares still refer to all three candidates.
```

Replace the client context with `kayak.load(device="auto")` for local execution.
Candidate IDs are preserved but never encoded. Candidates with identical text
remain separate IDs. Ties preserve insertion order. No search, candidate
generation, threshold, or tool execution is attached to ranking.

From a checkout, validate without a model, then call a running service:

```sh
uv run kayak validate --ranking examples/ranking.json --pretty
uv run kayak rank examples/ranking.json --pretty
uv run kayak rank - < examples/ranking.json
uv run -m examples.rerank_documents
```

**Before:** async applications used a blocking call or thread offload.
**After:** the base package provides real async HTTP:

```python
import asyncio
import kayak

async def main() -> None:
    async with kayak.AsyncClient(base_url="http://127.0.0.1:8000") as client:
        result = await client.rank(
            state="I was charged twice.",
            instructions="Which team should handle this request?",
            candidates={"billing": "Charges and refunds", "support": "Bugs and outages"},
        )
        print(result.ranked[0].id)

asyncio.run(main())
```

`await client.decide(...)` and `await client.model_info()` have the same values
and error policy as their synchronous counterparts. A runnable sequential
action-selection example is `uv run -m examples.async_decisions`.
Async calls release the event loop while waiting for HTTP; they do not increase
the service's single-inference capacity. Cancellation and client timeouts can
leave computation running remotely. Calls do not retry, queue, or execute tools.

These two improvements were selected because they remove concrete integration
work while retaining the current trained computation. A third production change
is unnecessary: batching/reuse needs measured benefit, and native Noul/Score
needs evidence this host cannot supply. No scheduling or cache framework was added.

## Noul/Score investigation and reproducible probes

**Update:** the checked recipes are now available through the Python
[`judge()` adapters](../typed-judgments.md). The deferral described in this historical
report applied to the earlier checkpoint; full-model task quality remains unverified.

For the supported nonblank text subset, both recipes prepare state exactly as
`state.strip() + "\n\n" + instructions.strip()`. They use the same state/action
roles, last-token pooling, FP32 projections, normalization, scale, and softmax
as Choice. No special Noul or Score checkpoint is loaded by the pinned engine.
This establishes implementation applicability to the heads, not task quality.

| Recipe | Action texts, in order | Decoded meaning |
| --- | --- | --- |
| Noul, default | `false: No. This is false: {stripped instructions}`, then `true: Yes. This is true: {stripped instructions}` | `noul = p_true`, conditional on those two candidates. |
| Noul, custom descriptions | `false: {description}`, then `true: {description}`; a missing/empty description uses the corresponding default | Same two-way share; prefixes are part of the recipe. |
| Score | Each rubric description verbatim; IDs are `"0"`, `"1"`, …; at least two levels | `score = sum(index * probability)`, an expected zero-based level under equal index spacing. |

The upstream Score `legend` maps indices to descriptions; `confidence` is the
top share minus the mean of the others, not a calibrator. An expected level can
fall between levels or equal the middle level when two extremes are plausible.
Upstream's diagnostic Noul label uses `p_true >= 0.5`; its tie rule differs from
Choice's first-maximum rule. The probe preserves that distinction without
introducing an application threshold.

The [probe tool](../../benchmarks/typed_judgments.py) compiles these recipes to
explicit Choice requests and checks prepared texts and decoded outputs against
the exact pinned upstream module. It requires that module's SHA-256 before
executing it. Download the reference once and run conformance without a model:

```sh
curl -fL https://raw.githubusercontent.com/Contrastive-LM/CLM/7956937c58ed5839c06ddc4dc6b6b61c3a3e4094/src/clm/schema.py -o /tmp/clm-schema-7956937.py
uv run -m benchmarks.typed_judgments --reference /tmp/clm-schema-7956937.py > conformance.json
```

To evaluate on the pinned full model, start `uv run --extra serve kayak serve`
on suitable hardware, then run:

```sh
uv run -m benchmarks.typed_judgments \
  --reference /tmp/clm-schema-7956937.py \
  --base-url http://127.0.0.1:8000 > typed-probes.json
```

The [fixed corpus](../../benchmarks/data/typed-judgments-v1.jsonl) contains eight
independently authored labels and four deliberately unlabeled ambiguous cases.
Labels were fixed before model execution, not generated from predictions.
They are synthetic and have not received independent domain review. Negation,
custom descriptions, a rubric boundary, missing evidence, and contradictory
evidence remain in the report. Each call's model identity, scores, distribution,
typed decoding, label, and correctness are saved. Failures stay in the labeled
denominator; ambiguous cases are never assigned invented ground truth.
Exit 0 means execution completed, not that quality passed; exit 1 retains a report
with request failures, and exit 2 reports configuration/input/conformance errors.

This probe is a research tool, not a public typed-judgment contract. The existing
`DecisionRequest` and `/v1/decide` still reject Noul/Score wire questions,
structured state, and missing Choice descriptions. Upstream's more permissive
input rendering is not silently introduced. Any future supported typed endpoint
must have an explicit contract version and retain `/v1`.

## Evidence and limits

Baseline on Python 3.13.15, PyTorch 2.14.0+cpu, Transformers 4.57.6:
**287 passed, one MPS-only skip** with released heads and the CI Hypothesis
profile; Ruff, formatting, and strict mypy passed. Sandbox restrictions prevented
live sockets/event-loop checks, so those checks were rerun with loopback access.
Interrupted restricted runs are not counted as passes.

The ranking increment passed 64 focused tests. The combined client/interface
increment passed 100 tests, including actual tiny Qwen local/live-HTTP agreement,
invalid-input recovery, strict response validation, async yielding, cancellation,
retained admission capacity, and shutdown. Exact numerical equality in these
checks supports reuse of the Choice computation; randomly initialized tiny
weights cannot establish learned ranking quality.

All 12 typed probes match the pinned schema's prepared inputs and decoding for
tied and nonuniform score vectors. Application quality and calibration are
explicitly `not_evaluated`. This CPU-only host has 7.6 GiB RAM and no CUDA/MPS;
full pinned 8B execution was unavailable within its memory capacity.
The earlier M4 Pro smoke evidence in [validation](../validation.md) does not answer
the new typed-quality questions.

Final checks on the reviewed source:

| Check | Observed result |
| --- | --- |
| Ruff and formatting | Passed across 84 files. |
| Strict mypy | Passed across 83 source files; no new suppressions. |
| Full pytest, CI Hypothesis profile, released heads enabled | **350 passed, one MPS-only skip**, 39.84 s. |
| Separate environment without inference dependencies | **318 passed, two pyperf-only skips, 31 inference cases deselected**, 27.38 s. The pyperf checks pass in the full environment. |
| Isolated compatibility matrix | **All four pairings passed**; 12 original check groups in baseline-client pairs, plus ranking/async in both current-client pairs. |
| Distribution build and strict Twine checks | Wheel and source archive passed; new tools, corpus, examples, and documentation are included in the source archive. |
| Fresh base-only wheel | Installed public ranking and async calls, imports, typing marker, model manifest, and CLI smoke checks passed without inference/server dependencies. |
| Pinned reference probe | All 12 input transformations and 24 tied/nonuniform decodings matched the exact upstream schema. |
| Final diff review | Existing `Model.decide` syntax unchanged. Decision contracts, preparation, loading, numerical heads, model manifest, server, dependencies, and frozen `/v1` fixtures unchanged. |

The existing Starlette test-client deprecation warning remains. Production
changes are confined to [ranking](../../kayak/ranking.py), the
[clients](../../kayak/client.py), their public exports, and small model/CLI adapters.
Input/score validation has one existing owner; sync and async transports share
configuration, response, and error checks without a new inheritance framework.
Review also challenged Score decoding with a reordered probability mapping:
the probe now follows rubric IDs, respecting the existing accepted `/v1` behavior.

Raw local evidence is under `.benchmarks/workflows/` (ignored by Git):
`baseline-full.log`, `final-pytest.log`, `client-only-pytest.log`,
`compatibility/report.json`, `typed-conformance.json`, and `wheel.log`.
The compatibility runner verifies current package digest
`1f5b07c5fddaeb86aeda6c5f9c3a52329bce1024621cf82ab8470a8bdde9a6aa`.

## Follow-up experiments identified in this review

These items describe the review's original priorities. Typed judgments and
bounded candidate reuse have since been implemented; their current contracts
and remaining validation requirements are linked above.

1. Run the fixed Noul/Score probes and representative, reviewed application cases
   on the pinned full model. Inspect ambiguity, negation, ordinal error, candidate
   order sensitivity, and probability reliability before choosing a native typed
   API and its version. Do not tune labels to the outputs.
2. Measure repeated decisions with fixed actions using the existing inference
   harness. Compare cold/warm latency, peak memory, and exact scores before adding
   bounded reuse keyed by model identity, recipe, role, and exact prepared input.
3. Measure a real multi-state ingestion workload before adding a bounded batch
   interface. Sequential examples already bound outstanding model work; async HTTP
   serves application integration, not throughput or encoder batching.
4. Evaluate domain-specific best-of-N/tool selection on reviewed labels. The
   application retains execution authority and thresholds; even all-bad candidate
   sets have a top-ranked member.
