# Integrate and evaluate your RAG application

## Give retrieved evidence to Kayak's typed questions

For a retrieval-augmented **Kayak response**, the flow is:

```text
query → your retriever → query + source text as state → Kayak questions → typed answers
```

Start with the [48-line decision example](../examples/rag_decisions.py). It puts
retrieved source IDs and text into `state`, asks `Choice`, `Noul`, and `Score`
questions with `client.judge`, and prints their native answers, distributions,
and model identity. Replace its tiny `retrieve(query)` lookup with your index or
RAG service. The client and index stay under your application's ownership.

```sh
uv sync
# Start your Kayak service separately; see the hardware guide before loading:
uv run --extra serve kayak serve --device auto
# In another terminal:
uv run -m examples.rag_decisions
```

The [hardware guide](validation.md) describes the full CLM's requirements.
`KAYAK_BASE_URL` and `KAYAK_API_KEY` point the example at an existing service.
For an already loaded local model, the same `model.judge(state=..., questions=...)`
call applies. A Choice-only task can use `decide`.

Kayak consumes text state, so this example explicitly serializes the query and
source mapping with `json.dumps(..., ensure_ascii=False)`. Questions stay separate
from that evidence. A Choice selects a supplied answer such as `30_days` or
`unknown`; Noul returns its true-candidate share; Score returns a rubric average.
These values are uncalibrated. An `unknown` candidate or an evidence question can
also be answered incorrectly. The example stops before inference on empty retrieval.

For a small experiment, compare the returned Choice ID against an independently
labeled ID. Inside `main`, after `judge` returns, a minimal check is:

```python
answer = result.answers["retention"]
assert answer.type == "choice"
print("Matches reference:", answer.choice == "30_days")
```

Keep that reference out of inference inputs. Retain the retrieved sources, exact
state, questions, and full result to distinguish a retrieval miss from a wrong
decision on available evidence. The printed check does not enforce an exit code.
Evaluate Noul and Score against separate reviewed labels; the model's own evidence
judgment is not proof its answer is correct. See [typed judgment evaluation](typed-judgments.md)
and [recorded stage assessment](rag-evaluation.md) for larger evaluations.

This follows the boundary shown by Jev's
[passage classification](https://docs.typesafe.ai/cookbooks/classifying_rag_passages)
and [Choice/Noul search](https://docs.typesafe.ai/cookbooks/semantic_find), and Laya's
[RAG relevance benchmark](https://github.com/NandhaKishorM/laya/blob/main/research/scripts/bench_apps.py):
supplied context goes into state and typed questions operate on it. Those examples
do not establish an automatic retrieval hook inside the model. The next example
adds free-text generation when that is the desired output.

## Try a small experiment

For a **generated text answer**, the
[standalone example](../examples/rag_quickstart.py) keeps retrieval and generation
in application code. It chooses the document with the highest positive word
overlap, includes its ID and unchanged text in the prompt, then calls local
Ollama. It records the source, retrieval score, packed context, prompt, model,
and actual answer in `RAGTrace`. Kayak checks that trace with explicit references:

```python
checks = trace.evaluate(expected_answer="30 days", expected_sources=["retention"])
print(trace.answer, checks.answer_correct, checks.source_coverage["context"])
```

There are no dataset or configuration files to create. The example's lexical
lookup and Ollama call are ordinary code you can replace with your own index,
retriever, reranker, or generator. This recipe uses only `kayak.eval`; the typed
decision path above is how retrieved evidence reaches Kayak's CLM.

With [Ollama](https://docs.ollama.com/quickstart) installed and running locally:

```sh
uv sync
ollama pull gemma3:1b
uv run -m examples.rag_quickstart
```

For a CLI installation, start `ollama serve` in another terminal first.
The example calls [Ollama's generation endpoint](https://docs.ollama.com/api/generate)
with the retrieved context. If the model returns the expected answer, it prints:

```text
Answer: 30 days
Expected source in context: True
Answer matches reference: True
```

Change the document's `30 days` to `90 days`, leaving the reference unchanged,
and rerun to inspect the changed answer and check. Answer checking is case-sensitive
exact matching after stripping surrounding whitespace. It does not assess semantic
equivalence or grounding. `expected_sources` means all listed sources are required;
other sources remain unjudged. Omitted labels and unobserved answers remain unknown.
The example stops before generation when no evidence matches. HTTP failures,
malformed responses, incomplete generation, and blank answers raise; the HTTP
client closes on success and failure. Quality checks print without enforcing an
exit code. Enforce a check explicitly in CI, for example
`assert checks.answer_correct is True`.

Keep actual source text and outputs when recording a trace, and omit stages you
did not observe. Traces may contain application data; persist them according to
your application's data policy. To retain partial failures or repeated attempts,
use the experiment examples below.

## Expand when you need more cases or controls

Use `kayak.eval` with an ordinary Python callable or recorded JSON from any
language. Your application owns its retriever, reranker, generator, configuration,
and resources. Kayak runs the callable or scores the recorded attempts, keeps
stage observations and independent judgments, and produces a checked report.

The synchronous boundary is `pipeline(RAGInput) -> RAGTrace | RAGOutput`.
Preserve the case ID and original query in your final trace. Use `RAGOutput` to
retain additional observed steps, such as rewritten-query searches. No framework,
base class, provider, or vector store is required by the evaluation APIs.

## Choose your starting point

| Your situation | Start here | What to replace or supply |
| --- | --- | --- |
| Retrieved knowledge should inform a Kayak typed response | [Choice/Noul/Score with retrieved context](../examples/rag_decisions.py) | Replace `retrieve`; retain the rendered state and native answers |
| Retrieve evidence and generate a text answer | [Standalone Ollama example](../examples/rag_quickstart.py) | Replace the lexical lookup or generation call; record actual outputs |
| Existing synchronous Python RAG | [Callable and configuration example](../examples/evaluate_rag_pipeline.py) | Replace `Pipeline.run`; construct your backend once |
| Async service, rewritten queries, or multiple retrieval steps | [Async example](../examples/evaluate_rag_async.py) | Await your calls; retain each observed step and the final trace |
| HTTP service, including a Go or TypeScript application | [HTTP adapter](../examples/evaluate_rag_http.py) | Map the service's response; own its client and timeout |
| Existing logs or a pipeline in another language | [JSON exchange below](#use-json-from-any-language) | Write attempts against the exported schema, then score offline |
| ColBERT, late interaction, cross-encoder, hybrid search, or a custom index | [Ranking and trace guide](rag-evaluation.md) | Supply stable source IDs and native scores; no embedding shape is prescribed |
| A reranker returning only its top results | [Incomplete rankings](rag-evaluation.md#score-recorded-rankings) | Declare a prefix; unobserved ranks remain unknown |
| Answer-only or retrieval-only observability | `RAGTrace` with the observed fields | Omit unobserved stages; add only the judgments you have |
| Find why an answer failed | [Stage assessment and diagnostic replay](rag-evaluation.md#test-a-failure-hypothesis) | Retain inputs and outputs, change a boundary, rerun downstream work |
| Human review or your own LLM judge | `RAGReview` or [external review JSONL](#use-json-from-any-language) | Record a rubric, reviewer identity, separate scores, and the supplied fingerprint |

The following model-free experiment examples run from the **0.5.0 development checkout**,
using only the base package:

```sh
uv sync
uv run -m examples.evaluate_rag_pipeline --output /tmp/kayak-rag-report.json
uv run kayak eval rag report /tmp/kayak-rag-report.json
uv run -m examples.evaluate_rag_pipeline --context-limit 0
uv run -m examples.evaluate_rag_async --max-concurrency 2 --repeats 2
uv run -m examples.evaluate_rag_http
```

Use a new report path each time. The first fixture passes its configured gates;
setting its context limit to zero changes the executed pipeline and fails them.
The async example retains four attempts and rewritten-query search steps. The
HTTP example defaults to `httpx.MockTransport`; `--base-url` explicitly enables
calls to your compatible service. These are integration fixtures, not evidence
of learned-model quality. The example programs print reports; their exit codes
do not enforce quality gates. Use the scoring CLI or check `report.passed` in CI.

## Keep inputs, settings, and judgments separate

```text
RAGCase.input → your pipeline → RAGOutput(final, steps)
RAGCase references + RAGOutput → your reviewer → RAGReview
dataset + settings + all attempts → score_rag → RAGReport
```

`RAGInput` contains `id`, `query`, and JSON `parameters` such as a tenant filter,
conversation history, or collection. The pipeline receives a detached copy of
this record. References and gold labels stay in `RAGCase`, visible to the reviewer
and scorer. Keep evaluation answers out of `parameters` and backend settings.

`RAGEvalConfig` owns `k`, `repeats`, `max_concurrency`, and `gates`. Its `system`
dictionary records **caller-owned settings**; Kayak does not apply them to a
backend. Validate those settings in your application, construct the backend,
and record the effective values, including defaults. The callable example does
this with a small `PipelineSettings` type and [editable config](../examples/rag/config.json).
Retain model, index, prompt, code revision, seeds, and relevant environment details
there or in declared trace provenance when they matter to reproduction.

This complete fixture integration illustrates the public API:

```python
from pathlib import Path

from examples.evaluate_rag_pipeline import Pipeline, PipelineSettings, review
from kayak.eval import RAGDataset, RAGEvalConfig, evaluate_rag, save_rag_report

dataset = RAGDataset.model_validate_json(Path("examples/rag/dataset.json").read_bytes())
config = RAGEvalConfig.model_validate_json(Path("examples/rag/config.json").read_bytes())
settings = PipelineSettings.model_validate(config.system)
config = RAGEvalConfig.model_validate(
    {**config.model_dump(), "system": settings.model_dump(mode="json")}
)
pipeline = Pipeline(settings)
report = evaluate_rag(dataset, pipeline.run, config=config, review=review)
save_rag_report(report, "/tmp/my-rag-report.json")
if report.passed is not True:
    raise SystemExit("The configured quality gates did not pass")
```

The imports from `examples` are fixture code to replace with your own pipeline
and reviewer. The [three-case dataset](../examples/rag/dataset.json) keeps its
labels independent of the pipeline. Add your reviewed ordinary, confusing,
missing-evidence, multilingual, and outside-scope cases before making quality claims.

The synchronous contract is `pipeline(RAGInput) -> RAGTrace | RAGOutput` and
optionally `review(RAGReviewInput) -> RAGReview`. `RAGTrace` is wrapped as a final
output for convenience. `aevaluate_rag` takes awaited versions of those functions;
it does not move blocking code to threads. Load indexes and models outside the
callbacks. The caller owns retries, timeouts, external side effects, and cleanup.

Each repetition invokes the pipeline anew. There is no implicit retry or cache.
Async workers bound concurrent pipeline/review/hook calls by `max_concurrency`,
without allocating a task per queued case. Reports use dataset/repetition order
even when calls finish out of order. Interruptions propagate; owned workers are
canceled and awaited. Synchronous evaluation requires `max_concurrency=1`.
The runner retains its dataset and report in memory. Bounded concurrency limits
active calls, not the total size of retained source text or attempt history.

## Map observations without inventing stages

Use the original input ID and query on `output.final`. Put rewritten queries in
`RAGOutput.steps`, each with a distinct trace ID. Steps retain their own source
IDs, outputs, errors, and provenance in recorded order. Only the final trace is
scored against final-task labels. To assess an intermediate query independently,
call `assess_rag` with its own judgments. Recorded order does not establish a
causal graph or prove that one stage caused another's failure.

An answer-only adapter can return `RAGTrace(id=request.id, query=request.query,
answer=answer)`. A retrieval adapter adds `retrieval=RankedOutput(ids=ids,
scores=scores)`. Omit scores if the backend does not provide them; supplied scores
must cover exactly the returned IDs and remain on their native finite scale.
Use `RAGContext(ids=ids, text=rendered_context)` for the context actually sent to
the generator. Record stable IDs and exact rendered text where available.
Missing fields mean unobserved; an observed empty list means no results.

Record known stage failures in `RAGTrace.errors` and downstream `blocked` stages
when partial observations survive. An uncaught pipeline exception instead becomes
a `RAGFailure` with phase and exception type; raw exception messages are omitted.
Invalid output is a separate `output` failure. Review failures preserve successful
pipeline output and appear in `review_error`. Later cases continue after ordinary
callback exceptions. These failures cannot satisfy experiment quality gates.

## Judge and configure meaningful gates

The reviewer sees `RAGReviewInput(case, output)`. Return a `RAGReview` with
`review_input_sha256=request.sha256` and independent `correct` and/or `grounded`
judgments, or named `RAGScore` values. Correctness needs an observed answer;
grounding also needs observed context. A reference answer alone is not a review.
Use a human rubric or your own judge; the fixture's exact equality and literal
grounding are deliberately limited. Record judge configuration and rubric in
`provenance`, and validate automated judgments on reviewed examples.

Custom scores retain their native scale. For example, a reviewer can return
`scores={"citation_accuracy": RAGScore(value=0.8)}` or
`RAGScore(value=None, reason="No citations were available to review")`.
Score names use lowercase letters, digits, and underscores; report names add
`custom.`. Define their meaning, units, and direction in your rubric.

| Metric family | Examples | Evidence required |
| --- | --- | --- |
| Routing | `routing.correct` | Observed route and acceptable route IDs |
| Ranking | `retrieval.hit_at_k`, `reranking.ndcg_at_k`, `shortlist_reranking.known_recall_at_k` | Ranked IDs and sufficient relevance judgments |
| Evidence | `source_coverage.context`, `context.required_text_retained` | Required source sets/text and observed context |
| Answer review | `answer.correct`, `answer.grounded` | Independent bound review |
| Your rubric | `custom.citation_accuracy` | Your named numeric score |
| Timing | `pipeline.duration_seconds`, `review.duration_seconds` | Observed callback wall time |

Each `RAGGate` sets `minimum` and/or `maximum` on the known-value mean, plus
`min_coverage` (default 1.0). For example, `RAGGate(metric="answer.correct",
minimum=0.9, min_coverage=1.0)` requires all eligible attempts reviewed and at
least 90% correct on those attempts. Unknown values are neither zero nor success.
A misspelled or absent metric has unknown coverage and cannot pass. Choose gates
before examining your holdout results; the starter thresholds are illustrative.

Reports retain every repeat, failure, missing attempt, and diagnostic exclusion.
Metrics show known/unknown counts, coverage, mean, minimum, maximum, and population
standard deviation. Coverage includes missing and failed eligible attempts.
Diagnostic replays are excluded from ordinary metrics; any such exclusion prevents
the whole experiment from passing configured gates. Missing attempts and any
pipeline, output, review, or recorded stage error also prevent passage.

`status="complete"` describes execution, not quality. `passed=None` means no
gates were configured. A gate can pass on an ordinary subset while the whole
report fails; use `report.passed` for the experiment decision. Differing output
hashes across repeats populate `unstable_cases`, including changes in intermediate
history or provenance. Repeats are not independent labeled tasks, and these
descriptive statistics are not confidence intervals. Timings exclude validation,
scoring, and completion hooks; await/synchronize device work inside your callback
when required. Imported timings and execution provenance remain declarations.

## Keep completed attempts during an interrupted run

An optional `on_attempt` hook receives a detached completed attempt, including
pipeline or judge failures, before that worker advances. The caller owns storage:

```python
from kayak.eval import RAGAttempt

with open("/tmp/rag-attempts.jsonl", "x", encoding="utf-8") as journal:
    def retain(attempt: RAGAttempt) -> None:
        journal.write(attempt.model_dump_json() + "\n")
        journal.flush()

    report = evaluate_rag(dataset, pipeline.run, config=config, review=review, on_attempt=retain)
```

Async hooks must be awaited functions and may run concurrently; synchronize your
writer if necessary. Hook failures propagate and stop owned workers rather than
becoming model errors. There is no automatic resume. Completed journal entries
can be scored offline; an interrupted or canceled invocation without an entry
remains missing. Partial output inside an interrupted callback is not recovered.

## Use JSON from any language

The versioned contract is ordinary JSON, with one attempt per JSONL line. Export
the exact schemas instead of guessing field names:

```sh
uv run kayak eval rag schema input > /tmp/rag-input.schema.json
uv run kayak eval rag schema output > /tmp/rag-output.schema.json
uv run kayak eval rag schema attempt > /tmp/rag-attempt.schema.json
uv run kayak eval rag validate examples/rag/dataset.json --config examples/rag/config.json
uv run kayak eval rag inputs examples/rag/dataset.json --config examples/rag/config.json \
  > /tmp/rag-inputs.jsonl
```

Each exported line contains `input`, zero-based `repeat`, and `system`; references
and judgments are absent. Your external worker applies the recorded settings and
sends only `input` to its pipeline. Retain the complete exported fields and add
`output: {"schema_version": 1, "final": ...}`. `final.id` and `final.query` must
match `input`. On failure, add `error: {"phase": "pipeline", "type": "TimeoutError"}`
instead of output. Optional durations are seconds. Do not drop failed calls or
silently replace recorded settings with another run's settings.

For a separate human or machine reviewer, export the references and fingerprint:

```sh
uv run kayak eval rag review-inputs examples/rag/dataset.json /tmp/rag-attempts.jsonl \
  --config examples/rag/config.json > /tmp/rag-review-inputs.jsonl
uv run kayak eval rag schema review_record > /tmp/rag-review.schema.json
```

Each line contains `case_id`, `repeat`, `review_input`, and
`review_input_sha256`. The reviewer inspects `review_input` and returns a line
with `case_id`, `repeat`, and `review`, for example:

```json
{"case_id":"invoice","repeat":0,"review":{"review_input_sha256":"COPY_THE_SUPPLIED_64_CHARACTER_FINGERPRINT","correct":true,"provenance":{"rubric":"human-reviewed-v1"}}}
```

Replace the case ID and fingerprint with the supplied values. **Copy the supplied
fingerprint**; hashing your serializer's JSON does not reproduce the normalized
Python record. The fingerprint binds the full case, references, parameters, final
output, and intermediate observations. Changed references or outputs require a
new review. Review files cannot replace an already recorded review outcome.

```sh
uv run kayak eval rag score examples/rag/dataset.json /tmp/rag-attempts.jsonl \
  --config examples/rag/config.json --reviews /tmp/rag-reviews.jsonl \
  --output /tmp/rag-scored.json
uv run kayak eval rag report /tmp/rag-scored.json > /tmp/rag-scored.md
```

Omit `--reviews` when attempts already contain reviews or none exist. Unreviewed
answer quality stays unknown. `score` writes the report even when configured gates
fail, then exits **1** for execution failure, missing attempts, or unpassed gates;
it exits **0** for complete execution with passing gates or no gates configured.
Invalid data, bindings, arguments, or report paths exit **2**. No CLI command here
loads a model, calls a service, or evaluates an answer with an implicit judge.
`python -m kayak.eval rag` exposes the same commands.

## Retain and challenge the evidence

`score_rag(dataset, attempts, config=config)` is the pure Python equivalent of
offline scoring. Input and system comparisons preserve JSON scalar types and
list order, while ignoring object key order. The report retains full inputs,
labels, settings, observations, errors, and judgments. `save_rag_report` verifies
and writes one JSON file atomically without overwriting existing evidence;
`load_rag_report` recomputes bindings, assessments, summaries, and gates.
`render_rag_report` produces verified Markdown without inference.

You may rescore retained attempts with another evaluation cutoff or gate, saving
a new report. Changing backend settings requires new attempts; changing references
invalidates bound reviews. Compare systems on the same reviewed cases and retain
individual disagreements and repeats. This interface does not supply automatic
framework adapters, baseline significance tests, or authenticated execution.
Consistency hashes detect mismatched evidence, not a fabricated but internally
consistent record. See [the integration research and challenge record](records/rag-integration-research.md)
for the basis and practical limits of this design.
