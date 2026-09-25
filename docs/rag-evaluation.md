# Evaluate ranking and RAG boundaries

`kayak.eval` can evaluate an external ranker and assess recorded RAG stages.
For retrieved evidence feeding Kayak's own typed answers, start with
[RAG decisions](../examples/rag_decisions.py): context goes into `state`, and
`judge` returns Choice/Noul/Score answers to evaluate against separate labels.
Record your application's actual retrieval, reranking, context, and answer in
`RAGTrace`, then use `trace.evaluate(...)` for small reference checks or `assess_rag`
for detailed judgments. Your application owns execution, model/index resources,
and reviews. The standalone example shows retrieval and optional text generation.
For setup, configuration, sync/async experiments, external judges, JSON exchange,
and quality gates, start with [RAG experiments](rag-experiments.md).

| What you need | Public API | Runnable starting point |
| --- | --- | --- |
| Check an application's retrieved context and generated answer | `RAGTrace`, `trace.evaluate` | [Standalone generation example](../examples/rag_quickstart.py) |
| Run labeled Choice or shortlist-ranking cases | `parse_cases`, `evaluate_cases`, `summarize_cases` | [Use-case runner](../examples/evaluate_use_cases.py), [three starter datasets](../examples/evaluations/README.md) |
| Plug in late interaction, a cross-encoder, or another ranker | A `rank(request)` callback returning `RankedOutput` | [Synthetic MaxSim example](../examples/evaluate_late_interaction.py) |
| Score retrieved IDs with partial or graded relevance judgments | `assess_ranking` | [Partial-judgment example below](#score-recorded-rankings) |
| Locate evidence loss and review final answers | `RAGTrace`, `RAGJudgments`, `assess_rag` | [RAG trace and diagnostic replay](../examples/evaluate_rag.py) |
| Run an existing RAG application and evaluate settings | `evaluate_rag`, `aevaluate_rag`, `score_rag` | [Experiment guide](rag-experiments.md), [callable example](../examples/evaluate_rag_pipeline.py) |
| Compare fixed-label classifiers and probability metrics | Existing `Suite`, `evaluate`, `benchmark` | [Classifier benchmark guide](classification-benchmarks.md) |

The existing classification `Suite`/`Report`, `load_report`, and CLI formats stay
separate. Case reports and RAG assessments do not become classification reports.
The APIs here require only the base package; imports do not load a model.

## Run cases with a callback

The [starter files](../examples/evaluations/README.md) contain complete requests
and judgments. Use the same data with a local model or HTTP client:

```python
from pathlib import Path

from kayak import Client, RankingRequest, RankingResult
from kayak.eval import evaluate_cases, parse_cases, summarize_cases

dataset = parse_cases(Path("examples/evaluations/rerank_documents.json").read_bytes())
with Client(base_url="http://127.0.0.1:8000") as client:
    def rank(request: RankingRequest) -> RankingResult:
        return client.rank(
            state=request.state,
            instructions=request.instructions,
            candidates=request.candidates,
        )

    reports = evaluate_cases(dataset, rank=rank)
print(summarize_cases(dataset, reports))
```

For Choice files, pass a `decide(request)` callback that calls
`client.decide(state=request.state, questions=request.questions)`. A combined
caller can provide both callbacks; only the dataset's kind is used. Requests can
vary between cases. The dataset's `examples` list is optional navigation metadata.

For an external ranker, implement one ordinary function or method:

```python
from kayak import RankingRequest
from kayak.eval import RankedOutput

def rank(request: RankingRequest) -> RankedOutput:
    # Replace this input-order baseline with your already-loaded backend.
    return RankedOutput(
        ids=list(request.candidates),
        provenance={"implementation": "input-order-baseline-v1"},
    )
```

`ids` are best first and must be a complete permutation of that case's candidate
IDs. Optional `scores` must contain exactly those IDs and finite values. They
retain the backend's native scale; the evaluator does not invent probabilities,
reorder scores, or require CLM metadata. Native `RankingResult` remains supported
with its existing validation. Equal external scores need an ordering policy in
your adapter; the MaxSim example preserves input order on ties.

Load weights, build an index, and prepare documents once, outside the callback.
The caller owns cleanup. A callback receives a fresh request, without case IDs
or labels. Inputs and outputs are copied and validated at the evaluation
boundary. Calls are sequential and happen once per case, without retries.
Failures remain in quality denominators; raw exception messages are omitted.
Model identity changes, declared provenance changes, and switching output types
within a run count as failed cases. Missing provenance stays unknown, and
declared identity is not independent proof of which backend ran.

Each `CaseReport` adds `case_sha256`, binding its exact request, labels, candidate
order, and metric cutoff. `summarize_cases` requires a complete ordered run and
verifies those bindings and the metrics against retained outputs. Changed labels
or inputs cannot silently reuse an old score. Keep the original dataset with
reports; the digest checks consistency rather than authenticating execution.

`duration_seconds` covers the callback and optional completion hook. Pass
`sync=your_device_synchronize` for asynchronous device work. Initial
synchronization, preparation, and result validation are excluded. Warmup,
caching, hardware, and which encoding steps occur inside the callback still
need recording before comparing speed. Interruptions propagate to the caller.

These case requests retain Kayak's request limits, including 256 candidates.
Use the ID-based assessment functions below for outputs from a larger corpus.
They do not impose CLM input limits or prescribe embedding shapes.

Run the small late-interaction example without a service:

```sh
uv run -m examples.evaluate_late_interaction
```

It computes MaxSim on manually supplied token vectors: sum each query token's
maximum dot product against document tokens. The arithmetic follows
[ColBERT's implementation](https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/modeling/colbert.py).
Replace its prepared matrices and scoring boundary with your own library.
This exercises the adapter and arithmetic; it does not load ColBERT or establish
learned-model accuracy or inference speed. Multi-vector and multimodal backends
fit when their adapters return stable source IDs; Kayak does not encode or
store their vectors, images, or model state.

## Score recorded rankings

```python
from kayak.eval import assess_ranking

assessment = assess_ranking(
    ["unreviewed", "invoice"],
    {"invoice": 2, "policy": 1, "irrelevant": 0},
    k=2,
)
print(assessment.metrics)
print(assessment.unavailable_reasons)
```

Positive integer grades mean relevant, zero means judged irrelevant, and an
absent ID is unjudged. In this example `hit_at_k` is 1, known-positive recall is
1/2, and top-one quality, reciprocal rank, and nDCG are unknown. An unjudged
first result might itself be relevant; removing it would change the ranking.

`known_recall_at_k` counts only the supplied positive judgments, including
positives absent from the returned list. It establishes total-corpus recall only
if the judgments cover that corpus. `ndcg_at_k` uses linear grade gains and
requires a fully judged prefix. Each unavailable metric is `None` with a reason.
An observed empty retrieval differs from a missing observation. No known
positives means an unknown recall denominator, not perfect recall.

Some rerankers return only a prefix. Pass `complete=False` to `assess_ranking`
for those observations. At cutoffs inside the recorded prefix, ordinary metrics
apply. Beyond it, only values already established by the prefix remain known:
for example, a found positive establishes a hit, while an unseen tail prevents
declaring a miss or complete ranking quality. The default `complete=True` retains
the complete-observation contract. The case callback's `RankedOutput` still
requires a full permutation; prefix support belongs to the assessment APIs.

The starter-case `ranking_metrics` convention is deliberately simpler: every
candidate outside `relevant` is labeled irrelevant, and each case needs a
positive. Use `assess_ranking` for incomplete labels rather than converting
unjudged sources to negatives. Keep unknown counts and execution failures
visible when aggregating your own assessments.

## Record a pipeline and inspect its boundaries

Run a complete, deterministic example:

```sh
uv run -m examples.evaluate_rag
```

It retains a relevant invoice passage during retrieval, ranks it second, and
packs only the first passage. A separate diagnostic run expands the context
and executes the same answer function again. Two JSON lines retain the traces,
judgments, and assessments. Both retrieval and reranking are declared fixtures;
packing and the answer function execute. No language model runs.

The records have three responsibilities:

- `RAGTrace` retains the query and observed route, retrieved IDs, reranking,
  exact rendered context, answer, and declared provenance. Optional document
  snapshots preserve source text. Explicit `errors` and `blocked` outcomes
  distinguish failed execution from stages that were not recorded.
- `RAGJudgments` holds separate relevance grades, acceptable routes, alternative
  sets of required sources, exact required text, and optional answer review.
- `assess_rag(trace, judgments, k=...)` compares those values without executing
  a stage or calling an answer judge.

Reranking defaults to preserving every known input ID. For an API returning only
its top results, set `reranking_complete=False` on the trace. Returned IDs must
still come from known inputs; unknown tail ranks stay unknown. Known positives
outside the returned prefix appear in `reranking_unreturned_known_positives`,
separately from observed below-cutoff losses. If you filter retrieved sources first,
record that subset as `reranking_input_ids`; omitted known positives are reported
at this input boundary. Context can also select a subset. For a partial recording,
`reranking_input_ids` or `context_input_ids` records input without claiming an
earlier stage ran. Keep source IDs stable; chunking or rewriting sources needs
its own mapping before constructing this trace. Invalid lineage is rejected
rather than silently repaired.

| Observation | What the assessment can establish |
| --- | --- |
| A known positive is absent from retrieval | An observed retrieval miss under those judgments |
| Filtering removes a retrieved positive before reranking | An observed omission at the ranker's input boundary |
| A retrieved positive falls below `k` | An observed ranking loss at that cutoff |
| A positive available to packing is omitted | An observed context-selection loss |
| A source ID survives but its required text is absent | Source coverage differs from literal text retention |
| A stage reports an error | Execution failed; quality for its missing output remains unknown |
| An answer has no independent review | Answer correctness and grounding remain unknown |

`retrieval` and `reranking` metrics use the full supplied relevance map.
`shortlist_reranking` conditions on the ranker's actual input: a perfect
conditional score cannot erase retrieval misses. `source_coverage` checks
whether at least one required source set is present: `[ ["a", "b"], ["c"] ]`
means both A and B, or C. Retrieval coverage uses all returned sources;
`reranking_top_k` uses the cutoff, and context coverage uses packed sources.
`required_text_retained` checks literal context content independently of IDs.
The per-source `context_absent_labeled_text` list can be nonempty when another
complete evidence route survives; it does not imply that the context failed.
None of these establishes semantic sufficiency, factual correctness, or a cause.

Supply `RAGAnswerReview(trace_sha256=trace.sha256, correct=..., grounded=...,
provenance=...)` after reviewing the actual answer. Correctness and grounding
are independent optional judgments. Record the reviewer or judge configuration
and rubric. A review bound to different inputs, source text, context, or answer
is rejected. The digest binds recorded bytes; it does not certify the reviewer.

## Test a failure hypothesis

Preserve the original trace. Change one boundary in the application, execute
downstream stages again, and attach `RAGReplay` with the original trace hash,
changed boundary, and purpose. This record declares what you did; it does not
run, reconstruct, or verify the intervention. Retain the actual inputs, model
and index versions, configuration, and outputs needed to repeat it.

The resulting assessment has `ordinary_aggregate_eligible=False`. Keep these
diagnostic runs out of ordinary quality aggregates, especially when supplying
gold evidence. The example's changed answer supports a context dependency in
that deterministic fixture. Real failure attribution needs controlled repeated
runs, an independent answer rubric, and attention to other changed variables.
An observed lost document is a reason to investigate a boundary, not proof that
one component caused the final answer failure.
