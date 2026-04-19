# Retrieval Diagnosis Playbook

This note turns the course material into an operational debugging map.

It answers four questions:

1. what symptom do I observe?
2. what is the most likely failure class?
3. how do I check that hypothesis with Kayak?
4. what intervention should I try next?

The goal is not to claim that Kayak fixes every retrieval problem.
The goal is to make the failure boundary explicit and testable.

For the broader complaint surface, pair this playbook with:

- [symptom_first_issue_catalog.md](symptom_first_issue_catalog.md)

## How To Use This Playbook

This should not be delivered like a diagnostic manual the learner memorizes up front.

Use it after the learner has already seen a miss.

The intended feel is:

- "this looks like my problem"
- "here is the quickest check"
- "here is the next thing to try"

That is better than:

- "first learn all four failure classes before touching the system"

## Quick Complaint Triage

Use this before the full pattern descriptions when a learner comes in with a
natural-language complaint.

| If the learner says | Start here | Why |
| --- | --- | --- |
| "the wrong thing keeps winning" or "it finds related pages, not the answer-bearing one" | Pattern 1, then Pattern 3 if the answer-bearing document is absent | The issue may be compressed scoring, or it may be a shortlist miss disguised as a scoring issue. |
| "we changed chunking and everything moved" | Pattern 2 | The first variable to isolate is the retrieval unit, not the reranker. |
| "the evidence seems split across pieces" | Pattern 2 | This is the cleanest entry into retrieval-unit failure. |
| "widening `k` helps" or "reranking did not help" | Pattern 3 | The answer-bearing document may never survive stage 1. |
| "long documents are where it breaks" | Pattern 3 first, then Pattern 2 | Long-document misses can be shortlist pressure, poor units, or both. |
| "the fast path is worse than the exact one" | Pattern 3 for shortlist checks, then the real-slice proxy note | This separates candidate loss from approximation loss. |
| "we made it faster and now we don't know what broke" | Decision Tree step 1, then Pattern 3 | The right first move is to reestablish one exact reference path. |
| "I think the LLM is hallucinating because retrieval is bad" | Decision Tree step 1 | First isolate whether retrieval can surface the evidence at all before blaming generation or retrieval globally. |

## Pattern 1: The Right Document Exists, But Partial Overlap Wins

### Symptom

- a document containing the real evidence loses to a distractor that repeats
  only part of the query

Typical shape:
- the distractor repeats one topical token many times
- the real document contains all needed evidence tokens plus lots of noise

This often sounds like:

- "the wrong page keeps winning"
- "we get near misses"
- "the result is related, but not actually answer-bearing"

### Why This Happens

Mean pooling or other compressed document representations can bury the evidence
structure.

The failure is:
- one repeated partial overlap dominates the pooled document representation
- the full conjunction signal is weakened by irrelevant content

### How To Check It With Kayak

Run one exact local comparison on the same slice:

- build a `LateIndex`
- run `kayak.search(...)` with the exact path
- compare that against your compressed or pooled baseline

Verified local teaching artifact:
- [lesson_01_rag_retrieval_debugging.md](lesson_01_rag_retrieval_debugging.md)
- [notebooks/rag_retrieval_debugging_with_kayak.ipynb](notebooks/rag_retrieval_debugging_with_kayak.ipynb)
- [../../../python/tests/test_course_rag_debugging_smoke.py](../../../python/tests/test_course_rag_debugging_smoke.py)

### What To Try Next

- keep the document as one retrieval unit if the evidence actually lives inside
  that unit
- use exact late interaction as the first debugging reference path
- delay approximation until the exact path shows the relevant document can win

This should sound like a practical next experiment, not a doctrine.

### Status

- `Verified locally`

## Pattern 2: Exact Search Still Misses Because The Evidence Is Split Across Chunks

### Symptom

- exact Kayak search returns multiple partial hits
- no single retrieved document carries the full evidence

Typical shape:
- one chunk contains one necessary concept
- another chunk contains the other necessary concept
- each chunk looks half-right in isolation

This often sounds like:

- "the evidence spans two chunks"
- "exact search still looks half-right"
- "no single result contains the whole thing"

### Why This Happens

This is not mainly a scoring failure.
It is a retrieval-unit failure.

If the evidence is split across different retrieval units, then a per-document
retriever cannot award a full score to a document that does not actually contain
the full match.

### How To Check It With Kayak

Build the exact same content two ways:

- as split chunks
- as a grouped retrieval unit

Then compare exact `kayak.search(...)` on both.

Verified local teaching artifact:
- [notebooks/retrieval_failure_pattern_catalog.ipynb](notebooks/retrieval_failure_pattern_catalog.ipynb)
- [../../../python/tests/test_course_failure_patterns_smoke.py](../../../python/tests/test_course_failure_patterns_smoke.py)

### What To Try Next

- regroup the retrieval unit
- widen chunk size when the evidence naturally spans chunk boundaries
- keep vector count and document grouping explicit when you compare alternatives

This is the moment to tell the learner:

- "your scoring may be fine; your unit of retrieval may be wrong"

### Important Boundary

Kayak helps here by making the retrieval unit explicit and comparable.
It does not magically reassemble split evidence if your chosen retrieval unit is
wrong.

### Status

- `Verified locally`

## Pattern 3: Exact Search Works, But The Approximate Pipeline Still Misses

### Symptom

- exact full-slice search finds the right document
- your cheaper candidate stage does not
- reranking never sees the oracle document

Typical shape:
- a narrow candidate window
- compressed stage-1 representations
- reranking blamed for a shortlist problem

This often sounds like:

- "widening `k` helps"
- "reranking did not help"
- "the right page appears only when I let more candidates through"

### Why This Happens

Reranking cannot recover a document that never entered the candidate set.

This is a candidate-stage recall failure, not mainly a stage-2 scoring failure.

### How To Check It With Kayak

Compare:

- `kayak.exact_full_scan_search_plan(...)`
- `kayak.document_proxy_search_plan(...)`

Then inspect:

- `result.candidate_stage.candidate_doc_ids`
- final `result.hits`

Verified local teaching artifact:
- [notebooks/retrieval_failure_pattern_catalog.ipynb](notebooks/retrieval_failure_pattern_catalog.ipynb)
- [../../../python/tests/test_course_failure_patterns_smoke.py](../../../python/tests/test_course_failure_patterns_smoke.py)

Broader repo evidence:
- [../../hard_recall_evaluation.md](../../hard_recall_evaluation.md)
- [../../traces/2026-04-12_synthetic_hard_recall_stage_aware.md](../../traces/2026-04-12_synthetic_hard_recall_stage_aware.md)
- [../../traces/2026-04-13_benchmark_ladder_and_long_document_hard_recall.md](../../traces/2026-04-13_benchmark_ladder_and_long_document_hard_recall.md)

### What To Try Next

- increase `candidate_k`
- increase stage-1 vector budgets
- debug on the exact path first
- only trust a compressed candidate stage after checking faithfulness against the
  exact baseline

This is where many users feel relief.
They learn they do not need a smarter reranker first.
They need the right document to survive stage 1.

### Status

- local toy case: `Verified locally`
- broader family-level claim: `Supported on a defined surface`

## Pattern 4: Quality Is Fine, But Repeated Queries Are Too Slow

### Symptom

- the same slice is searched repeatedly
- the system keeps rebuilding or rematerializing work

This often sounds like:

- "it works, but repeated use is too slow"
- "quality is okay once the index is loaded"
- "throughput is the issue now, not relevance"

### Why This Happens

The repeated-query bottleneck is often:
- loading or materializing the searchable slice over and over
- running high-level convenience flows per query

not the exact scorer alone

### How To Check It With Kayak

Compare:

- looped high-level per-query search
- one loaded `LateIndex`
- `kayak.search_batch(...)`

Relevant repo evidence:
- [../../../public/docs/storage-and-search.md](../../../public/docs/storage-and-search.md)
- [../../traces/2026-04-12_python_sdk_batch_fast_path.md](../../traces/2026-04-12_python_sdk_batch_fast_path.md)

### What To Try Next

- load one exact slice once
- reuse that `LateIndex`
- batch queries when the workload shape allows it

This section should stay late in the flow.
Users usually care more about "can it find the document?" before
"can it do that repeatedly at acceptable cost?"

### Status

- `Supported on a defined surface`

## Decision Tree

Use this order when debugging:

1. Can exact search on the intended slice find the right document?
2. If no, is the evidence split across retrieval units?
3. If yes, does the approximate or staged pipeline still lose it?
4. If no quality problem remains, is the issue now repeated-query cost rather
   than retrieval correctness?

That sequence matters because:

- it separates representation failures from stage-1 failures
- it separates stage-1 failures from stage-2 failures
- it separates quality debugging from systems tuning
