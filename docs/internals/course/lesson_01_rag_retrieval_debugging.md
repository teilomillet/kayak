# Lesson 1 Draft: RAG Retrieval Debugging

## Problem

The learner has a familiar complaint:

- "the answer exists in my corpus, but retrieval still returns the wrong text"

This lesson uses that pain to introduce late interaction as a debugging model,
not only as an implementation detail.

## Natural Opening

Do not open this lesson by defining late interaction.

Open it like this:

- show the learner one annoying retrieval miss
- show a second ranking where the right document wins
- ask what changed

Only after the learner feels that difference should the lesson name:

- mean pooling
- token-level interaction
- exact search as a correctness anchor

## What The Learner Should Be Able To Do

After this lesson, the learner should be able to:

- look at two rankings and explain why a pooled baseline can miss an
  evidence-bearing document
- describe `LateQuery`, `LateDocuments`, and `LateIndex` as explicit retrieval
  objects
- inspect a retrieval result in terms of query-token matches instead of only a
  single document embedding
- use exact local `kayak.search(...)` as a correctness anchor before discussing
  acceleration

## Primitives Introduced

- `LateQuery`
- `LateDocuments`
- `LateIndex`
- exact `search(...)`
- `query_batch(...)`
- `search_batch(...)`

## Core Claim

On a deterministic toy example, a mean-pooled dense baseline can rank a partial
match above the truly relevant document, while Kayak's exact late-interaction
path ranks the relevant document first.

That claim should land as:

- "I changed one simplification and the winner changed"

not as:

- "I have now completed a formal lesson on retrieval geometry"

Reason this is worth teaching:
- it is small enough to understand line by line
- it isolates one real failure mode:
  - an evidence-bearing document contains the right tokens but also lots of
    irrelevant content
- it lets the learner see what the late-interaction primitive is buying them

## Evidence

| Claim | Status | Evidence |
| --- | --- | --- |
| Kayak exact search ranks the evidence-bearing toy document above the partial-overlap distractor. | Verified locally | [../../../python/tests/test_course_rag_debugging_smoke.py](../../../python/tests/test_course_rag_debugging_smoke.py) |
| The paired mean-pooled dense baseline ranks the partial-overlap distractor above the evidence-bearing toy document. | Verified locally | [../../../python/tests/test_course_rag_debugging_smoke.py](../../../python/tests/test_course_rag_debugging_smoke.py) |
| Batch search on one fixed index matches per-query exact search on the same toy corpus. | Verified locally | [../../../python/tests/test_course_rag_debugging_smoke.py](../../../python/tests/test_course_rag_debugging_smoke.py) |

Runnable teaching artifact:
- [notebooks/rag_retrieval_debugging_with_kayak.ipynb](notebooks/rag_retrieval_debugging_with_kayak.ipynb)

## Rerun Command

```bash
PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_course_rag_debugging_smoke -v
```

## Where This Does Not Yet Generalize

This lesson is intentionally narrow.

It should feel playful and obvious to the learner.
But the conclusion is still narrow.

It does not prove that:

- late interaction always beats dense retrieval
- this exact failure mode dominates real production corpora
- the toy rank order predicts benchmark leaderboards

What it does prove:

- there exists one simple, deterministic setting where token-level interaction
  preserves the evidence signal and mean pooling hides it

That is the right first lesson because it teaches the primitive cleanly.
Broader quality claims should come later from judged slices and benchmark notes.

## Follow-On Evidence To Add

The next stronger version of this lesson should add:

- one compact real-data notebook on a judged retrieval slice
- one side-by-side failure analysis where dense or chunk-based retrieval loses
  the evidence before reranking even starts
