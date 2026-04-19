# Lesson 2 Draft: Recognizing Retrieval Failure Patterns

## Problem

Teams often know that retrieval is failing, but they do not know which layer is
actually broken.

This lesson teaches a stronger diagnostic claim:

- not every retrieval miss is the same kind of miss

## Natural Opening

The learner should not meet this lesson as a taxonomy first.

Open it more naturally:

- "I changed something and it still missed"
- "I made reranking stronger and it still missed"
- "I chunked differently and the results moved"

Then use those moments to introduce the three failure classes.

## What The Learner Should Be Able To Do

After this lesson, the learner should be able to:

- distinguish scorer failures from retrieval-unit failures
- distinguish retrieval-unit failures from candidate-stage failures
- explain why reranking cannot recover a document outside the shortlist
- choose a next action that matches the actual failure mode

## Primitives Introduced

- exact `search(...)`
- `exact_full_scan_search_plan(...)`
- `document_proxy_search_plan(...)`
- candidate window `candidate_k`
- retrieval-unit grouping through `LateDocuments`

## Evidence Map

| Failure pattern | Status | Evidence |
| --- | --- | --- |
| partial-overlap distractor beats evidence under compressed matching | Verified locally | [lesson_01_rag_retrieval_debugging.md](lesson_01_rag_retrieval_debugging.md), [../../../python/tests/test_course_rag_debugging_smoke.py](../../../python/tests/test_course_rag_debugging_smoke.py) |
| evidence split across retrieval units creates complementary partial hits | Verified locally | [../../../python/tests/test_course_failure_patterns_smoke.py](../../../python/tests/test_course_failure_patterns_smoke.py) |
| compressed stage 1 can miss the oracle document entirely, and reranking cannot recover it | Verified locally for the toy case; supported on a defined surface more broadly | [../../../python/tests/test_course_failure_patterns_smoke.py](../../../python/tests/test_course_failure_patterns_smoke.py), [../../hard_recall_evaluation.md](../../hard_recall_evaluation.md) |
| long late-evidence documents are much harsher on budgeted stage-1 sidecars | Supported on a defined surface | [../../traces/2026-04-13_benchmark_ladder_and_long_document_hard_recall.md](../../traces/2026-04-13_benchmark_ladder_and_long_document_hard_recall.md) |

## Lesson Structure

1. Show one visible miss caused by compressed matching.
2. Show one visible miss caused by the retrieval unit itself.
3. Show one visible miss caused by the candidate window.
4. Show that each miss calls for a different next move.

This order is deliberate.
It prevents the common mistake of reaching for "better reranking" too early.

It also matches how users naturally reason:

- they first notice a miss
- then they try the most obvious fix
- only then are they ready to hear why the fix was mismatched to the failure

## Rerun Commands

```bash
PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_course_rag_debugging_smoke -v
PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_course_failure_patterns_smoke -v
```

## Where This Does Not Yet Generalize

This lesson does not yet provide:

- a real judged dataset notebook for each failure mode
- a broad quantitative prevalence study of which failure class dominates in
  production

It does provide:

- one clean deterministic example for three different failure classes
- one benchmark-backed route for the broader stage-1 recall claim

The teaching point should therefore be:

- different misses deserve different fixes

not:

- the course has now classified every retrieval failure you will ever see
