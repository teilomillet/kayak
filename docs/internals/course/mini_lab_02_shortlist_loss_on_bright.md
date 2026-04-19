# Mini Lab 2: Widening `k` Helps Because Stage 1 Dropped The Answer

## Problem

Many users report the failure like this:

- "if I widen `k`, the right page sometimes appears"
- "reranking did not help"

This mini lab turns that into one explicit judged-slice diagnosis.

## Complaint Surface

The learner-facing version should sound like:

- "the answer page is not bad at reranking time; it is missing before reranking
  even starts"

## Why This Slice

The same `bright_stackoverflow_real_subset` pandas query from Mini Lab 1 is a
good shortlist example because:

- the complaint is realistic
- the judged relevant page is easy to name
- widening the candidate window changes the candidate set in a directly
  inspectable way

## Measured Local Result

On `2026-04-19`, I reran the same query with the public proxy search plan under
fixed vector budgets:

- `query_vector_budget=8`
- `document_vector_budget=8`
- `final_k=10`

Then I changed only:

- `candidate_k=10`
- `candidate_k=20`

Judged relevant documents:

- `Python_pandas_functions_with_style/General_Function_5_1.txt`
- `Python_pandas_functions_with_style/General_Function_5_2.txt`

Result summary:

| Path | Query-level `nDCG@10` | Query-level recall@10 | What stage 1 did |
| --- | ---: | ---: | --- |
| exact full scan | `0.7904` | `1.0000` | no shortlist loss |
| proxy, `candidate_k=10` | `0.0000` | `0.0000` | stage 1 excludes both judged relevant pages |
| proxy, `candidate_k=20` | `0.6131` | `0.5000` | stage 1 finally admits one judged relevant page |

Candidate-stage fact pattern:

- with `candidate_k=10`, neither judged relevant page is present in the
  candidate set
- with `candidate_k=20`,
  `Python_pandas_functions_with_style/General_Function_5_2.txt` enters the
  candidate set at position 19
- once it enters the shortlist, exact stage 2 reranks it to position 1 in the
  final hits

This is the key diagnosis:

- reranking was not the first problem
- stage 1 had already thrown the answer-bearing page away

## What The Learner Should Be Able To Do

After this mini lab, the learner should be able to:

- inspect candidate sets instead of blaming reranking by default
- verify whether widening the shortlist changes recall on the same query
- explain why "reranking did not help" can really mean "the answer never
  survived stage 1"

## Rerun Hook

This mini lab is kept honest by:

```bash
PYTHONPATH=python ./.venv/bin/python -m unittest \
  python.tests.test_course_complaint_mini_labs.CourseComplaintMiniLabTests.test_bright_stackoverflow_wider_candidate_window_recovers_answer_page \
  -v
```

## Where This Does Not Yet Generalize

This mini lab does not prove:

- that `candidate_k=20` is a generally good setting
- that widening the shortlist will always recover the oracle
- that every proxy miss is mainly a shortlist problem

What it proves locally is narrower:

- on one judged Python help query, stage 1 can exclude the answer-bearing page
  entirely, and widening the candidate set can partially recover it
