# Mini Lab 3: Long Documents Are A Harsher Regime

## Problem

Many users describe the failure like this:

- "short snippets are okay, but long documents are where it breaks"

This mini lab does not try to prove a universal law about long documents.
It gives a measured family-level surface plus one readable query example.

One boundary matters here:

- this mini lab is about the current stored late-interaction artifacts
- it is not yet a benchmark of modern raw-text chunk sizes such as
  `512` / `800` / `1200` tokens

## Complaint Surface

The learner-facing version should sound like:

- "the easy baseline looks much less trustworthy once the retrieval units are
  long, dense, and multi-topic"

## Why This Surface

The locally cached judged slices already contain several longer-document
families. They are not identical, and that difference matters.

Using more than one slice is justified here because the complaint itself is a
family claim, not just a single-query complaint.

## Measured Local Family Surface

On `2026-04-19`, I reran exact late interaction, one-vector-per-document, and a
budgeted document-proxy path on several cached judged slices.

The proxy configuration was:

- `query_vector_budget=8`
- `document_vector_budget=8`
- `candidate_k = final_k`

| Slice | Primary metric | Mean doc vectors | Max doc vectors | Exact | One vector | Proxy budget 8 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `lemb_narrativeqa_real_subset` | `nDCG@10` | `180.00` | `180` | `0.3750` | `0.2272` | `0.1250` |
| `browsecomp_plus_real_subset` | `nDCG@10` | `175.07` | `180` | `0.2623` | `0.1024` | `0.1297` |
| `legal_rag_bench_real_subset` | `MRR@10` | `156.88` | `180` | `0.1667` | `0.0125` | `0.1250` |
| `r2med_biology_real_subset` | `nDCG@10` | `129.60` | `180` | `0.8646` | `0.7955` | `0.7267` |

Why this table is useful:

- `NarrativeQA`, `BrowseComp+`, and `LegalRAGBench` all show clear degradation
  from the simpler paths
- `R2Med-Biology` still shows a gap, but a noticeably smaller one

That is the epistemically safer lesson:

- longer, denser slices are often harsher
- but not every long-ish slice breaks equally

It is **not** yet sufficient to claim:

- "therefore a particular production chunk size is right"

## Human-Readable Query Example

From `lemb_narrativeqa_real_subset`, one query asks:

```text
Which Secret Service agents allows the terrorists to board Air Force One?
```

Its judged relevant document is:

- `doc_157`

Measured local behavior on `2026-04-19`:

| Path | Query-level `nDCG@10` | Query-level recall@10 | What happened |
| --- | ---: | ---: | --- |
| exact late interaction | `1.0000` | `1.0000` | the judged script page is ranked first |
| one vector per document | `0.3869` | `1.0000` | the judged page slips to rank 5 |
| proxy budget 8 | `0.0000` | `0.0000` | the judged page never enters the candidate set |

Additional fact:

- the judged relevant document for this query has `180` vectors, which is the
  maximum document length in this slice

So the local teaching point is not:

- "long documents are impossible"

It is:

- "once the retrieval unit is long and dense, simple paths can lose a lot more
  structure, and you should measure that explicitly"

## What The Learner Should Be Able To Do

After this mini lab, the learner should be able to:

- treat long-document complaints as a distinct stress regime
- compare exact and simplified paths with vector counts visible
- avoid teaching a universal chunking or compression slogan from one workload

## Rerun Hook

This mini lab is kept honest by:

```bash
PYTHONPATH=python ./.venv/bin/python -m unittest \
  python.tests.test_course_complaint_mini_labs.CourseComplaintMiniLabTests.test_long_document_family_surface_is_harsher_but_not_uniform \
  -v
```

## Where This Does Not Yet Generalize

This mini lab does not prove:

- that document length alone explains every miss
- that one-vector or proxy baselines are always bad on long documents
- that these four cached slices represent the whole long-document frontier

What it does prove locally is narrower:

- several longer-document judged slices are harsher for simplified retrieval
  paths than the lighter `r2med_biology_real_subset`, and the effect is visible
  on at least one readable NarrativeQA query

What still needs a separate reproduction:

- a raw-text chunk-size sweep in a realistic regime such as
  `256 / 512 / 800 / 1200` tokens with overlap made explicit
