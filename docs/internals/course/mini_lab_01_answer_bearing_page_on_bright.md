# Mini Lab 1: The Answer-Bearing Page Loses To Related Material

## Problem

Many users describe the failure like this:

- "it retrieves related stuff, but not the page that actually answers"

This mini lab gives one judged-slice reproduction of that complaint on a real
Python help query.

## Complaint Surface

The learner-facing version should sound like:

- "I asked a pandas question, but it pulled in nearby technical material
  instead of the answer-bearing page"

That is more recognizable than:

- "today we will compare document interaction operators"

## Why This Slice

`bright_stackoverflow_real_subset` is a good complaint-first bridge because:

- the query language looks like a real debugging question
- the judged relevant documents are concrete documentation pages
- the miss is easy to read in top-hit form, not only as a metric table

## Measured Local Result

On `2026-04-19`, I reran this query from the cached
`bright_stackoverflow_real_subset` task JSON:

```text
I have below scenario where list str columns need to be merged with the dataframe.
...
Can only merge Series or DataFrame objects, a <class 'list'> was passed.
Please help.
```

Judged relevant documents:

- `Python_pandas_functions_with_style/General_Function_5_1.txt`
- `Python_pandas_functions_with_style/General_Function_5_2.txt`

Comparison on the same token vectors:

| Path | Query-level `nDCG@10` | Query-level recall@10 | What happened |
| --- | ---: | ---: | --- |
| exact late interaction | `0.7904` | `1.0000` | one judged pandas page ranks first and the second judged page stays in the top 10 |
| one vector per document | `0.0000` | `0.0000` | the top 10 contains no judged answer-bearing page |

Top hits:

- exact top 5:
  - `Python_pandas_functions_with_style/General_Function_5_2.txt`
  - `R_base_tools/strings.html7_78_0.txt`
  - `R_base_tools/strings.html7_108_0.txt`
  - `Python_pandas_functions/Series_286_6.txt`
  - `R_base_tools/strings.html6_16_0.txt`
- one-vector top 5:
  - `R_base_tools/stringr.html4_30_0.txt`
  - `R_base_tools/stringr.html4_8_0.txt`
  - `R_base_tools/strings.html7_131_0.txt`
  - `R_base_tools/strings.html7_68_0.txt`
  - `pytorch_torch_tensor_functions/pytorch_torch_tensor_functions_294_0.txt`

Human-readable interpretation:

- the exact path surfaces a judged pandas combine/merge page at rank 1
- the one-vector baseline drifts into non-answer-bearing R string pages and a
  PyTorch tensor page

That is the complaint in ordinary language:

- "the system found technical material, but not the page that actually answers
  the question"

## What The Learner Should Be Able To Do

After this mini lab, the learner should be able to:

- inspect a real judged query where exact late interaction keeps the
  answer-bearing page visible
- compare that against a one-vector baseline without changing the encoder
- recognize that "technical looking results" and "answer-bearing results" are
  not the same thing

## Rerun Hook

This mini lab is kept honest by:

```bash
PYTHONPATH=python ./.venv/bin/python -m unittest \
  python.tests.test_course_complaint_mini_labs.CourseComplaintMiniLabTests.test_bright_stackoverflow_answer_bearing_page_beats_one_vector \
  -v
```

## Where This Does Not Yet Generalize

This mini lab does not prove:

- that exact late interaction will always rank the judged page first
- that every dense or compressed baseline fails in the same way
- that one query from one judged slice should decide the whole product story

What it does prove locally is narrower:

- on one real judged Python help query, exact late interaction preserves the
  answer-bearing page and the one-vector baseline does not
