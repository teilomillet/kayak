# Lesson 3 Draft: Real Judged-Slice Diagnosis

## Problem

The toy lessons explain the primitives cleanly, but learners still need one
question answered on non-toy data:

- does this diagnosis workflow hold on a real judged slice?

This lesson answers that question on cached local task JSONs that already exist
in the repository.

## Natural Opening

Do not lead with the cross-slice table.

Lead with one concrete miss on `LIMIT-small` that the learner can inspect.

The intended feeling is:

- "this looks like the kind of retrieval miss I actually care about"
- "one path kept both relevant documents visible and the other path did not"

Only after that should the lesson widen to the cross-slice summary.

## Scope

This lesson is intentionally about:

- exact late interaction versus a compressed single-proxy candidate path
- real judged subsets that are already cached locally
- diagnosis, not leaderboard claims

It is not about:

- broad field-wide statements about every dense retriever
- service latency
- one final product recommendation for every workload

## Why `LIMIT-small` Is The Right Teaching Slice

The repo's benchmark rationale already marks `LIMIT-small` as important because
it is:

- officially small
- explicitly adversarial to single-vector retrieval
- aligned with `nDCG@10`

Source:
- [../../benchmark_rationale.md](../../benchmark_rationale.md)

That makes it a good real-data bridge between the toy lessons and the harder
benchmark notes.

## Measured Local Result

On `2026-04-19`, I reran a local exact-versus-proxy comparison on the cached
task JSONs already present in `.cache/kayak/`.

Method:

- build one packed `LateIndex` from each task JSON
- run exact full-scan late interaction with the public Python search-plan API
- run `document_proxy_search_plan(...)` with:
  - full document-proxy averaging
  - one narrower `query_vector_budget=8`, `document_vector_budget=8` variant
- score each ranked result with the task's own judged metrics

This is not a new benchmark suite.
It is a diagnosis-oriented local comparison on existing judged subsets.

That distinction should stay mostly backstage.
Onstage, the learner should experience this as:

- try one real slice
- notice the miss
- then look at the wider pattern

### Cross-slice summary

| Slice | Primary metric | Exact | Proxy full-budget | Proxy budget 8 | Exact recall@k | Proxy full recall@k | Proxy 8 recall@k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `limit_small_real_subset` | `nDCG@10` | `0.9674` | `0.7186` | `0.1341` | `1.0000` | `0.6719` | `0.1094` |
| `bright_stackoverflow_real_subset` | `nDCG@10` | `0.2914` | `0.1007` | `0.1784` | `0.5139` | `0.1389` | `0.2153` |
| `legal_rag_bench_real_subset` | `MRR@10` | `0.1667` | `0.0125` | `0.1250` | `0.2500` | `0.1250` | `0.1250` |
| `r2med_biology_real_subset` | `nDCG@10` | `0.8646` | `0.8317` | `0.7267` | `0.8109` | `0.7585` | `0.6252` |

Interpretation:

- exact late interaction is stronger on all four cached slices in this local
  comparison
- `LIMIT-small` is the clearest teaching case because the exact-versus-proxy
  gap is large and the slice is small enough to inspect by hand
- the `budget=8` proxy is **not** monotonically worse than the full-budget
  proxy on every slice

That last point matters.
It means:

- compressed proxy behavior is heuristic
- "more proxy input always helps" is not an epistemically safe lesson
- the right takeaway is to measure proxy faithfulness against the exact path

## Human-Readable `LIMIT-small` Example

One query from the cached `LIMIT-small` task is:

```text
Who likes Joshua Trees?
```

Its judged relevant documents are:

- `Geneva Durben`
- `Dorathea Bastress`

Under exact late interaction on the cached task vectors, the top hits begin
with:

- `Geneva Durben`
- `Dorathea Bastress`

Under the full-budget document-proxy path, the top hits begin with:

- `Geneva Durben`
- `Darwyn Raio`
- `Nathaniel Robens`
- `Chaney Gertman`
- `Ovid Rahm`

The practical teaching point is:

- one relevant document survives
- another relevant document drops out of the top ranked set
- a compressed document proxy is already losing judged recall on a small real
  slice that was selected specifically because shallow single-vector matching is
  not enough

The notebook for this lesson prints the underlying document texts for this
query so the learner can inspect the miss directly.

That direct inspection matters more pedagogically than the table alone.
It makes the lesson feel like a real debugging session instead of a benchmark recital.

## What The Learner Should Be Able To Do

After this lesson, the learner should be able to:

- run a real judged exact-versus-proxy comparison on a cached task JSON
- recognize when a compressed proxy path is losing judged quality
- avoid teaching themselves the wrong lesson from a convenient heuristic win
- use Kayak as the exact diagnosis surface before discussing acceleration

## Rerun Command

This lesson's notebook is:

- [notebooks/real_slice_proxy_diagnosis.ipynb](notebooks/real_slice_proxy_diagnosis.ipynb)

If the `LIMIT-small` task JSON is missing locally, rebuild it with:

```bash
PYTHONPATH=python ./.venv/bin/python python/scripts/build_task_json.py --dataset-key limit_small
```

## Where This Does Not Yet Generalize

This lesson still does not prove:

- that every dense or compressed retriever fails in the same way
- that the current proxy settings are the best non-exact baseline
- that `LIMIT-small` alone should anchor the whole course

What it does give us:

- one real judged slice with a clean exact-versus-proxy gap
- one cross-slice table showing the pattern is not isolated to a single toy
- one explicit caution that heuristic proxy settings do not move monotonically

So the learner-facing takeaway should be:

- "I can now check this on real judged data"

not:

- "I have now proved a universal law about compressed retrieval"
