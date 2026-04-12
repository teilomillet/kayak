# 2026-04-12 Synthetic Hard-Recall Stage-Aware Benchmark

## Goal

Implement the selected next harder family from
[docs/harder_recall_benchmark_selection.md](../harder_recall_benchmark_selection.md):

- a scalable synthetic conjunction-style hard-recall family
- explicit query/document vector counts
- exact-reference candidate recall reporting

This trace records the first measured result for that family.

## Why This Family

The earlier synthetic single-core scale fixture was useful for asymptotic
latency control, but it was too easy for the main hard-recall question.

The new family is stricter:
- each query specifies one value per shared attribute slot
- the corpus includes the Cartesian product of those slot values
- each document therefore shares many partial overlaps with many queries
- query-specific adversarial near-miss documents repeat matched slot values many
  times while inserting one exact mismatch sentinel

Reason:
- exact MaxSim should still prefer the true full conjunction
- stage-1 approximations that compress token structure should pay a measurable
  recall cost before exact reranking can recover

## Implementation

Added:
- [kayak/benchmarks/synthetic_hard_recall_fixture.mojo](../../kayak/benchmarks/synthetic_hard_recall_fixture.mojo)
- [benchmarks/synthetic_hard_recall_stage_aware.mojo](../../benchmarks/synthetic_hard_recall_stage_aware.mojo)
- [tests/test_synthetic_hard_recall_fixture.mojo](../../tests/test_synthetic_hard_recall_fixture.mojo)

The regression test also verifies that exact reranking recovers full recall
when the candidate set fully covers the exact oracle.

The current synthetic benchmark entrypoint now evaluates:
- `exact_full_scan`
- `document_proxy`
- `centroid_postings`
- `centroid_postings_flat`
- `centroid_heads`
- `centroid_postings_head`
- `centroid_postings_head_auto`
- `centroid_postings_blockmax`
- `centroid_postings_imputed`

## Command

```bash
pixi run test_synthetic_hard_recall_fixture
pixi run bench_synthetic_hard_recall_stage_aware_raw
```

Artifact:

```text
.cache/kayak/synthetic_hard_recall_stage_aware_search.json
```

## Profiles

The first measured family uses two profiles:

1. `slots6_values3_docs1530`
   - `slot_count = 6`
   - `values_per_slot = 3`
   - `filler_vector_count = 24`
   - nominal document vectors = `30`
   - base Cartesian corpus = `1458` documents
   - adversarial near-miss docs = `72`
   - total documents = `1530`

2. `slots6_values4_docs8288`
   - `slot_count = 6`
   - `values_per_slot = 4`
   - `filler_vector_count = 24`
   - nominal document vectors = `30`
   - base Cartesian corpus = `8192` documents
   - adversarial near-miss docs = `96`
   - total documents = `8288`

Every query keeps `final_k = 2`.

## Measured Result

### `slots6_values3_docs1530`

- exact full scan:
  - `candidate_k = 2`
  - `mean_candidate_recall_at_final_k = 1.0`
  - `primary_value = 1.0`
  - `mean_search_seconds = 0.000545`
- `document_proxy`, `centroid_postings`, and `centroid_postings_flat`:
  - stay at `0.0` candidate recall through `candidate_k = 16`
  - reach `0.25` at `candidate_k = 32`
  - recover to `1.0` only at `candidate_k = 64`
- `centroid_postings_imputed`:
  - also stays at `0.0` through `candidate_k = 16`
  - reaches `0.25` at `candidate_k = 32`
  - recovers to `1.0` at `candidate_k = 64`
  - but is slower than the non-imputed full-recall paths on the same slice
- `centroid_heads`:
  - stays at `0.0` through `candidate_k = 32`
  - reaches only `0.0833` at `candidate_k = 64` and `128`
- `centroid_postings_head` and `centroid_postings_head_auto`:
  - stay at `0.0` through `candidate_k = 32`
  - reach only `0.0833` at `candidate_k = 64` and `128`
- `centroid_postings_blockmax`:
  - stays at `0.0` through `candidate_k = 16`
  - reaches `0.25` only at `candidate_k = 128`

### `slots6_values4_docs8288`

- exact full scan:
  - `candidate_k = 2`
  - `mean_candidate_recall_at_final_k = 1.0`
  - `primary_value = 1.0`
  - `mean_search_seconds = 0.002371`
- `document_proxy`, `centroid_postings`, and `centroid_postings_flat`:
  - stay at `0.0` candidate recall through `candidate_k = 64`
  - recover to `1.0` only at `candidate_k = 128`
- `centroid_postings_imputed`:
  - also stays at `0.0` through `candidate_k = 64`
  - recovers to `1.0` only at `candidate_k = 128`
  - but is the slowest of the full-recall plans on this slice
- `centroid_heads`:
  - stays at `0.0` through `candidate_k = 32`
  - reaches only `0.0625` at `candidate_k = 64` and `128`
- `centroid_postings_head` and `centroid_postings_head_auto`:
  - match the same weak `0.0625` ceiling at `candidate_k = 64` and `128`
- `centroid_postings_blockmax`:
  - stays at `0.0` through `candidate_k = 64`
  - reaches only `0.0625` at `candidate_k = 128`

## What This Verifies

Verified locally:
- Kayak now has a scalable synthetic hard-recall family beyond the current
  small public slices
- that family keeps query vectors, document vectors, and candidate windows
  explicit
- exact-reference candidate recall and final judged quality move together in a
  meaningful way on this family
- stage-1 approximations that look comfortable on the small public slices can
  require materially larger `candidate_k` here

The most concrete frontier result is:
- on the `1530`-document synthetic slice, `document_proxy`,
  `centroid_postings`, and `centroid_postings_flat` need `candidate_k = 64`
  to recover full recall
- `centroid_postings_imputed` matches that same recovery point, but at higher
  latency
- on the `8288`-document synthetic slice, those same plans need
  `candidate_k = 128`
- `centroid_postings_imputed` also needs `candidate_k = 128` there and remains
  slower than the non-imputed full-recall plans
- the head-capped and blockmax variants do not beat that recovery frontier on
  the measured profiles

## What This Does Not Claim

This trace does not claim:
- that this synthetic family replaces a larger public hard-recall corpus
- that these two profiles are enough to establish asymptotic scaling laws
- that the current centroid-heads implementation is the best possible native
  stage

It only establishes the next honest state:
- the synthetic hard-recall family is real
- it is measurably harder than the current small public slices for the tested
  stage-1 plans
- it now belongs in the repo's benchmark surface
