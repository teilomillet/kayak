# Exact Stage Materialization Breakdown on BrowseComp-Plus Gold

## Claim

After the workspace search breakdown showed that stage2 dominates total search
time on `browsecomp_plus_gold`, the next concrete question was:

- inside exact stage-2 rerank, is the dominant cost the scoring kernel itself,
  or the per-query materialization of a packed candidate window

This trace measures that split directly.

## Code inspection that motivated the benchmark

I inspected [exact_stage.mojo](/Users/teilomillet/Code/kayak/kayak/planning/exact_stage.mojo).

The current stage2 path does the following for every rerank:

- `find_document_index_in_segment(...)`
  - linearly searches `doc_ids` inside the segment for each candidate hit
- `build_encoded_document_from_segment(...)`
  - copies every token vector for every candidate document into nested lists
- `materialize_candidate_index(...)`
  - repacks those copied documents via `pack_documents(...)`
- `exact_rerank_candidates_for_plan(...)`
  - runs `backend.score_all(...)` on the freshly materialized packed candidate
    index and then rebuilds top-k results

That made a materialization-vs-scoring benchmark the right next check.

## Benchmark surface

Added:

- `benchmarks/profile_exact_stage_materialization_browsecomp_gold.mojo`

Measured plans:

- `centroid_postings_imputed`
- `centroid_postings_imputed_flat`

Measured benchmark kinds:

- `materialize_candidate_index`
- `score_materialized_index`
- `exact_rerank_candidates_for_plan`

Artifact:

- `.cache/kayak/profile_exact_stage_materialization_browsecomp_gold.tsv`

Quiet-wrapper artifact:

- `.cache/kayak/bench_quiet/20260414T204206Z`

## Commands run

```bash
pixi run mojo -I . benchmarks/profile_exact_stage_materialization_browsecomp_gold.mojo
bash scripts/run_bench_quiet.sh --timeout-seconds 20 --force --repeats 1 -- pixi run mojo -I . benchmarks/profile_exact_stage_materialization_browsecomp_gold.mojo
```

The quiet wrapper again timed out waiting for a quiet host and force-ran the
benchmark. Pre-run competing load samples ranged from roughly `456.70` to
`662.70` aggregate `%CPU` from other processes, so the wrapped run remains
host-contended.

## Wrapped result

Measured means:

- `centroid_postings_imputed`
  - `materialize_candidate_index`: `1.4190566037735849e-03 s`
  - `score_materialized_index`: `6.393608937576074e-04 s`
  - `exact_rerank_candidates_for_plan`: `2.661458107759732e-03 s`
- `centroid_postings_imputed_flat`
  - `materialize_candidate_index`: `1.3965075987841945e-03 s`
  - `score_materialized_index`: `6.780131841443317e-04 s`
  - `exact_rerank_candidates_for_plan`: `2.508979498861048e-03 s`

Derived shares:

- `centroid_postings_imputed`
  - materialization share of exact rerank: `0.533188`
  - scoring share of exact rerank: `0.240230`
  - materialization vs scoring ratio: `2.219492`
  - residual share: `0.226583`
- `centroid_postings_imputed_flat`
  - materialization share of exact rerank: `0.556604`
  - scoring share of exact rerank: `0.270235`
  - materialization vs scoring ratio: `2.059706`
  - residual share: `0.173162`

Here, residual share means:

- `exact_rerank_candidates_for_plan - materialize_candidate_index - score_materialized_index`

which covers the remaining top-k rebuilding and other fixed overhead.

## Direct-run context

The exploratory direct run already showed the same order:

- `centroid_postings_imputed`
  - materialize: `1.4076959888493106e-03 s`
  - score: `4.3320739231023466e-04 s`
  - exact rerank: `2.0204572938689216e-03 s`
- `centroid_postings_imputed_flat`
  - materialize: `1.3931031023306066e-03 s`
  - score: `4.0882809083263245e-04 s`
  - exact rerank: `2.1369819221967964e-03 s`

So the main conclusion does not depend on the wrapped run alone.

## Interpretation

What is verified:

- on this heavier public slice, candidate-window materialization is the largest
  single measured component inside exact rerank
- materialization is roughly `2.06x` to `2.22x` the cost of scoring the already
  materialized packed candidate window
- scoring is important, but it is not currently the dominant subcomponent in
  this stage2 path

What this means for optimization priority:

- the next credible end-to-end win is more likely to come from removing or
  shrinking stage2 candidate-window materialization than from tuning the
  current score kernel in isolation

The code and benchmark together point at the same target:

- avoid repeated `doc_id` lookup
- avoid rebuilding nested `EncodedDocument` lists
- avoid repacking candidate windows with `pack_documents(...)` on every query

## Bottom line

The strongest measured next target is no longer stage-1 workspace reuse.
It is stage2 exact rerank materialization in
[exact_stage.mojo](/Users/teilomillet/Code/kayak/kayak/planning/exact_stage.mojo).

If the goal is a visible whole-search speedup on heavier slices, the next
implementation step should attack that repack path directly.
