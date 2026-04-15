# Document Proxy Stage-2 Candidate-Window Flat Compare

## Claim

The next untested stage-2 idea after the earlier exact-rerank investigations
was:

- build a dim128 flat index only for the resolved candidate window
- score that flat window with the existing hybrid-flat kernels
- adopt it only if `build + score` beats the current nested exact stage-2 path

This trace records the check that validates or debunks that idea on the
`browsecomp_plus_gold` `document_proxy` operating point.

## Why this check was needed

Previous stage-2 evidence had already established two things:

- `document_proxy` stage-2 on the gold slice is scorer-bound
- the existing full-segment hybrid-flat exact path is slower than the current
  nested scorer on that slice

What had **not** been tested was a narrower hypothesis:

- maybe the hybrid-flat kernels are good enough if we flatten only the
  candidate window instead of the full segment

That is a different claim, and it needed a direct benchmark before any
production edit.

## Benchmark seam

Added:

- `benchmarks/profile_document_proxy_stage2_candidate_flat_compare_browsecomp_gold.mojo`

The benchmark does four explicit things:

- precomputes candidate sets and resolved windows for the current
  `document_proxy` plan
- builds a `HybridFlatDim128Index` directly from each resolved candidate window
- validates score equivalence against
  `score_resolved_candidate_window_for_cpu`
- measures current score, flat-window build, flat-window score, and the honest
  combined `build + score` paths

The measured candidate-window shape on this slice was:

- mean candidate documents: `40.0`
- mean candidate vectors: `6974.75`

## Commands

```bash
pixi run mojo -I . benchmarks/profile_document_proxy_stage2_candidate_flat_compare_browsecomp_gold.mojo
bash scripts/run_bench_quiet.sh --repeats 2 --timeout-seconds 20 --force -- pixi run mojo -I . benchmarks/profile_document_proxy_stage2_candidate_flat_compare_browsecomp_gold.mojo
```

Artifacts:

- `.cache/kayak/profile_document_proxy_stage2_candidate_flat_compare_browsecomp_gold.tsv`
- `.cache/kayak/bench_quiet/20260415T072740Z/`

## Correctness result

Before timing, the benchmark compared stage-2 scores for every query and every
candidate document between:

- current nested exact stage-2 scoring
- flat candidate-window scoring with nested query
- flat candidate-window scoring with flat query

No mismatches were observed on the gold slice, so the candidate-window flat
builder preserves exact scores for this workload.

## Timing result

The quiet-wrapper run could not obtain a quiet host and had to force through in
both repeats. The machine was heavily loaded by unrelated work, including other
Mojo benchmarks, `soma_http`, Zed, and Helium renderer processes.

That matters, so the result below is stated conservatively:

- I use the average across the two forced wrapper runs
- I rely on the size and direction of the gap, not on tiny differences

Two-run means:

- current nested stage-2 score: `0.000880197457 s`
- build candidate-window flat index: `0.000364698267 s`
- prebuilt flat candidate-window score: `0.000771498897 s`
- prebuilt flat candidate-window score with flat query: `0.000753805298 s`
- build candidate-window flat index and score: `0.001141822382 s`
- build flat query, build candidate-window flat index, and score:
  `0.001199634360 s`

Derived ratios versus the current nested stage-2 scorer:

- prebuilt flat candidate-window score: `0.8765x`
  This is about `12.3%` faster scorer-only.
- prebuilt flat candidate-window score with flat query: `0.8564x`
  This is about `14.4%` faster scorer-only.
- build candidate-window flat index and score: `1.2972x`
  This is about `29.7%` slower end to end.
- build flat query, build candidate-window flat index, and score: `1.3629x`
  This is about `36.3%` slower end to end.

Per-run combined timings were also directionally stable:

- run 1: current `0.000910570068 s`, build+score `0.001111486341 s`
- run 2: current `0.000849824846 s`, build+score `0.001172158423 s`

That consistency matters more than the exact decimals: the build-inclusive path
lost in both wrapper runs.

## Interpretation

What this validates:

- the flat candidate-window idea is not wrong at the kernel level
- scorer-only, the flat window can beat the current nested scorer on this slice

What this debunks:

- flattening the candidate window inside stage-2 is not a production win here
- the window-build tax is larger than the scorer gain

The key distinction is:

- `score flat window` is a kernel result
- `build flat window + score flat window` is the real retrieval-stage result

Only the second one matters for production stage-2.

## Decision

- keep `kayak/planning/exact_stage.mojo` unchanged for this idea
- keep the benchmark seam because it narrows the remaining stage-2 search space
- do **not** ship candidate-window flattening as an optimization

## Next justified target

This benchmark leaves one clear follow-up hypothesis:

- keep the nested document layout
- test whether a flat-query specialization can capture some of the scorer-side
  gain without paying the candidate-window flattening cost

That is justified because the scorer-only flat-query path was the fastest
variant measured here, while the build tax was the part that killed the full
proposal.
