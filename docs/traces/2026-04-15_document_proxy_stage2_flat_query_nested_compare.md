# Document Proxy Stage-2 Flat-Query Nested Compare

## Claim

After the candidate-window flattening experiment failed end to end, the next
more surgical stage-2 hypothesis was:

- keep the current nested document layout
- flatten the query once into `FlatQueryDim128`
- score the existing resolved candidate window with a flat-query dim128 kernel

This claim is narrower than the rejected flat-window path because it removes the
document repack cost entirely. The only new setup cost is query flattening.

## Why this check was justified

The earlier candidate-window trace established:

- scorer-only flat kernels can be competitive
- candidate-window flattening loses because its build cost is too large

That leaves a natural follow-up:

- if query flattening is cheap, maybe a flat-query specialization on the
  existing nested exact scorer can keep the gain without reintroducing build
  cost

That needed direct measurement. It was not safe to infer the result from the
hybrid-flat benchmark.

## Benchmark seam

Added:

- `benchmarks/profile_document_proxy_stage2_flat_query_nested_compare_browsecomp_gold.mojo`

The benchmark:

- precomputes candidate sets and resolved windows for the current
  `document_proxy` plan
- builds `FlatQueryDim128` for each query
- validates score equivalence against the current nested stage-2 scorer
- measures:
  - current nested stage-2 scoring
  - `build_flat_query_dim128`
  - prebuilt flat-query nested stage-2 scoring
  - `build_flat_query + score`

Measured candidate-window shape:

- mean candidate documents: `40.0`
- mean candidate vectors: `6974.75`

## Commands

```bash
pixi run mojo -I . benchmarks/profile_document_proxy_stage2_flat_query_nested_compare_browsecomp_gold.mojo
bash scripts/run_bench_quiet.sh --repeats 2 --timeout-seconds 20 --force -- pixi run mojo -I . benchmarks/profile_document_proxy_stage2_flat_query_nested_compare_browsecomp_gold.mojo
```

Artifacts:

- `.cache/kayak/profile_document_proxy_stage2_flat_query_nested_compare_browsecomp_gold.tsv`
- `.cache/kayak/bench_quiet/20260415T074917Z/`

## Correctness result

The benchmark compared every stage-2 score between:

- current `score_resolved_candidate_window_for_cpu`
- the benchmark-local flat-query nested scorer on the same resolved windows

No mismatches were observed on the gold slice. So this flat-query nested
specialization preserves exact scores for the tested workload.

## Timing result

As with the other runs today, the quiet wrapper could not obtain a quiet host
and had to force through both repeats under heavy unrelated system load.

Because of that, I treat the result conservatively and rely on direction plus
cross-run consistency, not on tiny decimal differences.

Two-run means:

- current nested stage-2 score: `0.000661907989 s`
- `build_flat_query_dim128`: `0.000002077220 s`
- prebuilt flat-query nested score: `0.000681858071 s`
- `build_flat_query + nested score`: `0.000691049755 s`

Derived ratios versus the current nested stage-2 scorer:

- prebuilt flat-query nested score: `1.0301x`
  This is about `3.0%` slower.
- `build_flat_query + nested score`: `1.0440x`
  This is about `4.4%` slower.

Per-run direction was also stable:

- run 1:
  - current: `0.000650434934 s`
  - prebuilt flat-query nested: `0.000679267893 s`
  - build + score: `0.000688785061 s`
- run 2:
  - current: `0.000673381045 s`
  - prebuilt flat-query nested: `0.000684448248 s`
  - build + score: `0.000693314449 s`

Two details matter:

- query flattening itself is tiny on this slice, about `2.1e-06 s`
- the flat-query scorer still lost before that cost was even included

That means the problem is not setup overhead. The scorer kernel itself did not
beat the current nested path.

## Interpretation

What this validates:

- flattening the query is essentially free at this stage-2 shape
- score correctness is preserved

What this debunks:

- flat-query specialization alone is not a stage-2 win here
- the current nested scorer is still better than this particular alternative

This result is important because it rules out a tempting false narrative:

- the rejected candidate-window path did **not** fail only because query
  flattening was too expensive
- it failed mainly because candidate-window build was expensive, and this
  benchmark shows that the flat-query nested scorer is not better either

## Decision

- keep `kayak/scoring/maxsim.mojo` and `kayak/planning/exact_stage.mojo`
  unchanged for this idea
- keep the benchmark seam as evidence
- do **not** ship a flat-query nested stage-2 specialization

## Next justified target

The remaining stage-2 search space is now mostly kernel-level, not policy-level.
The next credible experiment should be a genuinely different exact scorer
microkernel, such as a tiled multi-query dim128 kernel that reduces repeated
document-token loads without requiring candidate-window repacking.
